//! Pool mode: inference nodes sharing one cache ([`PooledSemanticCache`]), over any
//! [`CacheTransport`] ([`DialerTransport`] for `weirdnet-agent` channels, served by [`serve`]).
//!
//! Each entry lives on its shard's home node and one replica ([`ShardRing`]), so a lookup is one
//! call. A node keeps the shards it holds in its own [`SemanticCache`] (served to the others by
//! a [`SharedCacheHost`]) and, optionally, a small local copy (L1) of entries it fetched or wrote
//! for shards it doesn't hold.
//!
//! **A cache outage never blocks inference.** Every remote call has a timeout; a node that fails
//! or times out is skipped for a while ([`PoolConfig::down_for`]) and its replica asked instead;
//! when nothing answers, a lookup is a miss ([`PoolLookup::Miss`] with `degraded: true`). Writes
//! to the replica run in the background.

use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use serde::Serialize;
use serde::de::DeserializeOwned;
use tokio::sync::Notify;
use weirdnet_agent::remote::{Authorize, Caller, Channel, Dialer, RemoteError};

use crate::semantic::{Lookup, SemanticCache, SemanticConfig, SemanticError, normalize};
use crate::shard::{ShardPlan, ShardRing};
use crate::shared::{PROTOCOL, SharedCacheHost, SharedCacheReq, SharedCacheResp};
use crate::{AnswerStore, CachedAnswer, Key};

/// Sends one shared-cache request to a node and returns its answer.
///
/// [`DialerTransport`] does it over `weirdnet-agent` channels; tests and apps with their own
/// links implement it directly. Implementations need not time out: [`PooledSemanticCache`]
/// bounds every call.
#[async_trait]
pub trait CacheTransport<V = CachedAnswer>: Send + Sync {
    async fn call(&self, node: &str, req: SharedCacheReq<V>) -> Result<SharedCacheResp<V>, RemoteError>;
}

/// A [`CacheTransport`] over a `weirdnet-agent` [`Dialer`]: one channel on [`PROTOCOL`] per call,
/// one CBOR request, one CBOR response.
pub struct DialerTransport<D> {
    dialer: D,
}

impl<D> DialerTransport<D> {
    pub fn new(dialer: D) -> Self {
        Self { dialer }
    }

    pub fn dialer(&self) -> &D {
        &self.dialer
    }
}

#[async_trait]
impl<D, V> CacheTransport<V> for DialerTransport<D>
where
    D: Dialer,
    V: Serialize + DeserializeOwned + Send + Sync + 'static,
{
    async fn call(&self, node: &str, req: SharedCacheReq<V>) -> Result<SharedCacheResp<V>, RemoteError> {
        let mut ch = self.dialer.open(node, PROTOCOL).await?;
        ch.tx.send(req.encode()?).await?;
        match ch.rx.recv().await? {
            Some(bytes) => SharedCacheResp::decode(&bytes),
            None => Err(RemoteError::Closed),
        }
    }
}

/// Serves [`PROTOCOL`] on one channel: authorizes `caller` with `auth` first (for Weirdnet, the
/// pool's roster: `weirdnet_agent::serve::RosterAuth`), then answers requests from `host` until
/// the asker closes. A refused caller gets one [`SharedCacheResp::Error`] and the refusal.
pub async fn serve<V>(mut ch: Channel, caller: &Caller, auth: &dyn Authorize, host: &SharedCacheHost<V>) -> Result<(), RemoteError>
where
    V: Clone + Serialize + DeserializeOwned + Send + Sync,
{
    if let Err(e) = auth.authorize(caller) {
        let _ = ch.tx.send(SharedCacheResp::<V>::error(e.to_string()).encode()?).await;
        return Err(e);
    }
    while let Some(bytes) = ch.rx.recv().await? {
        ch.tx.send(host.handle_bytes(&bytes)?).await?;
    }
    Ok(())
}

/// How a [`PooledSemanticCache`] waits, retries and copies.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct PoolConfig {
    /// Longest wait for one remote lookup (a lookup asks at most the home and the replica).
    pub lookup_timeout: Duration,
    /// Longest wait for one remote write or removal.
    pub write_timeout: Duration,
    /// A node that failed or timed out is skipped this long. Zero never skips.
    pub down_for: Duration,
    /// Local copies (L1) of entries for shards this node doesn't hold; 0 keeps none.
    pub l1_entries: usize,
    /// L1 copies expire after this; `None` keeps them until evicted.
    pub l1_ttl: Option<Duration>,
    /// When the home misses, ask the replica too (and copy a hit back to the home). Costs a
    /// second call per miss; finds entries right after a node joined and took over a shard.
    pub ask_replica_on_miss: bool,
}

impl PoolConfig {
    /// 150 ms lookups, 2 s writes, 10 s down time, 256 L1 entries for 5 minutes, home-only misses.
    pub fn new() -> Self {
        Self {
            lookup_timeout: Duration::from_millis(150),
            write_timeout: Duration::from_secs(2),
            down_for: Duration::from_secs(10),
            l1_entries: 256,
            l1_ttl: Some(Duration::from_secs(300)),
            ask_replica_on_miss: false,
        }
    }

    pub fn with_lookup_timeout(mut self, t: Duration) -> Self {
        self.lookup_timeout = t;
        self
    }

    pub fn with_write_timeout(mut self, t: Duration) -> Self {
        self.write_timeout = t;
        self
    }

    pub fn with_down_for(mut self, t: Duration) -> Self {
        self.down_for = t;
        self
    }

    /// L1 with at most `entries` (0 turns it off) expiring after `ttl`.
    pub fn with_l1(mut self, entries: usize, ttl: Option<Duration>) -> Self {
        self.l1_entries = entries;
        self.l1_ttl = ttl;
        self
    }

    pub fn with_ask_replica_on_miss(mut self, ask: bool) -> Self {
        self.ask_replica_on_miss = ask;
        self
    }
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Where a [`PoolLookup::Hit`] came from.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum HitFrom {
    /// This node's L1 copy.
    L1,
    /// This node's own shard (it is the home or the replica).
    Shard,
    /// Another node: the shard's home, or its replica.
    Peer { node: String, replica: bool },
}

/// The result of [`PooledSemanticCache::lookup`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum PoolLookup<V> {
    Hit {
        value: V,
        score: f32,
        from: HitFrom,
    },
    Miss {
        /// The best score below the threshold, when an owner reported one.
        best: Option<f32>,
        /// An owner failed, timed out or was skipped as down, so this may be a miss only because
        /// of the outage.
        degraded: bool,
    },
}

impl<V> PoolLookup<V> {
    pub fn is_hit(&self) -> bool {
        matches!(self, Self::Hit { .. })
    }

    /// The hit's score, or the best score of a miss.
    pub fn score(&self) -> Option<f32> {
        match self {
            Self::Hit { score, .. } => Some(*score),
            Self::Miss { best, .. } => *best,
        }
    }

    pub fn into_value(self) -> Option<V> {
        match self {
            Self::Hit { value, .. } => Some(value),
            Self::Miss { .. } => None,
        }
    }
}

/// Which owners took a write or removal.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct PoolWrite {
    /// Nodes that confirmed it (this node included when it holds the shard).
    pub acked: Vec<String>,
    /// Nodes it was sent to in the background (the replica); see [`PooledSemanticCache::settle`].
    pub queued: Vec<String>,
    /// Nodes that refused, failed, timed out or were skipped as down.
    pub failed: Vec<String>,
}

/// Counters since the cache was made.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct PoolStats {
    pub l1_hits: u64,
    pub shard_hits: u64,
    pub peer_hits: u64,
    pub misses: u64,
    /// Misses with an owner unreachable.
    pub degraded: u64,
    /// Remote calls that failed or were refused.
    pub peer_errors: u64,
    /// Remote calls that timed out.
    pub timeouts: u64,
    /// Owners skipped because they were marked down.
    pub skipped: u64,
}

#[derive(Default)]
struct Counters {
    l1_hits: AtomicU64,
    shard_hits: AtomicU64,
    peer_hits: AtomicU64,
    misses: AtomicU64,
    degraded: AtomicU64,
    peer_errors: AtomicU64,
    timeouts: AtomicU64,
    skipped: AtomicU64,
}

fn better(a: Option<f32>, b: Option<f32>) -> Option<f32> {
    match (a, b) {
        (Some(x), Some(y)) => Some(x.max(y)),
        (x, None) => x,
        (None, y) => y,
    }
}

fn bump(c: &AtomicU64) {
    c.fetch_add(1, Ordering::Relaxed);
}

/// The calling side's shared state, cloned into background writes.
struct Peers<V> {
    transport: Arc<dyn CacheTransport<V>>,
    down: Mutex<HashMap<String, Instant>>,
    counters: Counters,
    in_flight: AtomicUsize,
    settled: Notify,
    down_for: Duration,
}

/// Why a remote call didn't give an answer.
enum CallFailure {
    /// Down, failed or timed out: the node is marked down.
    Unreachable,
    /// The node answered with an error (refused, bad input): not marked down.
    Refused,
}

impl<V: Send + 'static> Peers<V> {
    fn is_down(&self, node: &str) -> bool {
        let mut down = self.down.lock().unwrap_or_else(|e| e.into_inner());
        match down.get(node) {
            Some(until) if *until > Instant::now() => true,
            Some(_) => {
                down.remove(node);
                false
            }
            None => false,
        }
    }

    fn mark_down(&self, node: &str) {
        if !self.down_for.is_zero() {
            self.down.lock().unwrap_or_else(|e| e.into_inner()).insert(node.to_owned(), Instant::now() + self.down_for);
        }
    }

    fn mark_up(&self, node: &str) {
        self.down.lock().unwrap_or_else(|e| e.into_inner()).remove(node);
    }

    async fn call(&self, node: &str, req: SharedCacheReq<V>, timeout: Duration) -> Result<SharedCacheResp<V>, CallFailure> {
        if self.is_down(node) {
            bump(&self.counters.skipped);
            return Err(CallFailure::Unreachable);
        }
        match tokio::time::timeout(timeout, self.transport.call(node, req)).await {
            Ok(Ok(SharedCacheResp::Error { .. })) => {
                self.mark_up(node);
                bump(&self.counters.peer_errors);
                Err(CallFailure::Refused)
            }
            Ok(Ok(resp)) => {
                self.mark_up(node);
                Ok(resp)
            }
            Ok(Err(e)) => {
                bump(&self.counters.peer_errors);
                if !e.is_request_error() {
                    self.mark_down(node);
                }
                Err(CallFailure::Unreachable)
            }
            Err(_) => {
                bump(&self.counters.timeouts);
                self.mark_down(node);
                Err(CallFailure::Unreachable)
            }
        }
    }

    /// Sends `req` to `node` in the background.
    fn spawn(self: &Arc<Self>, node: String, req: SharedCacheReq<V>, timeout: Duration)
    where
        V: Sync,
    {
        self.in_flight.fetch_add(1, Ordering::SeqCst);
        let peers = self.clone();
        tokio::spawn(async move {
            let _ = peers.call(&node, req, timeout).await;
            if peers.in_flight.fetch_sub(1, Ordering::SeqCst) == 1 {
                peers.settled.notify_waiters();
            }
        });
    }
}

/// One inference node's view of the pool's shared semantic cache (and, with
/// [`with_exact`](Self::with_exact), of its exact entries).
///
/// - [`lookup`](Self::lookup): L1, then the shard's home (this node's own cache when it is the
///   home), then the replica if the home is unreachable. Never fails because of a peer.
/// - [`insert`](Self::insert): this node's own cache when it holds the shard, the home (waited
///   for, bounded), the replica in the background, and L1.
/// - [`remove_scope`](Self::remove_scope): everywhere, every member (so entries left behind by a
///   membership change go too).
///
/// Members come from the app ([`set_nodes`](Self::set_nodes), [`join`](Self::join),
/// [`leave`](Self::leave)), for example the pool's inference nodes from the roster; every node
/// must use the same ids, and must use the same embedding model and revision. Serve
/// [`host`](Self::host) on [`PROTOCOL`] so the others can reach this node's shards.
pub struct PooledSemanticCache<V = CachedAnswer> {
    node: String,
    ring: RwLock<ShardRing>,
    shard: Arc<SemanticCache<V>>,
    l1: Option<SemanticCache<V>>,
    exact: Option<Arc<dyn AnswerStore>>,
    peers: Arc<Peers<V>>,
    config: PoolConfig,
}

impl<V> fmt::Debug for PooledSemanticCache<V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PooledSemanticCache")
            .field("node", &self.node)
            .field("ring", &*self.ring.read().unwrap_or_else(|e| e.into_inner()))
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl<V> PooledSemanticCache<V>
where
    V: Clone + Serialize + DeserializeOwned + Send + Sync + 'static,
{
    /// This node is `node` (its id in the ring), keeps the shards it holds in `shard`, and
    /// reaches the others over `transport`. The ring starts with this node alone.
    pub fn new(node: impl Into<String>, shard: Arc<SemanticCache<V>>, transport: Arc<dyn CacheTransport<V>>, config: PoolConfig) -> Self {
        let node = node.into();
        let l1 = (config.l1_entries > 0).then(|| {
            let mut c = SemanticConfig::new(config.l1_entries).with_threshold(shard.config().threshold);
            c.ttl = config.l1_ttl;
            c.dimension = shard.config().dimension;
            SemanticCache::with_config(c)
        });
        let peers = Arc::new(Peers {
            transport,
            down: Mutex::new(HashMap::new()),
            counters: Counters::default(),
            in_flight: AtomicUsize::new(0),
            settled: Notify::new(),
            down_for: config.down_for,
        });
        Self { ring: RwLock::new(ShardRing::new([node.clone()])), node, shard, l1, exact: None, peers, config }
    }

    /// Shares exact entries too, keeping this node's in `store`.
    pub fn with_exact(mut self, store: Arc<dyn AnswerStore>) -> Self {
        self.exact = Some(store);
        self
    }

    /// This node's id.
    pub fn node(&self) -> &str {
        &self.node
    }

    pub fn config(&self) -> &PoolConfig {
        &self.config
    }

    /// The shards this node holds.
    pub fn shard(&self) -> &Arc<SemanticCache<V>> {
        &self.shard
    }

    /// Serves this node's shards to the others (see [`serve`]).
    pub fn host(&self) -> SharedCacheHost<V> {
        let host = SharedCacheHost::new(self.shard.clone());
        match &self.exact {
            Some(store) => host.with_exact(store.clone()),
            None => host,
        }
    }

    /// The current members.
    pub fn ring(&self) -> ShardRing {
        self.ring.read().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// Replaces the members (usually including this node; without it, this node holds no
    /// shards and only asks).
    pub fn set_nodes<I, S>(&self, nodes: I)
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        *self.ring.write().unwrap_or_else(|e| e.into_inner()) = ShardRing::new(nodes);
    }

    /// Adds a member; `false` when it already was one.
    pub fn join(&self, node: impl Into<String>) -> bool {
        self.ring.write().unwrap_or_else(|e| e.into_inner()).insert(node)
    }

    /// Removes a member; `false` when it wasn't one.
    pub fn leave(&self, node: &str) -> bool {
        self.ring.write().unwrap_or_else(|e| e.into_inner()).remove(node)
    }

    /// The home and replica for `scope` (or an exact key) now.
    pub fn plan(&self, shard: impl AsRef<[u8]>) -> Option<ShardPlan> {
        self.ring.read().unwrap_or_else(|e| e.into_inner()).plan(shard)
    }

    /// `node` was marked down after a failure and is being skipped.
    pub fn is_down(&self, node: &str) -> bool {
        self.peers.is_down(node)
    }

    pub fn stats(&self) -> PoolStats {
        let c = &self.peers.counters;
        let get = |a: &AtomicU64| a.load(Ordering::Relaxed);
        PoolStats {
            l1_hits: get(&c.l1_hits),
            shard_hits: get(&c.shard_hits),
            peer_hits: get(&c.peer_hits),
            misses: get(&c.misses),
            degraded: get(&c.degraded),
            peer_errors: get(&c.peer_errors),
            timeouts: get(&c.timeouts),
            skipped: get(&c.skipped),
        }
    }

    /// Waits until background writes have finished (each is bounded by the write timeout).
    pub async fn settle(&self) {
        loop {
            let notified = self.peers.settled.notified();
            if self.peers.in_flight.load(Ordering::SeqCst) == 0 {
                return;
            }
            notified.await;
        }
    }

    fn miss(&self, best: Option<f32>, degraded: bool) -> PoolLookup<V> {
        bump(&self.peers.counters.misses);
        if degraded {
            bump(&self.peers.counters.degraded);
        }
        PoolLookup::Miss { best, degraded }
    }

    /// The most similar entry for `embedding` in `scope`, wherever in the pool it lives. Errors
    /// only for an embedding no cache would take (empty, not finite, zero, wrong dimension for
    /// this node's own cache); peer failures are misses.
    pub async fn lookup(&self, scope: &str, embedding: &[f32]) -> Result<PoolLookup<V>, SemanticError> {
        let query = normalize(embedding)?;
        if let Some(l1) = &self.l1
            && let Ok(Lookup::Hit { value, score, .. }) = l1.lookup(scope, &query)
        {
            bump(&self.peers.counters.l1_hits);
            return Ok(PoolLookup::Hit { value, score, from: HitFrom::L1 });
        }
        let Some(plan) = self.plan(scope) else {
            return Ok(self.miss(None, false));
        };
        let mut best: Option<f32> = None;
        let mut degraded = false;
        let mut home_missed = false;
        for (i, owner) in plan.owners().enumerate() {
            let replica = i > 0;
            if replica && home_missed && !self.config.ask_replica_on_miss {
                break;
            }
            if owner == self.node {
                match self.shard.lookup(scope, &query)? {
                    Lookup::Hit { value, score, .. } => {
                        bump(&self.peers.counters.shard_hits);
                        if replica && home_missed {
                            self.repair(&plan.home, scope, &query, &value);
                        }
                        return Ok(PoolLookup::Hit { value, score, from: HitFrom::Shard });
                    }
                    Lookup::Miss { best: b } => {
                        best = better(best, b);
                        home_missed = true;
                        continue;
                    }
                }
            }
            let req = SharedCacheReq::Lookup { scope: scope.to_owned(), embedding: query.clone() };
            match self.peers.call(owner, req, self.config.lookup_timeout).await {
                Ok(SharedCacheResp::Hit { value, score, embedding }) => {
                    bump(&self.peers.counters.peer_hits);
                    if let Some(l1) = &self.l1 {
                        let at = if embedding.len() == query.len() { &embedding } else { &query };
                        let _ = l1.insert(scope, at, value.clone());
                    }
                    if replica && home_missed {
                        self.repair(&plan.home, scope, &query, &value);
                    }
                    return Ok(PoolLookup::Hit { value, score, from: HitFrom::Peer { node: owner.to_owned(), replica } });
                }
                Ok(SharedCacheResp::Miss { best: b }) => {
                    best = better(best, b);
                    home_missed = true;
                }
                Ok(_) | Err(_) => degraded = true,
            }
        }
        Ok(self.miss(best, degraded))
    }

    /// Copies a replica's hit back to the home that missed it (read repair).
    fn repair(&self, home: &str, scope: &str, embedding: &[f32], value: &V) {
        if home == self.node {
            let _ = self.shard.insert(scope, embedding, value.clone());
        } else {
            let req = SharedCacheReq::Insert { scope: scope.to_owned(), embedding: embedding.to_vec(), value: value.clone(), age_ms: 0 };
            self.peers.spawn(home.to_owned(), req, self.config.write_timeout);
        }
    }

    /// Stores `value` under `embedding` in `scope` on the shard's owners. Errors only for an
    /// embedding no cache would take; peer failures show in the [`PoolWrite`].
    pub async fn insert(&self, scope: &str, embedding: &[f32], value: V) -> Result<PoolWrite, SemanticError> {
        let vector = normalize(embedding)?;
        let mut out = PoolWrite::default();
        let Some(plan) = self.plan(scope) else {
            return Ok(out);
        };
        if plan.owns(&self.node) {
            self.shard.insert(scope, &vector, value.clone())?;
            out.acked.push(self.node.clone());
        } else if let Some(l1) = &self.l1 {
            let _ = l1.insert(scope, &vector, value.clone());
        }
        let req = |value: V| SharedCacheReq::Insert { scope: scope.to_owned(), embedding: vector.clone(), value, age_ms: 0 };
        if plan.home != self.node {
            match self.peers.call(&plan.home, req(value.clone()), self.config.write_timeout).await {
                Ok(SharedCacheResp::Stored) => out.acked.push(plan.home.clone()),
                _ => out.failed.push(plan.home.clone()),
            }
        }
        if let Some(replica) = plan.replica.filter(|r| *r != self.node) {
            if self.peers.is_down(&replica) {
                bump(&self.peers.counters.skipped);
                out.failed.push(replica);
            } else {
                self.peers.spawn(replica.clone(), req(value), self.config.write_timeout);
                out.queued.push(replica);
            }
        }
        Ok(out)
    }

    /// Drops every entry in `scope` here and on every member (waited for, bounded).
    pub async fn remove_scope(&self, scope: &str) -> PoolWrite {
        if let Some(l1) = &self.l1 {
            l1.remove_scope(scope);
        }
        self.shard.remove_scope(scope);
        let mut out = PoolWrite::default();
        let ring = self.ring();
        if ring.contains(&self.node) {
            out.acked.push(self.node.clone());
        }
        let mut tasks = tokio::task::JoinSet::new();
        for node in ring.nodes().filter(|n| *n != self.node) {
            let (peers, node, scope, timeout) = (self.peers.clone(), node.to_owned(), scope.to_owned(), self.config.write_timeout);
            tasks.spawn(async move {
                let ok = matches!(peers.call(&node, SharedCacheReq::RemoveScope { scope }, timeout).await, Ok(SharedCacheResp::Removed { .. }));
                (node, ok)
            });
        }
        while let Some(done) = tasks.join_next().await {
            match done {
                Ok((node, true)) => out.acked.push(node),
                Ok((node, false)) => out.failed.push(node),
                Err(_) => {}
            }
        }
        out.acked.sort();
        out.failed.sort();
        out
    }

    /// The exact entry under `key` (see [`exact_key_rev`](crate::exact_key_rev)) from its home,
    /// or its replica when the home is unreachable. `None` without [`with_exact`](Self::with_exact).
    pub async fn exact_get(&self, key: &Key) -> Option<Vec<u8>> {
        let exact = self.exact.as_ref()?;
        let plan = self.plan(key)?;
        for owner in plan.owners() {
            if owner == self.node {
                match exact.get(key) {
                    Some(v) => {
                        bump(&self.peers.counters.shard_hits);
                        return Some(v);
                    }
                    None => break,
                }
            }
            match self.peers.call(owner, SharedCacheReq::ExactGet { key: *key }, self.config.lookup_timeout).await {
                Ok(SharedCacheResp::Exact { value: Some(v) }) => {
                    bump(&self.peers.counters.peer_hits);
                    return Some(v);
                }
                Ok(SharedCacheResp::Exact { value: None }) => break,
                _ => {}
            }
        }
        bump(&self.peers.counters.misses);
        None
    }

    /// Stores an exact entry on its home (waited for, bounded) and replica (in the background).
    /// Nothing is stored without [`with_exact`](Self::with_exact).
    pub async fn exact_put(&self, key: &Key, value: Vec<u8>) -> PoolWrite {
        let mut out = PoolWrite::default();
        let (Some(exact), Some(plan)) = (self.exact.as_ref(), self.plan(key)) else {
            return out;
        };
        if plan.owns(&self.node) {
            exact.put(key, value.clone());
            out.acked.push(self.node.clone());
        }
        if plan.home != self.node {
            match self.peers.call(&plan.home, SharedCacheReq::ExactPut { key: *key, value: value.clone() }, self.config.write_timeout).await {
                Ok(SharedCacheResp::Stored) => out.acked.push(plan.home.clone()),
                _ => out.failed.push(plan.home.clone()),
            }
        }
        if let Some(replica) = plan.replica.filter(|r| *r != self.node) {
            self.peers.spawn(replica.clone(), SharedCacheReq::ExactPut { key: *key, value }, self.config.write_timeout);
            out.queued.push(replica);
        }
        out
    }
}
