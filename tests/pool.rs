//! Pool mode over in-process channels: several inference nodes sharing one cache.
#![cfg(feature = "pool")]

use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use weirdnet_agent::remote::{Caller, MemoryDialer, RemoteError};
use weirdnet_semcache::pool::{CacheTransport, DialerTransport, HitFrom, PoolConfig, PoolLookup, PooledSemanticCache, serve};
use weirdnet_semcache::{MemoryStore, PROTOCOL, SemanticCache, SemanticConfig, ShardPlan, ShardRing, SharedCacheReq, SharedCacheResp, scope_key};

type Cache = PooledSemanticCache<String>;

struct Pool {
    dialer: MemoryDialer,
    nodes: Vec<Arc<Cache>>,
}

impl Pool {
    fn node(&self, name: &str) -> &Arc<Cache> {
        self.nodes.iter().find(|n| n.node() == name).unwrap()
    }
}

fn shard_config() -> SemanticConfig {
    SemanticConfig::new(64).with_threshold(0.9)
}

fn route(dialer: &MemoryDialer, cache: &Cache, allow: bool) {
    let host = cache.host();
    dialer.route(cache.node(), move |ch, protocol| {
        assert_eq!(protocol, PROTOCOL);
        let host = host.clone();
        tokio::spawn(async move {
            let auth = move |_: &Caller| if allow { Ok(()) } else { Err(RemoteError::refused("not in this pool")) };
            let _ = serve(ch, &Caller::new("asker"), &auth, &host).await;
        });
    });
}

fn pool_with(names: &[&str], config: PoolConfig, shard: SemanticConfig) -> Pool {
    let dialer = MemoryDialer::new();
    let nodes = names
        .iter()
        .map(|name| {
            let transport = Arc::new(DialerTransport::new(dialer.clone()));
            let cache = PooledSemanticCache::new(*name, Arc::new(SemanticCache::with_config(shard.clone())), transport, config.clone())
                .with_exact(Arc::new(MemoryStore::new(64)));
            cache.set_nodes(names.iter().copied());
            route(&dialer, &cache, true);
            Arc::new(cache)
        })
        .collect();
    Pool { dialer, nodes }
}

fn pool(names: &[&str]) -> Pool {
    pool_with(names, PoolConfig::new(), shard_config())
}

/// A scope whose plan on `ring` satisfies `want`.
fn scope_where(ring: &ShardRing, want: impl Fn(&ShardPlan) -> bool) -> String {
    (0..10_000).map(|i| scope_key(&["model", &i.to_string()])).find(|s| want(&ring.plan(s).unwrap())).unwrap()
}

const E: [f32; 3] = [1.0, 0.0, 0.0];
const NEAR: [f32; 3] = [1.0, 0.1, 0.0];

#[tokio::test]
async fn a_hit_anywhere_is_a_hit_everywhere() {
    let p = pool(&["a", "b", "c", "d"]);
    let scope = scope_where(&p.node("a").ring(), |plan| !plan.owns("d"));
    let plan = p.node("a").plan(&scope).unwrap();
    let writer = p.node("d");
    let w = writer.insert(&scope, &E, "answer".to_string()).await.unwrap();
    assert_eq!(w.acked, vec![plan.home.clone()]);
    assert_eq!(w.queued, vec![plan.replica.clone().unwrap()]);
    assert!(w.failed.is_empty());
    writer.settle().await;
    assert_eq!(p.node(&plan.home).shard().len(), 1);
    assert_eq!(p.node(plan.replica.as_deref().unwrap()).shard().len(), 1);
    assert!(writer.shard().is_empty(), "a non-owner keeps only an L1 copy");

    // The writer answers from its L1 copy; the owners from their own shards; a non-owner asks.
    let hit = writer.lookup(&scope, &NEAR).await.unwrap();
    assert!(matches!(hit, PoolLookup::Hit { from: HitFrom::L1, .. }), "{hit:?}");
    let home = p.node(&plan.home).lookup(&scope, &NEAR).await.unwrap();
    assert!(matches!(home, PoolLookup::Hit { from: HitFrom::Shard, .. }), "{home:?}");
    let other = p.nodes.iter().find(|n| !plan.owns(n.node()) && n.node() != "d").unwrap();
    match other.lookup(&scope, &NEAR).await.unwrap() {
        PoolLookup::Hit { value, score, from } => {
            assert_eq!(value, "answer");
            assert!(score > 0.99);
            assert_eq!(from, HitFrom::Peer { node: plan.home.clone(), replica: false });
        }
        miss => panic!("{miss:?}"),
    }
    // Copied to its L1 under the stored entry's direction.
    assert!(matches!(other.lookup(&scope, &E).await.unwrap(), PoolLookup::Hit { from: HitFrom::L1, .. }));
    let stats = other.stats();
    assert_eq!((stats.peer_hits, stats.l1_hits, stats.misses), (1, 1, 0));
}

#[tokio::test]
async fn the_replica_answers_when_the_home_dies() {
    let p = pool_with(&["a", "b", "c"], PoolConfig::new().with_l1(0, None), shard_config());
    let ring = p.node("a").ring();
    let scope = scope_where(&ring, |plan| !plan.owns("c"));
    let plan = ring.plan(&scope).unwrap();
    let c = p.node("c");
    c.insert(&scope, &E, "kept".to_string()).await.unwrap();
    c.settle().await;

    p.dialer.unroute(&plan.home);
    let started = Instant::now();
    match c.lookup(&scope, &NEAR).await.unwrap() {
        PoolLookup::Hit { value, from, .. } => {
            assert_eq!(value, "kept");
            assert_eq!(from, HitFrom::Peer { node: plan.replica.clone().unwrap(), replica: true });
        }
        miss => panic!("{miss:?}"),
    }
    assert!(started.elapsed() < Duration::from_secs(1));
    assert!(c.is_down(&plan.home));
    // Marked down: skipped without another call.
    assert!(c.lookup(&scope, &NEAR).await.unwrap().is_hit());
    assert_eq!(c.stats().skipped, 1);

    // And when the pool drops the dead node, the replica is the home.
    for n in &p.nodes {
        n.leave(&plan.home);
    }
    assert_eq!(c.plan(&scope).unwrap().home, plan.replica.unwrap());
    match c.lookup(&scope, &NEAR).await.unwrap() {
        PoolLookup::Hit { from: HitFrom::Peer { replica, .. }, .. } => assert!(!replica),
        other => panic!("{other:?}"),
    }
}

#[tokio::test]
async fn writes_go_to_the_replica_when_the_home_is_down() {
    let p = pool_with(&["a", "b", "c"], PoolConfig::new().with_l1(0, None), shard_config());
    let ring = p.node("a").ring();
    let scope = scope_where(&ring, |plan| !plan.owns("c"));
    let plan = ring.plan(&scope).unwrap();
    p.dialer.unroute(&plan.home);
    let c = p.node("c");
    let w = c.insert(&scope, &E, "v".to_string()).await.unwrap();
    assert_eq!(w.failed, vec![plan.home.clone()]);
    assert_eq!(w.queued, vec![plan.replica.clone().unwrap()]);
    c.settle().await;
    assert!(c.lookup(&scope, &E).await.unwrap().is_hit());
}

#[tokio::test]
async fn scopes_never_cross() {
    let p = pool(&["a", "b", "c"]);
    let (s1, s2) = (scope_key(&["m", "r1", "sys"]), scope_key(&["m", "r2", "sys"]));
    p.node("a").insert(&s1, &E, "r1".to_string()).await.unwrap();
    p.node("a").settle().await;
    for n in &p.nodes {
        assert!(n.lookup(&s1, &E).await.unwrap().is_hit());
        assert!(!n.lookup(&s2, &E).await.unwrap().is_hit(), "another revision misses on {}", n.node());
    }
}

#[tokio::test]
async fn entries_expire_across_the_pool() {
    let shard = shard_config().with_ttl(Duration::from_millis(60));
    let p = pool_with(&["a", "b", "c"], PoolConfig::new().with_l1(16, Some(Duration::from_millis(60))), shard);
    let scope = scope_where(&p.node("a").ring(), |plan| !plan.owns("c"));
    p.node("c").insert(&scope, &E, "short".to_string()).await.unwrap();
    p.node("c").settle().await;
    for n in &p.nodes {
        assert!(n.lookup(&scope, &E).await.unwrap().is_hit());
    }
    tokio::time::sleep(Duration::from_millis(100)).await;
    for n in &p.nodes {
        assert!(!n.lookup(&scope, &E).await.unwrap().is_hit(), "expired on {}", n.node());
    }
}

#[tokio::test]
async fn shards_evict_least_recently_used() {
    let p = pool_with(&["a"], PoolConfig::new(), SemanticConfig::new(2).with_threshold(0.9));
    let a = p.node("a");
    let s = scope_key(&["m"]);
    for (i, e) in [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]].iter().enumerate() {
        a.insert(&s, e, i.to_string()).await.unwrap();
    }
    assert_eq!(a.shard().len(), 2);
    assert!(!a.lookup(&s, &[1.0, 0.0, 0.0]).await.unwrap().is_hit());
    assert!(a.lookup(&s, &[0.0, 0.0, 1.0]).await.unwrap().is_hit());
}

/// A peer that never answers.
struct Stall;

#[async_trait]
impl CacheTransport<String> for Stall {
    async fn call(&self, _: &str, _: SharedCacheReq<String>) -> Result<SharedCacheResp<String>, RemoteError> {
        tokio::time::sleep(Duration::from_secs(30)).await;
        Err(RemoteError::Timeout)
    }
}

/// A peer that always fails.
struct Broken;

#[async_trait]
impl CacheTransport<String> for Broken {
    async fn call(&self, node: &str, _: SharedCacheReq<String>) -> Result<SharedCacheResp<String>, RemoteError> {
        Err(RemoteError::Unreachable { message: format!("{node} is gone") })
    }
}

fn alone_with(transport: Arc<dyn CacheTransport<String>>, config: PoolConfig) -> Cache {
    let c = PooledSemanticCache::new("me", Arc::new(SemanticCache::with_config(shard_config())), transport, config);
    c.set_nodes(["p1", "p2"]);
    c
}

#[tokio::test]
async fn a_stalled_pool_is_a_quick_miss() {
    let config = PoolConfig::new().with_lookup_timeout(Duration::from_millis(30)).with_write_timeout(Duration::from_millis(30));
    let c = alone_with(Arc::new(Stall), config);
    let scope = scope_key(&["m"]);
    let started = Instant::now();
    assert_eq!(c.lookup(&scope, &E).await.unwrap(), PoolLookup::Miss { best: None, degraded: true });
    assert!(started.elapsed() < Duration::from_millis(500), "{:?}", started.elapsed());
    assert_eq!(c.stats().timeouts, 2, "home and replica");
    // Both are down now: the next lookup doesn't wait at all.
    let started = Instant::now();
    assert!(!c.lookup(&scope, &E).await.unwrap().is_hit());
    assert!(started.elapsed() < Duration::from_millis(20));
    assert_eq!(c.stats().skipped, 2);
    let w = c.insert(&scope, &E, "v".to_string()).await.unwrap();
    assert_eq!(w.acked.len() + w.queued.len(), 0);
    assert_eq!(w.failed.len(), 2);
    // Its own L1 still serves what it wrote.
    assert!(matches!(c.lookup(&scope, &E).await.unwrap(), PoolLookup::Hit { from: HitFrom::L1, .. }));
    c.settle().await;
}

#[tokio::test]
async fn failing_peers_are_misses_and_retried_after_down_for() {
    let config = PoolConfig::new().with_down_for(Duration::from_millis(40)).with_l1(0, None);
    let c = alone_with(Arc::new(Broken), config);
    let scope = scope_key(&["m"]);
    assert_eq!(c.lookup(&scope, &E).await.unwrap(), PoolLookup::Miss { best: None, degraded: true });
    assert!(c.is_down("p1") && c.is_down("p2"));
    tokio::time::sleep(Duration::from_millis(60)).await;
    assert!(!c.is_down("p1"));
    c.lookup(&scope, &E).await.unwrap();
    let s = c.stats();
    assert_eq!((s.peer_errors, s.misses, s.degraded), (4, 2, 2));
}

#[tokio::test]
async fn refused_callers_fail_over_without_marking_the_node_down() {
    let p = pool_with(&["a", "b", "c"], PoolConfig::new().with_l1(0, None), shard_config());
    let a = p.node("a");
    let scope = scope_where(&a.ring(), |plan| plan.home == "b" && plan.replica.as_deref() == Some("c"));
    p.node("b").insert(&scope, &E, "kept".to_string()).await.unwrap();
    p.node("b").settle().await;
    route(&p.dialer, p.node("b"), false);
    match a.lookup(&scope, &E).await.unwrap() {
        PoolLookup::Hit { from, .. } => assert_eq!(from, HitFrom::Peer { node: "c".into(), replica: true }),
        miss => panic!("{miss:?}"),
    }
    assert!(!a.is_down("b"), "a refusal is an answer, not an outage");
    assert_eq!(a.stats().peer_errors, 1);
    route(&p.dialer, p.node("c"), false);
    assert_eq!(a.lookup(&scope, &E).await.unwrap(), PoolLookup::Miss { best: None, degraded: true });
}

#[tokio::test]
async fn remove_scope_reaches_every_member() {
    let p = pool(&["a", "b", "c"]);
    let scope = scope_key(&["m"]);
    p.node("a").insert(&scope, &E, "old".to_string()).await.unwrap();
    p.node("a").settle().await;
    for n in &p.nodes {
        assert!(n.lookup(&scope, &E).await.unwrap().is_hit());
    }
    let w = p.node("c").remove_scope(&scope).await;
    assert_eq!(w.acked, vec!["a", "b", "c"]);
    for n in &p.nodes {
        assert!(n.shard().is_empty());
    }
    assert!(!p.node("c").lookup(&scope, &E).await.unwrap().is_hit());
    // A dead member is reported, not waited on forever.
    p.dialer.unroute("b");
    let w = p.node("a").remove_scope(&scope).await;
    assert_eq!(w.failed, vec!["b"]);
}

#[tokio::test]
async fn a_joining_home_is_repaired_from_the_old_home() {
    let config = PoolConfig::new().with_l1(0, None).with_ask_replica_on_miss(true);
    let p = pool_with(&["a", "b", "c", "new"], config, shard_config());
    let mut before = ShardRing::new(["a", "b", "c"]);
    let scope = scope_where(&p.node("a").ring(), |plan| plan.home == "new" && !plan.owns("c"));
    for n in &p.nodes {
        n.leave("new");
    }
    assert_eq!(p.node("a").ring(), before);
    let old = before.plan(&scope).unwrap();
    p.node("c").insert(&scope, &E, "v".to_string()).await.unwrap_or_else(|e| panic!("{e}"));
    p.node("c").settle().await;

    before.insert("new");
    for n in &p.nodes {
        n.join("new");
    }
    let plan = p.node("c").plan(&scope).unwrap();
    assert_eq!((plan.home.as_str(), plan.replica.as_deref()), ("new", Some(old.home.as_str())));
    let c = p.node("c");
    match c.lookup(&scope, &E).await.unwrap() {
        PoolLookup::Hit { from, .. } => assert_eq!(from, HitFrom::Peer { node: old.home.clone(), replica: true }),
        miss => panic!("{miss:?}"),
    }
    c.settle().await;
    assert_eq!(p.node("new").shard().len(), 1, "read repair copied it to the new home");
    match c.lookup(&scope, &E).await.unwrap() {
        PoolLookup::Hit { from, .. } => assert_eq!(from, HitFrom::Peer { node: "new".into(), replica: false }),
        miss => panic!("{miss:?}"),
    }
}

#[tokio::test]
async fn exact_entries_are_shared() {
    let p = pool(&["a", "b", "c"]);
    let key = [3u8; 32];
    assert_eq!(p.node("a").exact_get(&key).await, None);
    let w = p.node("a").exact_put(&key, b"answer".to_vec()).await;
    assert_eq!(w.acked.len() + w.queued.len(), 2);
    p.node("a").settle().await;
    for n in &p.nodes {
        assert_eq!(n.exact_get(&key).await.as_deref(), Some(&b"answer"[..]), "on {}", n.node());
    }
    let plan = p.node("a").plan(key).unwrap();
    p.dialer.unroute(&plan.home);
    let asker = p.nodes.iter().find(|n| n.node() != plan.home).unwrap();
    assert_eq!(asker.exact_get(&key).await.as_deref(), Some(&b"answer"[..]), "from the replica");
}

#[tokio::test]
async fn only_bad_embeddings_are_errors() {
    let p = pool(&["a", "b"]);
    let a = p.node("a");
    let s = scope_key(&["m"]);
    assert!(a.lookup(&s, &[]).await.is_err());
    assert!(a.insert(&s, &[f32::NAN], "v".to_string()).await.is_err());
    assert!(a.lookup(&s, &[0.0, 0.0]).await.is_err());
    // An empty ring holds nothing and asks no one.
    a.set_nodes(Vec::<String>::new());
    assert_eq!(a.lookup(&s, &E).await.unwrap(), PoolLookup::Miss { best: None, degraded: false });
    assert_eq!(a.insert(&s, &E, "v".to_string()).await.unwrap(), Default::default());
}
