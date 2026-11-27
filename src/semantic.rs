//! Similarity lookup by embedding: [`SemanticCache`].
//!
//! The caller embeds the prompt (with whatever embedding model it runs) and asks the cache for
//! the stored entry whose embedding is closest by cosine similarity. A stored value stands in
//! only when that similarity reaches the configured threshold, and only within the same scope,
//! so answers never cross models, revisions or system prompts. See [`scope_key`].
//!
//! Embeddings are validated (non-empty, finite, not all zero, the scope's dimension) and
//! normalized to unit length on the way in, so callers may pass raw or normalized vectors.

use std::collections::HashMap;
use std::fmt;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use weirdnet_agent::remote::{from_cbor, to_cbor};

use crate::CachedAnswer;

/// Similarity a lookup needs to hit when none is configured.
pub const DEFAULT_THRESHOLD: f32 = 0.95;

/// Two normalized embeddings at least this similar are the same entry: inserting one replaces
/// the other's value instead of adding a near-duplicate.
const SAME_ENTRY: f32 = 1.0 - 1e-6;

/// How a [`SemanticCache`] matches and evicts.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct SemanticConfig {
    /// At most this many entries across all scopes (at least 1); the least recently used goes first.
    pub max_entries: usize,
    /// Cosine similarity, in `-1.0..=1.0`, a lookup needs to hit.
    pub threshold: f32,
    /// Entries older than this never hit and are dropped. `None` keeps them until evicted.
    pub ttl: Option<Duration>,
    /// Every embedding must have this many dimensions. `None` fixes each scope's dimension from
    /// its first entry (until the scope is empty again).
    pub dimension: Option<usize>,
}

impl SemanticConfig {
    /// At most `max_entries`, threshold [`DEFAULT_THRESHOLD`], no TTL, any dimension.
    pub fn new(max_entries: usize) -> Self {
        Self { max_entries: max_entries.max(1), threshold: DEFAULT_THRESHOLD, ttl: None, dimension: None }
    }

    /// Hit at `threshold` or above, clamped to `-1.0..=1.0`. A NaN keeps the current threshold.
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        if !threshold.is_nan() {
            self.threshold = threshold.clamp(-1.0, 1.0);
        }
        self
    }

    /// Entries expire `ttl` after they were stored.
    pub fn with_ttl(mut self, ttl: Duration) -> Self {
        self.ttl = Some(ttl);
        self
    }

    /// Every embedding must have `dimension` dimensions.
    pub fn with_dimension(mut self, dimension: usize) -> Self {
        self.dimension = Some(dimension);
        self
    }
}

impl Default for SemanticConfig {
    fn default() -> Self {
        Self::new(256)
    }
}

/// Why a [`SemanticCache`] call was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum SemanticError {
    /// The embedding has no dimensions.
    Empty,
    /// The embedding holds a NaN or an infinity.
    NotFinite,
    /// The embedding is all zeros (or too small to normalize), so it has no direction.
    ZeroVector,
    /// The embedding's dimension isn't the configured one or the scope's.
    DimensionMismatch { expected: usize, found: usize },
    /// A snapshot couldn't be written or read.
    Snapshot(String),
}

impl fmt::Display for SemanticError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => f.write_str("embedding is empty"),
            Self::NotFinite => f.write_str("embedding holds a NaN or an infinity"),
            Self::ZeroVector => f.write_str("embedding is a zero vector"),
            Self::DimensionMismatch { expected, found } => write!(f, "embedding has {found} dimensions, expected {expected}"),
            Self::Snapshot(e) => write!(f, "snapshot: {e}"),
        }
    }
}

impl std::error::Error for SemanticError {}

/// The result of [`SemanticCache::lookup`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Lookup<V> {
    /// The most similar live entry in the scope reached the threshold.
    Hit {
        /// A copy of the stored value.
        value: V,
        /// Cosine similarity to the query.
        score: f32,
        /// The entry, for [`SemanticCache::remove`].
        id: u64,
    },
    /// Nothing reached the threshold.
    Miss {
        /// The best similarity found below the threshold, `None` when the scope had no live entries.
        best: Option<f32>,
    },
}

impl<V> Lookup<V> {
    pub fn is_hit(&self) -> bool {
        matches!(self, Self::Hit { .. })
    }

    /// The hit's score, or the best score of a miss.
    pub fn score(&self) -> Option<f32> {
        match self {
            Self::Hit { score, .. } => Some(*score),
            Self::Miss { best } => *best,
        }
    }

    /// The value of a hit.
    pub fn into_value(self) -> Option<V> {
        match self {
            Self::Hit { value, .. } => Some(value),
            Self::Miss { .. } => None,
        }
    }
}

/// Counters since the cache was made or last [`clear`](SemanticCache::clear)ed.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct SemanticStats {
    /// Live entries now (expired ones not yet dropped included).
    pub entries: usize,
    pub hits: u64,
    pub misses: u64,
    /// Entries dropped to make room.
    pub evictions: u64,
    /// Entries dropped because their TTL ran out.
    pub expired: u64,
}

/// A stable scope for [`SemanticCache`]: hex SHA-256 over the length-prefixed `parts`, such as
/// the model id, its revision and a hash of the system prompt. Different parts (or the same
/// text split differently) give different scopes.
pub fn scope_key(parts: &[&str]) -> String {
    let mut h = Sha256::new();
    h.update(b"weirdnet-semcache/scope/v1");
    for p in parts {
        h.update((p.len() as u64).to_le_bytes());
        h.update(p.as_bytes());
    }
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

/// `embedding` scaled to unit length, after checking it is non-empty, finite and not zero.
pub fn normalize(embedding: &[f32]) -> Result<Vec<f32>, SemanticError> {
    if embedding.is_empty() {
        return Err(SemanticError::Empty);
    }
    if embedding.iter().any(|x| !x.is_finite()) {
        return Err(SemanticError::NotFinite);
    }
    let norm = embedding.iter().map(|&x| f64::from(x) * f64::from(x)).sum::<f64>().sqrt();
    if !norm.is_normal() {
        return Err(SemanticError::ZeroVector);
    }
    let v: Vec<f32> = embedding.iter().map(|&x| (f64::from(x) / norm) as f32).collect();
    if v.iter().all(|&x| x == 0.0) {
        return Err(SemanticError::ZeroVector);
    }
    Ok(v)
}

/// Cosine similarity of two embeddings, or `None` when either is empty, not finite or zero, or
/// their dimensions differ.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> Option<f32> {
    if a.len() != b.len() {
        return None;
    }
    let (a, b) = (normalize(a).ok()?, normalize(b).ok()?);
    Some(dot(&a, &b))
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    let s: f64 = a.iter().zip(b).map(|(&x, &y)| f64::from(x) * f64::from(y)).sum();
    (s as f32).clamp(-1.0, 1.0)
}

struct Entry<V> {
    id: u64,
    vector: Vec<f32>,
    value: V,
    stored: Instant,
    used: u64,
}

struct Scope<V> {
    dimension: usize,
    entries: Vec<Entry<V>>,
}

struct State<V> {
    tick: u64,
    next_id: u64,
    scopes: HashMap<String, Scope<V>>,
    stats: SemanticStats,
}

impl<V> State<V> {
    fn len(&self) -> usize {
        self.scopes.values().map(|s| s.entries.len()).sum()
    }

    fn drop_empty(&mut self) {
        self.scopes.retain(|_, s| !s.entries.is_empty());
    }

    fn purge_expired(&mut self, ttl: Option<Duration>, now: Instant) {
        let Some(ttl) = ttl else { return };
        let mut gone = 0;
        for s in self.scopes.values_mut() {
            let before = s.entries.len();
            s.entries.retain(|e| !expired(e.stored, ttl, now));
            gone += (before - s.entries.len()) as u64;
        }
        self.stats.expired += gone;
        self.drop_empty();
    }

    fn evict_lru(&mut self) {
        let oldest = self
            .scopes
            .iter()
            .flat_map(|(k, s)| s.entries.iter().enumerate().map(move |(i, e)| (e.used, k, i)))
            .min_by_key(|(used, _, _)| *used)
            .map(|(_, k, i)| (k.clone(), i));
        if let Some((k, i)) = oldest
            && let Some(s) = self.scopes.get_mut(&k)
        {
            s.entries.swap_remove(i);
            self.stats.evictions += 1;
        }
        self.drop_empty();
    }
}

fn expired(stored: Instant, ttl: Duration, now: Instant) -> bool {
    now.saturating_duration_since(stored) >= ttl
}

/// Embedding-similarity cache: the nearest stored embedding in the same scope, by cosine
/// similarity, stands in when it reaches the threshold.
///
/// Bounded and least-recently-used across scopes like [`MemoryStore`](crate::MemoryStore), with
/// an optional TTL. Values default to [`CachedAnswer`] but may be any `Clone` type. Callers
/// decide what may be cached; for infer/2 requests check [`may_read`](crate::may_read) and
/// [`may_store`](crate::may_store) first.
pub struct SemanticCache<V = CachedAnswer> {
    config: SemanticConfig,
    state: Mutex<State<V>>,
}

impl<V: Clone> SemanticCache<V> {
    /// In memory, at most `max_entries`, with the default threshold and no TTL.
    pub fn new(max_entries: usize) -> Self {
        Self::with_config(SemanticConfig::new(max_entries))
    }

    pub fn with_config(config: SemanticConfig) -> Self {
        let config = SemanticConfig { max_entries: config.max_entries.max(1), ..config };
        Self { config, state: Mutex::new(State { tick: 0, next_id: 1, scopes: HashMap::new(), stats: SemanticStats::default() }) }
    }

    pub fn config(&self) -> &SemanticConfig {
        &self.config
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, State<V>> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn check(&self, scope: Option<&Scope<V>>, v: &[f32]) -> Result<(), SemanticError> {
        let expected = self.config.dimension.or(scope.map(|s| s.dimension));
        match expected {
            Some(expected) if expected != v.len() => Err(SemanticError::DimensionMismatch { expected, found: v.len() }),
            _ => Ok(()),
        }
    }

    /// The most similar live entry in `scope`: a [`Lookup::Hit`] when its similarity reaches the
    /// threshold (and the entry counts as used), a [`Lookup::Miss`] with the best score otherwise.
    pub fn lookup(&self, scope: &str, embedding: &[f32]) -> Result<Lookup<V>, SemanticError> {
        self.lookup_at(scope, embedding, Instant::now())
    }

    fn lookup_at(&self, scope: &str, embedding: &[f32], now: Instant) -> Result<Lookup<V>, SemanticError> {
        self.lookup_with_vector(scope, embedding, now).map(|(l, _)| l)
    }

    /// [`lookup`](Self::lookup), also returning the hit's stored (normalized) embedding.
    pub(crate) fn lookup_with_vector(&self, scope: &str, embedding: &[f32], now: Instant) -> Result<(Lookup<V>, Option<Vec<f32>>), SemanticError> {
        let query = normalize(embedding)?;
        let mut g = self.lock();
        g.purge_expired(self.config.ttl, now);
        self.check(g.scopes.get(scope), &query)?;
        g.tick += 1;
        let tick = g.tick;
        let threshold = self.config.threshold;
        let best = g.scopes.get_mut(scope).and_then(|s| s.entries.iter_mut().map(|e| (dot(&e.vector, &query), e)).max_by(|a, b| a.0.total_cmp(&b.0)));
        let (result, vector) = match best {
            Some((score, e)) if score >= threshold => {
                e.used = tick;
                (Lookup::Hit { value: e.value.clone(), score, id: e.id }, Some(e.vector.clone()))
            }
            other => (Lookup::Miss { best: other.map(|(score, _)| score) }, None),
        };
        if result.is_hit() {
            g.stats.hits += 1;
        } else {
            g.stats.misses += 1;
        }
        Ok((result, vector))
    }

    /// Stores `value` under `embedding` in `scope` and returns the entry's id. An entry with the
    /// same direction in the scope is replaced (keeping its id); otherwise, when full, expired
    /// entries go first, then the least recently used.
    pub fn insert(&self, scope: &str, embedding: &[f32], value: V) -> Result<u64, SemanticError> {
        self.insert_at(scope, embedding, value, Instant::now())
    }

    pub(crate) fn insert_at(&self, scope: &str, embedding: &[f32], value: V, now: Instant) -> Result<u64, SemanticError> {
        let vector = normalize(embedding)?;
        let mut g = self.lock();
        g.purge_expired(self.config.ttl, now);
        self.check(g.scopes.get(scope), &vector)?;
        g.tick += 1;
        let tick = g.tick;
        if let Some(e) = g.scopes.get_mut(scope).and_then(|s| s.entries.iter_mut().find(|e| dot(&e.vector, &vector) >= SAME_ENTRY)) {
            e.vector = vector;
            e.value = value;
            e.stored = now;
            e.used = tick;
            return Ok(e.id);
        }
        while g.len() >= self.config.max_entries {
            g.evict_lru();
        }
        let id = g.next_id;
        g.next_id += 1;
        let dimension = vector.len();
        g.scopes.entry(scope.to_owned()).or_insert_with(|| Scope { dimension, entries: Vec::new() }).entries.push(Entry {
            id,
            vector,
            value,
            stored: now,
            used: tick,
        });
        Ok(id)
    }

    /// Removes entry `id` and returns its value.
    pub fn remove(&self, id: u64) -> Option<V> {
        let mut g = self.lock();
        let found = g.scopes.values_mut().find_map(|s| {
            let i = s.entries.iter().position(|e| e.id == id)?;
            Some(s.entries.swap_remove(i).value)
        });
        g.drop_empty();
        found
    }

    /// Removes every entry in `scope` and returns how many there were.
    pub fn remove_scope(&self, scope: &str) -> usize {
        self.lock().scopes.remove(scope).map_or(0, |s| s.entries.len())
    }

    /// Removes every entry and resets the counters.
    pub fn clear(&self) {
        let mut g = self.lock();
        g.scopes.clear();
        g.stats = SemanticStats::default();
    }

    /// Entries stored, expired ones not yet dropped included.
    pub fn len(&self) -> usize {
        self.lock().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn stats(&self) -> SemanticStats {
        let g = self.lock();
        SemanticStats { entries: g.len(), ..g.stats }
    }
}

#[derive(Serialize, Deserialize)]
struct SnapshotEntry<V> {
    scope: String,
    vector: Vec<f32>,
    /// Age when written, so the TTL keeps counting after a restore.
    age_ms: u64,
    value: V,
}

#[derive(Serialize, Deserialize)]
struct Snapshot<V> {
    v: u8,
    entries: Vec<SnapshotEntry<V>>,
}

impl<V: Clone + Serialize + DeserializeOwned> SemanticCache<V> {
    /// The live entries as CBOR, least recently used first, for [`restore`](Self::restore).
    pub fn snapshot(&self) -> Result<Vec<u8>, SemanticError> {
        let now = Instant::now();
        let g = self.lock();
        let mut entries: Vec<(u64, SnapshotEntry<V>)> = g
            .scopes
            .iter()
            .flat_map(|(k, s)| s.entries.iter().map(move |e| (k, e)))
            .filter(|(_, e)| self.config.ttl.is_none_or(|ttl| !expired(e.stored, ttl, now)))
            .map(|(k, e)| {
                let age_ms = u64::try_from(now.saturating_duration_since(e.stored).as_millis()).unwrap_or(u64::MAX);
                (e.used, SnapshotEntry { scope: k.clone(), vector: e.vector.clone(), age_ms, value: e.value.clone() })
            })
            .collect();
        entries.sort_by_key(|(used, _)| *used);
        let snap = Snapshot { v: 1, entries: entries.into_iter().map(|(_, e)| e).collect() };
        to_cbor(&snap).map_err(|e| SemanticError::Snapshot(e.to_string()))
    }

    /// A cache with `config` holding the entries of a [`snapshot`](Self::snapshot). Entries that
    /// fail validation under `config` (dimension, finite) or have expired are skipped; the most
    /// recently used are kept when there are more than `max_entries`.
    pub fn restore(config: SemanticConfig, bytes: &[u8]) -> Result<Self, SemanticError> {
        let snap: Snapshot<V> = from_cbor(bytes).map_err(|e| SemanticError::Snapshot(e.to_string()))?;
        if snap.v != 1 {
            return Err(SemanticError::Snapshot(format!("unknown snapshot version {}", snap.v)));
        }
        let cache = Self::with_config(config);
        let now = Instant::now();
        for e in snap.entries {
            let stored = now.checked_sub(Duration::from_millis(e.age_ms)).unwrap_or(now);
            if cache.config.ttl.is_some_and(|ttl| expired(stored, ttl, now)) {
                continue;
            }
            let _ = cache.insert_at(&e.scope, &e.vector, e.value, stored);
        }
        cache.lock().stats = SemanticStats::default();
        Ok(cache)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cache(max: usize, threshold: f32) -> SemanticCache<&'static str> {
        SemanticCache::with_config(SemanticConfig::new(max).with_threshold(threshold))
    }

    #[test]
    fn exact_and_close_embeddings_hit() {
        let c = cache(8, 0.9);
        let id = c.insert("s", &[1.0, 0.0, 0.0], "a").unwrap();
        match c.lookup("s", &[2.0, 0.0, 0.0]).unwrap() {
            Lookup::Hit { value, score, id: hit } => {
                assert_eq!((value, hit), ("a", id));
                assert!((score - 1.0).abs() < 1e-6, "{score}");
            }
            other => panic!("{other:?}"),
        }
        let near = c.lookup("s", &[1.0, 0.1, 0.0]).unwrap();
        assert!(near.is_hit() && near.score().unwrap() > 0.99);
        assert_eq!(near.into_value(), Some("a"));
    }

    #[test]
    fn below_threshold_misses_with_its_best_score() {
        let c = cache(8, 0.9);
        c.insert("s", &[1.0, 0.0], "a").unwrap();
        let l = c.lookup("s", &[1.0, 1.0]).unwrap();
        assert!(!l.is_hit());
        let best = l.score().unwrap();
        assert!((best - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-5, "{best}");
        assert_eq!(c.lookup("empty", &[1.0, 0.0]).unwrap(), Lookup::Miss { best: None });
        assert!(cache(8, 0.7).insert("s", &[1.0, 0.0], "a").is_ok());
    }

    #[test]
    fn most_similar_entry_wins() {
        let c = cache(8, 0.5);
        c.insert("s", &[1.0, 0.0, 0.0], "x").unwrap();
        c.insert("s", &[0.8, 0.6, 0.0], "xy").unwrap();
        c.insert("s", &[0.0, 1.0, 0.0], "y").unwrap();
        assert_eq!(c.lookup("s", &[0.7, 0.7, 0.0]).unwrap().into_value(), Some("xy"));
        assert_eq!(c.lookup("s", &[0.1, 0.9, 0.0]).unwrap().into_value(), Some("y"));
        assert_eq!(c.lookup("s", &[0.9, 0.05, 0.1]).unwrap().into_value(), Some("x"));
    }

    #[test]
    fn scopes_never_cross() {
        let c = cache(8, 0.9);
        let a = scope_key(&["qwen3-8b", "rev1", "sys-hash"]);
        let b = scope_key(&["qwen3-8b", "rev2", "sys-hash"]);
        assert_ne!(a, b);
        assert_ne!(scope_key(&["ab", "c"]), scope_key(&["a", "bc"]), "parts are length-prefixed");
        assert_eq!(a.len(), 64);
        c.insert(&a, &[1.0, 0.0], "from a").unwrap();
        assert_eq!(c.lookup(&b, &[1.0, 0.0]).unwrap(), Lookup::Miss { best: None });
        c.insert(&b, &[1.0, 0.0], "from b").unwrap();
        assert_eq!(c.lookup(&a, &[1.0, 0.0]).unwrap().into_value(), Some("from a"));
        assert_eq!(c.lookup(&b, &[1.0, 0.0]).unwrap().into_value(), Some("from b"));
        // Scopes may even use different dimensions.
        c.insert("other", &[1.0, 0.0, 0.0], "3d").unwrap();
        assert_eq!(c.len(), 3);
    }

    #[test]
    fn dimensions_are_checked() {
        let c = cache(8, 0.9);
        c.insert("s", &[1.0, 0.0], "a").unwrap();
        let e = SemanticError::DimensionMismatch { expected: 2, found: 3 };
        assert_eq!(c.lookup("s", &[1.0, 0.0, 0.0]), Err(e.clone()));
        assert_eq!(c.insert("s", &[1.0, 0.0, 0.0], "b"), Err(e));
        assert_eq!(c.remove_scope("s"), 1);
        c.insert("s", &[1.0, 0.0, 0.0], "b").unwrap();

        let fixed: SemanticCache<u8> = SemanticCache::with_config(SemanticConfig::new(4).with_dimension(4));
        assert_eq!(fixed.insert("s", &[1.0, 2.0], 1), Err(SemanticError::DimensionMismatch { expected: 4, found: 2 }));
        assert_eq!(fixed.lookup("new", &[1.0]), Err(SemanticError::DimensionMismatch { expected: 4, found: 1 }));
        assert!(fixed.insert("s", &[1.0, 2.0, 3.0, 4.0], 1).is_ok());
        assert_eq!(cosine_similarity(&[1.0], &[1.0, 0.0]), None);
    }

    #[test]
    fn nan_zero_and_empty_embeddings_are_refused() {
        let c = cache(8, 0.9);
        assert_eq!(c.insert("s", &[], "a"), Err(SemanticError::Empty));
        assert_eq!(c.insert("s", &[f32::NAN, 1.0], "a"), Err(SemanticError::NotFinite));
        assert_eq!(c.insert("s", &[f32::INFINITY, 1.0], "a"), Err(SemanticError::NotFinite));
        assert_eq!(c.insert("s", &[0.0, 0.0], "a"), Err(SemanticError::ZeroVector));
        assert_eq!(c.insert("s", &[-0.0, 0.0], "a"), Err(SemanticError::ZeroVector));
        assert!(c.is_empty());
        c.insert("s", &[1.0, 0.0], "a").unwrap();
        assert_eq!(c.lookup("s", &[f32::NAN, 0.0]), Err(SemanticError::NotFinite));
        assert_eq!(c.lookup("s", &[0.0, 0.0]), Err(SemanticError::ZeroVector));
        assert_eq!(c.stats().misses + c.stats().hits, 0, "refused lookups aren't counted");
        assert_eq!(cosine_similarity(&[0.0, 0.0], &[1.0, 0.0]), None);
        // A NaN threshold keeps the default; out-of-range ones clamp.
        assert_eq!(SemanticConfig::new(1).with_threshold(f32::NAN).threshold, DEFAULT_THRESHOLD);
        assert_eq!(SemanticConfig::new(1).with_threshold(3.0).threshold, 1.0);
        // Large values don't overflow the norm.
        assert!(normalize(&[f32::MAX, f32::MAX]).is_ok());
    }

    #[test]
    fn entries_expire_after_the_ttl() {
        let c: SemanticCache<&str> = SemanticCache::with_config(SemanticConfig::new(8).with_threshold(0.9).with_ttl(Duration::from_secs(60)));
        let t0 = Instant::now();
        c.insert_at("s", &[1.0, 0.0], "old", t0).unwrap();
        c.insert_at("s", &[0.0, 1.0], "new", t0 + Duration::from_secs(30)).unwrap();
        assert!(c.lookup_at("s", &[1.0, 0.0], t0 + Duration::from_secs(59)).unwrap().is_hit());
        assert_eq!(c.lookup_at("s", &[1.0, 0.0], t0 + Duration::from_secs(60)).unwrap(), Lookup::Miss { best: Some(0.0) });
        assert_eq!(c.len(), 1);
        assert_eq!(c.stats().expired, 1);
        // Re-inserting refreshes the age.
        c.insert_at("s", &[0.0, 1.0], "new2", t0 + Duration::from_secs(80)).unwrap();
        assert_eq!(c.lookup_at("s", &[0.0, 1.0], t0 + Duration::from_secs(130)).unwrap().into_value(), Some("new2"));
        assert!(c.lookup_at("s", &[0.0, 1.0], t0 + Duration::from_secs(140)).unwrap().score().is_none());
        assert!(c.is_empty());
    }

    #[test]
    fn evicts_least_recently_used_across_scopes() {
        let c = cache(2, 0.9);
        c.insert("s", &[1.0, 0.0], "a").unwrap();
        c.insert("t", &[0.0, 1.0], "b").unwrap();
        assert!(c.lookup("s", &[1.0, 0.0]).unwrap().is_hit());
        c.insert("s", &[0.0, 1.0], "c").unwrap();
        assert_eq!(c.len(), 2);
        assert!(c.lookup("s", &[1.0, 0.0]).unwrap().is_hit());
        assert_eq!(c.lookup("t", &[0.0, 1.0]).unwrap(), Lookup::Miss { best: None });
        assert_eq!(c.stats().evictions, 1);
    }

    #[test]
    fn same_direction_replaces_instead_of_duplicating() {
        let c = cache(4, 0.9);
        let id = c.insert("s", &[1.0, 1.0], "a").unwrap();
        assert_eq!(c.insert("s", &[3.0, 3.0], "b").unwrap(), id);
        assert_eq!(c.len(), 1);
        assert_eq!(c.lookup("s", &[1.0, 1.0]).unwrap().into_value(), Some("b"));
    }

    #[test]
    fn remove_clear_and_stats() {
        let c = cache(8, 0.9);
        let a = c.insert("s", &[1.0, 0.0], "a").unwrap();
        c.insert("s", &[0.0, 1.0], "b").unwrap();
        c.insert("t", &[0.0, 1.0], "c").unwrap();
        assert_eq!(c.remove(a), Some("a"));
        assert_eq!(c.remove(a), None);
        assert!(!c.lookup("s", &[1.0, 0.0]).unwrap().is_hit());
        assert!(c.lookup("s", &[0.0, 1.0]).unwrap().is_hit());
        let s = c.stats();
        assert_eq!((s.entries, s.hits, s.misses, s.evictions, s.expired), (2, 1, 1, 0, 0));
        assert_eq!(c.remove_scope("t"), 1);
        assert_eq!(c.remove_scope("t"), 0);
        c.clear();
        assert!(c.is_empty());
        assert_eq!(c.stats(), SemanticStats::default());
    }

    #[test]
    fn snapshot_round_trips_answers() {
        let config = SemanticConfig::new(8).with_threshold(0.9).with_ttl(Duration::from_secs(3600));
        let c: SemanticCache = SemanticCache::with_config(config.clone());
        c.insert("s", &[1.0, 0.0], CachedAnswer::new("m", "hello", "stop")).unwrap();
        c.insert("t", &[0.0, 1.0], CachedAnswer::new("m", "other", "stop")).unwrap();
        let bytes = c.snapshot().unwrap();
        let back: SemanticCache = SemanticCache::restore(config, &bytes).unwrap();
        assert_eq!(back.len(), 2);
        assert_eq!(back.lookup("s", &[1.0, 0.01]).unwrap().into_value().map(|a| a.text), Some("hello".into()));
        assert_eq!(back.stats().hits, 1);
        // A smaller cache keeps the most recently used; a fixed dimension skips others.
        let small: SemanticCache = SemanticCache::restore(SemanticConfig::new(1), &bytes).unwrap();
        assert_eq!(small.len(), 1);
        let fixed: SemanticCache = SemanticCache::restore(SemanticConfig::new(8).with_dimension(3), &bytes).unwrap();
        assert!(fixed.is_empty());
        assert!(matches!(SemanticCache::<CachedAnswer>::restore(SemanticConfig::new(1), &[0xff]), Err(SemanticError::Snapshot(_))));
    }

    #[test]
    fn shared_across_threads() {
        let c = std::sync::Arc::new(cache(64, 0.9));
        let handles: Vec<_> = (0..4)
            .map(|i| {
                let c = c.clone();
                std::thread::spawn(move || {
                    let scope = format!("s{i}");
                    c.insert(&scope, &[1.0, i as f32], "v").unwrap();
                    assert!(c.lookup(&scope, &[1.0, i as f32]).unwrap().is_hit());
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(c.len(), 4);
    }
}
