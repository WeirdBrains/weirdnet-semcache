//! The shared-cache wire protocol and its serving side: [`SharedCacheReq`], [`SharedCacheResp`]
//! and [`SharedCacheHost`].
//!
//! Inference nodes in one pool share a cache by sending each other these messages (CBOR, one
//! request and one response per exchange, over [`PROTOCOL`]). A node answers for the shards it
//! holds (see [`ShardRing`](crate::ShardRing)) from its own [`SemanticCache`] and, optionally, an
//! [`AnswerStore`] for exact entries. The transport, and who may call, belong to the app; with
//! the `pool` feature, `pool::serve` and `pool::DialerTransport` run it over `weirdnet-agent`
//! channels and `pool::PooledSemanticCache` is the asking side.

use std::fmt;
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use weirdnet_agent::remote::{RemoteError, from_cbor, to_cbor};

use crate::semantic::{Lookup, SemanticCache};
use crate::{AnswerStore, CachedAnswer, Key};

/// The protocol name nodes serve the shared cache on.
pub const PROTOCOL: &str = "/weirdnet-semcache/shared/1";

/// The message version this crate writes and reads.
pub const WIRE_VERSION: u8 = 1;

/// A shared-cache request, sent to the node that holds the shard.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum SharedCacheReq<V = CachedAnswer> {
    /// The most similar entry in `scope` ([`SemanticCache::lookup`]).
    Lookup { scope: String, embedding: Vec<f32> },
    /// Store `value` under `embedding` in `scope` ([`SemanticCache::insert`]). `age_ms` is how old
    /// the entry already is (0 for a fresh answer), so a copied entry keeps its TTL.
    Insert {
        scope: String,
        embedding: Vec<f32>,
        value: V,
        #[serde(default)]
        age_ms: u64,
    },
    /// Drop every entry in `scope` ([`SemanticCache::remove_scope`]).
    RemoveScope { scope: String },
    /// The exact entry under `key` ([`AnswerStore::get`]).
    ExactGet {
        #[serde(with = "wire_bytes::key")]
        key: Key,
    },
    /// Store an exact entry ([`AnswerStore::put`]).
    ExactPut {
        #[serde(with = "wire_bytes::key")]
        key: Key,
        #[serde(with = "wire_bytes::vec")]
        value: Vec<u8>,
    },
    /// Drop an exact entry ([`AnswerStore::remove`]).
    ExactRemove {
        #[serde(with = "wire_bytes::key")]
        key: Key,
    },
    /// Liveness and size.
    Ping,
}

/// The answer to a [`SharedCacheReq`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum SharedCacheResp<V = CachedAnswer> {
    /// A [`Lookup`](SharedCacheReq::Lookup) reached the threshold. `embedding` is the stored
    /// entry's (normalized), so the asker can keep a local copy under the same direction.
    Hit {
        value: V,
        score: f32,
        #[serde(default)]
        embedding: Vec<f32>,
    },
    /// A [`Lookup`](SharedCacheReq::Lookup) found nothing at the threshold; `best` is the best
    /// score below it.
    Miss { best: Option<f32> },
    /// An insert or exact put was taken (a store may still drop it at any time).
    Stored,
    /// A removal; `count` entries went.
    Removed { count: u64 },
    /// An [`ExactGet`](SharedCacheReq::ExactGet).
    Exact {
        #[serde(with = "wire_bytes::opt")]
        value: Option<Vec<u8>>,
    },
    /// A [`Ping`](SharedCacheReq::Ping); `entries` is the node's semantic entry count.
    Pong { entries: u64 },
    /// The node refused or couldn't serve the request (bad input, no exact store, not
    /// authorized, an unknown request from a newer version).
    Error { message: String },
}

impl<V> SharedCacheResp<V> {
    pub fn error(message: impl Into<String>) -> Self {
        Self::Error { message: message.into() }
    }
}

#[derive(Serialize)]
struct WireOut<'a, T> {
    v: u8,
    body: &'a T,
}

#[derive(Deserialize)]
struct WireIn<T> {
    body: T,
}

#[derive(Deserialize)]
struct Head {
    v: u8,
}

fn encode<T: Serialize>(body: &T) -> Result<Vec<u8>, RemoteError> {
    to_cbor(&WireOut { v: WIRE_VERSION, body })
}

fn decode<T: DeserializeOwned>(bytes: &[u8]) -> Result<T, RemoteError> {
    let head: Head = from_cbor(bytes)?;
    if head.v != WIRE_VERSION {
        return Err(RemoteError::Unsupported { message: format!("shared cache message version {}", head.v) });
    }
    Ok(from_cbor::<WireIn<T>>(bytes)?.body)
}

impl<V: Serialize> SharedCacheReq<V> {
    /// CBOR, with the [`WIRE_VERSION`].
    pub fn encode(&self) -> Result<Vec<u8>, RemoteError> {
        encode(self)
    }
}

impl<V: DeserializeOwned> SharedCacheReq<V> {
    /// From [`encode`](Self::encode). Another version is [`RemoteError::Unsupported`]; anything
    /// unreadable (including a request kind from a newer version) is [`RemoteError::Protocol`].
    pub fn decode(bytes: &[u8]) -> Result<Self, RemoteError> {
        decode(bytes)
    }
}

impl<V: Serialize> SharedCacheResp<V> {
    /// CBOR, with the [`WIRE_VERSION`].
    pub fn encode(&self) -> Result<Vec<u8>, RemoteError> {
        encode(self)
    }
}

impl<V: DeserializeOwned> SharedCacheResp<V> {
    /// From [`encode`](Self::encode), like [`SharedCacheReq::decode`].
    pub fn decode(bytes: &[u8]) -> Result<Self, RemoteError> {
        decode(bytes)
    }
}

/// What a [`SharedCacheHost`] accepts from other nodes.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct HostLimits {
    /// Longest scope, in bytes ([`scope_key`](crate::scope_key) gives 64).
    pub max_scope_len: usize,
    /// Most embedding dimensions.
    pub max_dimension: usize,
    /// Largest exact value, in bytes.
    pub max_exact_value: usize,
}

impl HostLimits {
    /// 512-byte scopes, 16384 dimensions, 8 MiB exact values.
    pub fn new() -> Self {
        Self { max_scope_len: 512, max_dimension: 16384, max_exact_value: 8 << 20 }
    }

    pub fn with_max_scope_len(mut self, n: usize) -> Self {
        self.max_scope_len = n;
        self
    }

    pub fn with_max_dimension(mut self, n: usize) -> Self {
        self.max_dimension = n;
        self
    }

    pub fn with_max_exact_value(mut self, n: usize) -> Self {
        self.max_exact_value = n;
        self
    }
}

impl Default for HostLimits {
    fn default() -> Self {
        Self::new()
    }
}

/// Answers shared-cache requests from this node's own cache: the shards it holds.
///
/// It has no notion of who is asking; authorize callers before handing it a request (the `pool`
/// feature's `serve` does).
pub struct SharedCacheHost<V = CachedAnswer> {
    semantic: Arc<SemanticCache<V>>,
    exact: Option<Arc<dyn AnswerStore>>,
    limits: HostLimits,
}

impl<V> Clone for SharedCacheHost<V> {
    fn clone(&self) -> Self {
        Self { semantic: self.semantic.clone(), exact: self.exact.clone(), limits: self.limits.clone() }
    }
}

impl<V> fmt::Debug for SharedCacheHost<V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SharedCacheHost").field("exact", &self.exact.is_some()).field("limits", &self.limits).finish_non_exhaustive()
    }
}

impl<V: Clone> SharedCacheHost<V> {
    /// Serves `semantic`, with no exact store and default [`HostLimits`].
    pub fn new(semantic: Arc<SemanticCache<V>>) -> Self {
        Self { semantic, exact: None, limits: HostLimits::new() }
    }

    /// Serves exact entries from `store` too.
    pub fn with_exact(mut self, store: Arc<dyn AnswerStore>) -> Self {
        self.exact = Some(store);
        self
    }

    pub fn with_limits(mut self, limits: HostLimits) -> Self {
        self.limits = limits;
        self
    }

    pub fn semantic(&self) -> &Arc<SemanticCache<V>> {
        &self.semantic
    }

    pub fn exact(&self) -> Option<&Arc<dyn AnswerStore>> {
        self.exact.as_ref()
    }

    pub fn limits(&self) -> &HostLimits {
        &self.limits
    }

    fn check_scope(&self, scope: &str) -> Result<(), String> {
        if scope.len() > self.limits.max_scope_len {
            return Err(format!("scope is {} bytes, at most {}", scope.len(), self.limits.max_scope_len));
        }
        Ok(())
    }

    fn check_embedding(&self, embedding: &[f32]) -> Result<(), String> {
        if embedding.len() > self.limits.max_dimension {
            return Err(format!("embedding has {} dimensions, at most {}", embedding.len(), self.limits.max_dimension));
        }
        Ok(())
    }

    /// Answers one request.
    pub fn handle(&self, req: SharedCacheReq<V>) -> SharedCacheResp<V> {
        match self.try_handle(req) {
            Ok(resp) => resp,
            Err(message) => SharedCacheResp::Error { message },
        }
    }

    fn try_handle(&self, req: SharedCacheReq<V>) -> Result<SharedCacheResp<V>, String> {
        let exact = || self.exact.as_ref().ok_or_else(|| "this node keeps no exact entries".to_string());
        Ok(match req {
            SharedCacheReq::Lookup { scope, embedding } => {
                self.check_scope(&scope)?;
                self.check_embedding(&embedding)?;
                match self.semantic.lookup_with_vector(&scope, &embedding, Instant::now()).map_err(|e| e.to_string())? {
                    (Lookup::Hit { value, score, .. }, vector) => SharedCacheResp::Hit { value, score, embedding: vector.unwrap_or_default() },
                    (Lookup::Miss { best }, _) => SharedCacheResp::Miss { best },
                }
            }
            SharedCacheReq::Insert { scope, embedding, value, age_ms } => {
                self.check_scope(&scope)?;
                self.check_embedding(&embedding)?;
                let now = Instant::now();
                let age = Duration::from_millis(age_ms);
                if self.semantic.config().ttl.is_some_and(|ttl| age >= ttl) {
                    // Already expired here: taken and dropped, like any entry a cache may drop.
                    return Ok(SharedCacheResp::Stored);
                }
                let stored = now.checked_sub(age).unwrap_or(now);
                self.semantic.insert_at(&scope, &embedding, value, stored).map_err(|e| e.to_string())?;
                SharedCacheResp::Stored
            }
            SharedCacheReq::RemoveScope { scope } => SharedCacheResp::Removed { count: self.semantic.remove_scope(&scope) as u64 },
            SharedCacheReq::ExactGet { key } => SharedCacheResp::Exact { value: exact()?.get(&key) },
            SharedCacheReq::ExactPut { key, value } => {
                if value.len() > self.limits.max_exact_value {
                    return Err(format!("exact value is {} bytes, at most {}", value.len(), self.limits.max_exact_value));
                }
                exact()?.put(&key, value);
                SharedCacheResp::Stored
            }
            SharedCacheReq::ExactRemove { key } => {
                let store = exact()?;
                let had = store.get(&key).is_some();
                store.remove(&key);
                SharedCacheResp::Removed { count: u64::from(had) }
            }
            SharedCacheReq::Ping => SharedCacheResp::Pong { entries: self.semantic.len() as u64 },
        })
    }
}

impl<V: Clone + Serialize + DeserializeOwned> SharedCacheHost<V> {
    /// Answers one encoded request with an encoded response. Unreadable requests get a
    /// [`SharedCacheResp::Error`] too, so the asker learns why.
    pub fn handle_bytes(&self, bytes: &[u8]) -> Result<Vec<u8>, RemoteError> {
        let resp = match SharedCacheReq::<V>::decode(bytes) {
            Ok(req) => self.handle(req),
            Err(e) => SharedCacheResp::Error { message: e.to_string() },
        };
        resp.encode()
    }
}

/// Serde helpers that write bytes as CBOR byte strings.
mod wire_bytes {
    use serde::de::{self, SeqAccess, Visitor};
    use serde::{Deserializer, Serializer};

    struct BytesVisitor;

    impl<'de> Visitor<'de> for BytesVisitor {
        type Value = Vec<u8>;

        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            f.write_str("a byte string")
        }

        fn visit_bytes<E: de::Error>(self, v: &[u8]) -> Result<Vec<u8>, E> {
            Ok(v.to_vec())
        }

        fn visit_byte_buf<E: de::Error>(self, v: Vec<u8>) -> Result<Vec<u8>, E> {
            Ok(v)
        }

        fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Vec<u8>, A::Error> {
            let mut out = Vec::new();
            while let Some(b) = seq.next_element::<u8>()? {
                out.push(b);
            }
            Ok(out)
        }
    }

    pub mod vec {
        use super::*;

        pub fn serialize<S: Serializer>(v: &[u8], s: S) -> Result<S::Ok, S::Error> {
            s.serialize_bytes(v)
        }

        pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<u8>, D::Error> {
            d.deserialize_byte_buf(BytesVisitor)
        }
    }

    pub mod key {
        use super::*;
        use crate::Key;

        pub fn serialize<S: Serializer>(v: &Key, s: S) -> Result<S::Ok, S::Error> {
            s.serialize_bytes(v)
        }

        pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Key, D::Error> {
            let v = d.deserialize_byte_buf(BytesVisitor)?;
            let n = v.len();
            v.try_into().map_err(|_| de::Error::invalid_length(n, &"32 bytes"))
        }
    }

    pub mod opt {
        use serde::{Deserialize, Deserializer, Serializer};

        struct Wrap<'a>(&'a [u8]);

        impl serde::Serialize for Wrap<'_> {
            fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
                s.serialize_bytes(self.0)
            }
        }

        #[derive(Deserialize)]
        struct Owned(#[serde(with = "super::vec")] Vec<u8>);

        pub fn serialize<S: Serializer>(v: &Option<Vec<u8>>, s: S) -> Result<S::Ok, S::Error> {
            match v {
                Some(b) => s.serialize_some(&Wrap(b)),
                None => s.serialize_none(),
            }
        }

        pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Option<Vec<u8>>, D::Error> {
            Ok(Option::<Owned>::deserialize(d)?.map(|o| o.0))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MemoryStore, SemanticConfig};

    fn host() -> SharedCacheHost<String> {
        let semantic = Arc::new(SemanticCache::with_config(SemanticConfig::new(8).with_threshold(0.9)));
        SharedCacheHost::new(semantic).with_exact(Arc::new(MemoryStore::new(8)))
    }

    fn round_trip_req(req: SharedCacheReq<String>) {
        let bytes = req.encode().unwrap();
        assert_eq!(SharedCacheReq::<String>::decode(&bytes).unwrap(), req);
    }

    fn round_trip_resp(resp: SharedCacheResp<String>) {
        let bytes = resp.encode().unwrap();
        assert_eq!(SharedCacheResp::<String>::decode(&bytes).unwrap(), resp);
    }

    #[test]
    fn every_message_round_trips() {
        let key = [7u8; 32];
        round_trip_req(SharedCacheReq::Lookup { scope: "s".into(), embedding: vec![1.0, -0.5] });
        round_trip_req(SharedCacheReq::Insert { scope: "s".into(), embedding: vec![1.0], value: "v".into(), age_ms: 12 });
        round_trip_req(SharedCacheReq::RemoveScope { scope: "s".into() });
        round_trip_req(SharedCacheReq::ExactGet { key });
        round_trip_req(SharedCacheReq::ExactPut { key, value: vec![1, 2, 3] });
        round_trip_req(SharedCacheReq::ExactRemove { key });
        round_trip_req(SharedCacheReq::Ping);
        round_trip_resp(SharedCacheResp::Hit { value: "v".into(), score: 0.97, embedding: vec![0.6, 0.8] });
        round_trip_resp(SharedCacheResp::Miss { best: Some(0.5) });
        round_trip_resp(SharedCacheResp::Miss { best: None });
        round_trip_resp(SharedCacheResp::Stored);
        round_trip_resp(SharedCacheResp::Removed { count: 3 });
        round_trip_resp(SharedCacheResp::Exact { value: Some(vec![9; 40]) });
        round_trip_resp(SharedCacheResp::Exact { value: None });
        round_trip_resp(SharedCacheResp::Pong { entries: 4 });
        round_trip_resp(SharedCacheResp::error("no"));
    }

    #[test]
    fn answers_round_trip_as_the_default_value() {
        let resp: SharedCacheResp = SharedCacheResp::Hit { value: CachedAnswer::new("m", "hello", "stop"), score: 1.0, embedding: vec![1.0] };
        assert_eq!(SharedCacheResp::<CachedAnswer>::decode(&resp.encode().unwrap()).unwrap(), resp);
    }

    #[test]
    fn bytes_are_byte_strings() {
        let req: SharedCacheReq<String> = SharedCacheReq::ExactPut { key: [0xab; 32], value: vec![0xcd; 64] };
        let bytes = req.encode().unwrap();
        // As byte strings, 32 + 64 bytes plus small headers; as arrays of numbers it would be ~2x.
        assert!(bytes.len() < 140, "{}", bytes.len());
    }

    #[test]
    fn other_versions_and_garbage_are_refused() {
        #[derive(Serialize)]
        struct Future {
            v: u8,
            body: SharedCacheReq<String>,
        }
        let bytes = to_cbor(&Future { v: 2, body: SharedCacheReq::Ping }).unwrap();
        assert!(matches!(SharedCacheReq::<String>::decode(&bytes), Err(RemoteError::Unsupported { .. })));
        assert!(matches!(SharedCacheReq::<String>::decode(&[0xff, 0x00]), Err(RemoteError::Protocol { .. })));
        #[derive(Serialize)]
        struct Unknown {
            v: u8,
            body: &'static str,
        }
        let newer = to_cbor(&Unknown { v: 1, body: "compact" }).unwrap();
        assert!(matches!(SharedCacheReq::<String>::decode(&newer), Err(RemoteError::Protocol { .. })));
        // The host answers unreadable requests with an error it can read back.
        let resp = SharedCacheResp::<String>::decode(&host().handle_bytes(&newer).unwrap()).unwrap();
        assert!(matches!(resp, SharedCacheResp::Error { .. }));
    }

    #[test]
    fn the_host_serves_its_cache() {
        let h = host();
        let insert =
            |scope: &str, e: Vec<f32>, v: &str| h.handle(SharedCacheReq::Insert { scope: scope.into(), embedding: e, value: v.into(), age_ms: 0 });
        assert_eq!(insert("s", vec![2.0, 0.0], "a"), SharedCacheResp::Stored);
        assert_eq!(insert("t", vec![0.0, 1.0], "b"), SharedCacheResp::Stored);
        match h.handle(SharedCacheReq::Lookup { scope: "s".into(), embedding: vec![1.0, 0.05] }) {
            SharedCacheResp::Hit { value, score, embedding } => {
                assert_eq!(value, "a");
                assert!(score > 0.99);
                assert_eq!(embedding, vec![1.0, 0.0], "the stored entry's normalized embedding");
            }
            other => panic!("{other:?}"),
        }
        assert!(matches!(h.handle(SharedCacheReq::Lookup { scope: "t".into(), embedding: vec![1.0, 0.0] }), SharedCacheResp::Miss { best: Some(_) }));
        assert_eq!(h.handle(SharedCacheReq::Lookup { scope: "u".into(), embedding: vec![1.0, 0.0] }), SharedCacheResp::Miss { best: None });
        assert_eq!(h.handle(SharedCacheReq::Ping), SharedCacheResp::Pong { entries: 2 });
        assert_eq!(h.handle(SharedCacheReq::RemoveScope { scope: "s".into() }), SharedCacheResp::Removed { count: 1 });
        assert_eq!(h.semantic().len(), 1);
        // Bad embeddings and dimension mismatches are errors, not misses.
        assert!(matches!(insert("t", vec![1.0, 0.0, 0.0], "c"), SharedCacheResp::Error { .. }));
        assert!(matches!(insert("t", vec![f32::NAN, 1.0], "c"), SharedCacheResp::Error { .. }));
    }

    #[test]
    fn the_host_serves_exact_entries() {
        let h = host();
        let key = [1u8; 32];
        assert_eq!(h.handle(SharedCacheReq::ExactGet { key }), SharedCacheResp::Exact { value: None });
        assert_eq!(h.handle(SharedCacheReq::ExactPut { key, value: vec![5] }), SharedCacheResp::Stored);
        assert_eq!(h.handle(SharedCacheReq::ExactGet { key }), SharedCacheResp::Exact { value: Some(vec![5]) });
        assert_eq!(h.handle(SharedCacheReq::ExactRemove { key }), SharedCacheResp::Removed { count: 1 });
        assert_eq!(h.handle(SharedCacheReq::ExactRemove { key }), SharedCacheResp::Removed { count: 0 });
        let plain: SharedCacheHost<String> = SharedCacheHost::new(Arc::new(SemanticCache::new(1)));
        assert!(matches!(plain.handle(SharedCacheReq::ExactGet { key }), SharedCacheResp::Error { .. }));
    }

    #[test]
    fn the_host_enforces_its_limits() {
        let h = host().with_limits(HostLimits::new().with_max_scope_len(4).with_max_dimension(2).with_max_exact_value(2));
        let long = SharedCacheReq::Lookup { scope: "scope".into(), embedding: vec![1.0] };
        assert!(matches!(h.handle(long), SharedCacheResp::Error { .. }));
        let wide = SharedCacheReq::Insert { scope: "s".into(), embedding: vec![1.0; 3], value: "v".into(), age_ms: 0 };
        assert!(matches!(h.handle(wide), SharedCacheResp::Error { .. }));
        assert!(matches!(h.handle(SharedCacheReq::ExactPut { key: [0; 32], value: vec![0; 3] }), SharedCacheResp::Error { .. }));
        assert!(h.semantic().is_empty() && h.exact().unwrap().is_empty());
        assert_eq!(h.limits().max_dimension, 2);
    }

    #[test]
    fn copied_entries_keep_their_age() {
        let semantic = Arc::new(SemanticCache::with_config(SemanticConfig::new(8).with_ttl(Duration::from_secs(60))));
        let h: SharedCacheHost<String> = SharedCacheHost::new(semantic);
        let old = SharedCacheReq::Insert { scope: "s".into(), embedding: vec![1.0], value: "v".into(), age_ms: 60_000 };
        assert_eq!(h.handle(old), SharedCacheResp::Stored);
        assert!(h.semantic().is_empty(), "already past the TTL");
        let young = SharedCacheReq::Insert { scope: "s".into(), embedding: vec![1.0], value: "v".into(), age_ms: 1_000 };
        assert_eq!(h.handle(young), SharedCacheResp::Stored);
        assert_eq!(h.semantic().len(), 1);
    }
}
