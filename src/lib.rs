//! Answer cache for Weirdnet inference nodes.
//!
//! An inference node asks an [`AnswerCache`] before it routes a prompt and offers it the finished
//! answer afterwards. The cache decides what it keeps and when a stored answer may stand in.
//!
//! - [`AnswerCache`]: the seam (`get`, `put`) over `weirdnet-agent` infer/2 [`Request`]s.
//! - [`AnswerStore`]: where cached answers live, as opaque bytes under 32-byte keys, so a store
//!   can be in memory, on disk or shared. [`MemoryStore`] is bounded and least-recently-used.
//! - [`ExactCache`]: answers only requests identical to a stored one that are deterministic
//!   (temperature 0, no tools, inline messages) for the same model, and keeps only answers that
//!   finished (`stop` or `length`).
//! - [`NoCache`]: caches nothing.
//! - [`may_read`] / [`may_store`]: the request's own [`CacheMode`](weirdnet_agent::remote::infer::v2::CacheMode) (`weirdnet-agent` 0.5.1):
//!   `bypass` neither reads nor stores, `refresh` skips the read and stores the fresh answer.
//!   [`ExactCache`] honours both.
//! - Revisions: [`AnswerCache::get_rev`] / [`AnswerCache::put_rev`] and [`exact_key_rev`] key an
//!   answer on the model's weights as well as its id, so a fine-tuned model under the same id
//!   doesn't replay answers from the old weights.
//!
//! Semantic matching (nearest stored prompt by embedding) is not here yet.
#![forbid(unsafe_code)]

use std::collections::HashMap;
use std::sync::Mutex;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use weirdnet_agent::ToolCall;
use weirdnet_agent::remote::infer::v2::{Input, Request, Usage};
use weirdnet_agent::remote::pool::Answer;
use weirdnet_agent::remote::{from_cbor, to_cbor};

/// A stored answer, replayed instead of running the model.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct CachedAnswer {
    /// The model that produced it.
    pub model: String,
    pub text: String,
    #[serde(default)]
    pub reasoning: String,
    #[serde(default)]
    pub tool_calls: Vec<ToolCall>,
    pub finish_reason: Option<String>,
    pub usage: Option<Usage>,
}

impl CachedAnswer {
    pub fn new(model: impl Into<String>, text: impl Into<String>, finish_reason: impl Into<String>) -> Self {
        Self { model: model.into(), text: text.into(), finish_reason: Some(finish_reason.into()), ..Default::default() }
    }

    /// Finished normally (`stop` or `length`), so it is safe to replay.
    pub fn complete(&self) -> bool {
        matches!(self.finish_reason.as_deref(), Some("stop" | "length"))
    }
}

impl From<&Answer> for CachedAnswer {
    fn from(a: &Answer) -> Self {
        Self {
            model: a.model.clone(),
            text: a.text.clone(),
            reasoning: a.reasoning.clone(),
            tool_calls: a.tool_calls.clone(),
            finish_reason: a.finish_reason.clone(),
            usage: a.usage,
        }
    }
}

/// What an inference node asks before and after routing a prompt.
pub trait AnswerCache: Send + Sync {
    /// A stored answer for `request` on `model`.
    fn get(&self, model: &str, request: &Request) -> Option<CachedAnswer>;
    /// Offers a finished answer; the cache decides whether to keep it.
    fn put(&self, model: &str, request: &Request, answer: &CachedAnswer);

    /// A stored answer for `request` on `model` at weights `revision` (a Hugging Face commit, a
    /// file hash, a fine-tune id). With no revision this is [`get`](Self::get). The default
    /// misses when a revision is given, so a cache that doesn't key on revisions never replays
    /// answers from other weights.
    fn get_rev(&self, model: &str, revision: Option<&str>, request: &Request) -> Option<CachedAnswer> {
        match revision {
            None => self.get(model, request),
            Some(_) => None,
        }
    }

    /// Offers a finished answer from `model` at `revision`. The default keeps it only when no
    /// revision is given (see [`get_rev`](Self::get_rev)).
    fn put_rev(&self, model: &str, revision: Option<&str>, request: &Request, answer: &CachedAnswer) {
        if revision.is_none() {
            self.put(model, request, answer);
        }
    }
}

/// The request lets a cache answer it ([`CacheMode::Use`](weirdnet_agent::remote::infer::v2::CacheMode::Use)).
pub fn may_read(request: &Request) -> bool {
    request.cache.may_read()
}

/// The request lets a cache keep its answer ([`CacheMode::Use`](weirdnet_agent::remote::infer::v2::CacheMode::Use) or [`CacheMode::Refresh`](weirdnet_agent::remote::infer::v2::CacheMode::Refresh)).
pub fn may_store(request: &Request) -> bool {
    request.cache.may_store()
}

/// Caches nothing.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoCache;

impl AnswerCache for NoCache {
    fn get(&self, _: &str, _: &Request) -> Option<CachedAnswer> {
        None
    }

    fn put(&self, _: &str, _: &Request, _: &CachedAnswer) {}
}

/// A cache key: SHA-256 over the model and the parts of the request that shape the answer.
pub type Key = [u8; 32];

/// Where cached answers live. Values are opaque bytes; a store may drop entries at any time.
pub trait AnswerStore: Send + Sync {
    fn get(&self, key: &Key) -> Option<Vec<u8>>;
    fn put(&self, key: &Key, value: Vec<u8>);
    fn remove(&self, key: &Key);
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A use counter plus entries, each with its last use.
type Entries = (u64, HashMap<Key, (u64, Vec<u8>)>);

/// In-memory store holding at most `max_entries`, evicting the least recently used.
pub struct MemoryStore {
    max_entries: usize,
    inner: Mutex<Entries>,
}

impl MemoryStore {
    pub fn new(max_entries: usize) -> Self {
        Self { max_entries: max_entries.max(1), inner: Mutex::new((0, HashMap::new())) }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Entries> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }
}

impl AnswerStore for MemoryStore {
    fn get(&self, key: &Key) -> Option<Vec<u8>> {
        let mut g = self.lock();
        g.0 += 1;
        let tick = g.0;
        let hit = g.1.get_mut(key)?;
        hit.0 = tick;
        Some(hit.1.clone())
    }

    fn put(&self, key: &Key, value: Vec<u8>) {
        let mut g = self.lock();
        g.0 += 1;
        let tick = g.0;
        if g.1.len() >= self.max_entries
            && !g.1.contains_key(key)
            && let Some(oldest) = g.1.iter().min_by_key(|(_, v)| v.0).map(|(k, _)| *k)
        {
            g.1.remove(&oldest);
        }
        g.1.insert(*key, (tick, value));
    }

    fn remove(&self, key: &Key) {
        self.lock().1.remove(key);
    }

    fn len(&self) -> usize {
        self.lock().1.len()
    }
}

/// Requests an exact cache may answer: temperature 0, no tools, inline messages.
pub fn deterministic(request: &Request) -> bool {
    request.sampling.temperature == Some(0.0) && request.tools.is_empty() && matches!(request.input, Input::Messages(_))
}

/// The key for `request` on `model`, or `None` when it is not [`deterministic`]. The same as
/// [`exact_key_rev`] with no revision.
pub fn exact_key(model: &str, request: &Request) -> Option<Key> {
    exact_key_rev(model, None, request)
}

/// The key for `request` on `model` at weights `revision`, or `None` when it is not
/// [`deterministic`]. With no revision the key is the 0.2.0 one (`v: 1`), so stored answers
/// still hit; with one it is a `v: 2` key that includes it. The request's [`CacheMode`](weirdnet_agent::remote::infer::v2::CacheMode) is not
/// part of the key, so `refresh` overwrites the entry `use` reads.
pub fn exact_key_rev(model: &str, revision: Option<&str>, request: &Request) -> Option<Key> {
    if !deterministic(request) {
        return None;
    }
    #[derive(Serialize)]
    struct Parts<'a> {
        v: u8,
        model: &'a str,
        #[serde(skip_serializing_if = "Option::is_none")]
        revision: Option<&'a str>,
        input: &'a Input,
        sampling: &'a weirdnet_agent::remote::infer::v2::Sampling,
        seed: Option<u64>,
        response_format: &'a Option<weirdnet_agent::remote::infer::v2::ResponseFormat>,
        raw: bool,
    }
    let parts = Parts {
        v: if revision.is_some() { 2 } else { 1 },
        model,
        revision,
        input: &request.input,
        sampling: &request.sampling,
        seed: request.seed,
        response_format: &request.response_format,
        raw: request.raw,
    };
    let bytes = to_cbor(&parts).ok()?;
    Some(Sha256::digest(&bytes).into())
}

/// Exact-match cache for deterministic requests over an [`AnswerStore`].
pub struct ExactCache<S: AnswerStore = MemoryStore> {
    store: S,
}

impl ExactCache<MemoryStore> {
    /// In memory, at most `max_entries` answers.
    pub fn new(max_entries: usize) -> Self {
        Self { store: MemoryStore::new(max_entries) }
    }
}

impl<S: AnswerStore> ExactCache<S> {
    pub fn with_store(store: S) -> Self {
        Self { store }
    }

    pub fn store(&self) -> &S {
        &self.store
    }

    pub fn len(&self) -> usize {
        self.store.len()
    }

    pub fn is_empty(&self) -> bool {
        self.store.is_empty()
    }
}

impl<S: AnswerStore> ExactCache<S> {
    fn read(&self, key: Option<Key>, request: &Request) -> Option<CachedAnswer> {
        if !may_read(request) {
            return None;
        }
        let key = key?;
        let bytes = self.store.get(&key)?;
        match from_cbor::<CachedAnswer>(&bytes) {
            Ok(a) => Some(a),
            Err(_) => {
                // Written by another version: forget it.
                self.store.remove(&key);
                None
            }
        }
    }

    fn write(&self, key: Option<Key>, request: &Request, answer: &CachedAnswer) {
        if !may_store(request) || !answer.complete() {
            return;
        }
        let Some(key) = key else { return };
        if let Ok(bytes) = to_cbor(answer) {
            self.store.put(&key, bytes);
        }
    }
}

/// Honours [`may_read`] and [`may_store`]; `get_rev` / `put_rev` key on the revision too.
impl<S: AnswerStore> AnswerCache for ExactCache<S> {
    fn get(&self, model: &str, request: &Request) -> Option<CachedAnswer> {
        self.read(exact_key(model, request), request)
    }

    fn put(&self, model: &str, request: &Request, answer: &CachedAnswer) {
        self.write(exact_key(model, request), request, answer);
    }

    fn get_rev(&self, model: &str, revision: Option<&str>, request: &Request) -> Option<CachedAnswer> {
        self.read(exact_key_rev(model, revision, request), request)
    }

    fn put_rev(&self, model: &str, revision: Option<&str>, request: &Request, answer: &CachedAnswer) {
        self.write(exact_key_rev(model, revision, request), request, answer);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use weirdnet_agent::remote::infer::v2::{CacheMode, ContextRef, Sampling};
    use weirdnet_agent::{ContentPart, Message, ToolSpec};

    fn req(temp: f32, text: &str) -> Request {
        let mut s = Sampling::default();
        s.temperature = Some(temp);
        Request::messages(vec![Message::user(text)]).with_sampling(s)
    }

    fn answer() -> CachedAnswer {
        CachedAnswer::new("m", "hi", "stop")
    }

    #[test]
    fn only_deterministic_requests_hit() {
        let c = ExactCache::new(2);
        c.put("m", &req(0.7, "a"), &answer());
        assert!(c.is_empty());
        c.put("m", &req(0.0, "a"), &answer());
        assert_eq!(c.get("m", &req(0.0, "a")), Some(answer()));
        assert!(c.get("other", &req(0.0, "a")).is_none());
        assert!(c.get("m", &req(0.0, "b")).is_none());
        assert!(c.get("m", &req(0.0, "a").with_seed(7)).is_none(), "the seed is part of the key");
    }

    #[test]
    fn tools_and_context_refs_are_never_cached() {
        let tool = ToolSpec::new("t", "a tool");
        let with_tools = req(0.0, "a").with_tools(vec![tool]);
        assert!(!deterministic(&with_tools));
        let mut ctx = Request::context(ContextRef::new("conv", Some("head".into())));
        ctx.sampling.temperature = Some(0.0);
        assert!(!deterministic(&ctx));
        assert!(exact_key("m", &ctx).is_none());
    }

    #[test]
    fn image_parts_are_part_of_the_key() {
        let with = |url: &str| {
            let mut s = Sampling::default();
            s.temperature = Some(0.0);
            let parts = vec![ContentPart::text("what is this?"), ContentPart::image(url)];
            Request::messages(vec![Message::user(parts)]).with_sampling(s)
        };
        let (a, b) = (with("data:image/png;base64,AAAA"), with("data:image/png;base64,BBBB"));
        assert!(deterministic(&a));
        let c = ExactCache::new(2);
        c.put("m", &a, &answer());
        assert_eq!(c.get("m", &a), Some(answer()));
        assert!(c.get("m", &b).is_none(), "a different image is a different request");
        assert!(c.get("m", &req(0.0, "what is this?")).is_none(), "text alone is a different request");
    }

    #[test]
    fn bypass_neither_reads_nor_stores() {
        let c = ExactCache::new(4);
        let bypass = req(0.0, "a").with_cache(CacheMode::Bypass);
        assert!(!may_read(&bypass) && !may_store(&bypass));
        c.put("m", &bypass, &answer());
        assert!(c.is_empty(), "bypass is not stored");
        c.put("m", &req(0.0, "a"), &answer());
        assert_eq!(c.get("m", &req(0.0, "a")), Some(answer()));
        assert!(c.get("m", &bypass).is_none(), "bypass never reads");
        assert_eq!(exact_key("m", &bypass), exact_key("m", &req(0.0, "a")), "the mode is not part of the key");
    }

    #[test]
    fn refresh_overwrites_what_use_reads() {
        let c = ExactCache::new(4);
        let refresh = req(0.0, "a").with_cache(CacheMode::Refresh);
        assert!(!may_read(&refresh) && may_store(&refresh));
        c.put("m", &req(0.0, "a"), &CachedAnswer::new("m", "old", "stop"));
        assert!(c.get("m", &refresh).is_none(), "refresh skips the read");
        c.put("m", &refresh, &CachedAnswer::new("m", "new", "stop"));
        assert_eq!(c.len(), 1);
        assert_eq!(c.get("m", &req(0.0, "a")).map(|a| a.text), Some("new".into()));
        // Also through the revision methods.
        c.put_rev("m", Some("r1"), &refresh, &CachedAnswer::new("m", "r1", "stop"));
        assert_eq!(c.get_rev("m", Some("r1"), &req(0.0, "a")).map(|a| a.text), Some("r1".into()));
        assert!(c.get_rev("m", Some("r1"), &refresh).is_none());
    }

    #[test]
    fn revisions_key_on_the_weights() {
        let r = req(0.0, "a");
        // No revision: the 0.2.0 key, so answers stored before still hit.
        assert_eq!(exact_key_rev("m", None, &r), exact_key("m", &r));
        let v1 = exact_key_rev("m", Some("r1"), &r).unwrap();
        assert_ne!(Some(v1), exact_key("m", &r));
        assert_ne!(Some(v1), exact_key_rev("m", Some("r2"), &r));

        let c = ExactCache::new(4);
        c.put_rev("m", Some("r1"), &r, &CachedAnswer::new("m", "from r1", "stop"));
        assert!(c.get_rev("m", Some("r2"), &r).is_none(), "new weights don't replay old answers");
        assert!(c.get("m", &r).is_none());
        assert_eq!(c.get_rev("m", Some("r1"), &r).map(|a| a.text), Some("from r1".into()));
        c.put_rev("m", None, &r, &answer());
        assert_eq!(c.get("m", &r), Some(answer()));
    }

    #[test]
    fn default_revision_methods_never_cross_weights() {
        struct Plain(ExactCache);
        impl AnswerCache for Plain {
            fn get(&self, model: &str, request: &Request) -> Option<CachedAnswer> {
                self.0.get(model, request)
            }
            fn put(&self, model: &str, request: &Request, answer: &CachedAnswer) {
                self.0.put(model, request, answer)
            }
        }
        let p = Plain(ExactCache::new(4));
        let r = req(0.0, "a");
        p.put_rev("m", Some("r1"), &r, &answer());
        assert!(p.0.is_empty(), "a revision-blind cache keeps nothing for a revision");
        p.put_rev("m", None, &r, &answer());
        assert_eq!(p.get_rev("m", None, &r), Some(answer()));
        assert!(p.get_rev("m", Some("r1"), &r).is_none());
        assert!(NoCache.get_rev("m", None, &r).is_none());
    }

    #[test]
    fn evicts_least_recently_used() {
        let c = ExactCache::new(2);
        c.put("m", &req(0.0, "a"), &answer());
        c.put("m", &req(0.0, "b"), &answer());
        c.get("m", &req(0.0, "a"));
        c.put("m", &req(0.0, "c"), &answer());
        assert!(c.get("m", &req(0.0, "a")).is_some());
        assert!(c.get("m", &req(0.0, "b")).is_none());
        assert_eq!(c.len(), 2);
    }

    #[test]
    fn keeps_only_finished_answers() {
        let c = ExactCache::new(2);
        c.put("m", &req(0.0, "a"), &CachedAnswer::new("m", "h", "cancelled"));
        let mut none = answer();
        none.finish_reason = None;
        c.put("m", &req(0.0, "a"), &none);
        assert!(c.is_empty());
        c.put("m", &req(0.0, "a"), &CachedAnswer::new("m", "h", "length"));
        assert_eq!(c.len(), 1);
    }

    #[test]
    fn unreadable_entries_are_dropped() {
        let c = ExactCache::new(2);
        let key = exact_key("m", &req(0.0, "a")).unwrap();
        c.store().put(&key, vec![0xff, 0x00]);
        assert!(c.get("m", &req(0.0, "a")).is_none());
        assert!(c.is_empty());
    }

    #[test]
    fn from_a_pool_answer() {
        let mut a = Answer::default();
        a.text = "x".into();
        a.model = "m".into();
        a.finish_reason = Some("stop".into());
        let cached = CachedAnswer::from(&a);
        assert!(cached.complete());
        assert_eq!((cached.text.as_str(), cached.model.as_str()), ("x", "m"));
    }
}
