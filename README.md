# weirdnet-semcache

The answer cache for Weirdnet inference nodes. Before an inference node routes a prompt it asks
the cache; afterwards it offers the finished answer. Requests and answers are
[`weirdnet-agent`](https://crates.io/crates/weirdnet-agent) 0.5.1 infer/2 types.

- `AnswerCache`: the seam, `get(model, request)` and `put(model, request, answer)`.
- `CachedAnswer`: text, reasoning, tool calls, finish reason, usage and model; `From<&Answer>`
  for a `weirdnet_agent::remote::pool::Answer`.
- `AnswerStore`: where answers live, opaque bytes under 32-byte keys, so the store can be in
  memory, on disk or shared between nodes. `MemoryStore` is bounded and least-recently-used.
- `ExactCache`: answers only byte-identical **deterministic** requests (temperature 0, no
  tools, inline messages, same model, seed, response format and raw flag), and keeps only answers
  that finished with `stop` or `length`. The key is SHA-256 over the CBOR of those parts.
- `NoCache`: caches nothing.
- `may_read(request)` / `may_store(request)`: the request's own `CacheMode`. `bypass` neither
  reads nor stores; `refresh` skips the read and stores the fresh answer over the old one.
  `ExactCache` honours both.
- `get_rev` / `put_rev` and `exact_key_rev`: key on the model's revision (its weights) as well
  as its id, so a fine-tune served under the same id doesn't replay the old weights' answers.
  With no revision the key is unchanged from 0.2.0.
- `SemanticCache`: similarity lookup by embedding. You embed the prompt; the nearest stored
  embedding in the same scope stands in when its cosine similarity reaches the threshold
  (default 0.95). Embeddings (`&[f32]`, any length) are checked (non-empty, finite, not zero,
  the scope's dimension) and normalized on the way in. `lookup` returns a `Lookup::Hit` with
  the value, score and entry id, or a `Lookup::Miss` with the best score below the threshold.
  Bounded and least-recently-used across scopes, with an optional TTL, `remove`,
  `remove_scope`, `clear`, `stats`, and a CBOR `snapshot` / `restore`. Values default to
  `CachedAnswer` but can be any `Clone` type.
- `scope_key(&[model, revision, system_prompt_hash])`: a scope so answers never cross models,
  weights or system prompts. `cosine_similarity` and `normalize` are public too.

```rust
use weirdnet_semcache::{AnswerCache, CachedAnswer, ExactCache};

let cache = ExactCache::new(256);
if let Some(hit) = cache.get_rev("qwen3-8b", Some(revision), &request) {
    // replay hit.text; mark the answer as a cache hit (infer/2 `Meta::cache_hit`)
}
// after a finished answer:
cache.put_rev("qwen3-8b", Some(revision), &request, &CachedAnswer::from(&answer));
```

Semantic lookup (check `may_read` / `may_store` on infer/2 requests first, as for `ExactCache`):

```rust
use std::time::Duration;
use weirdnet_semcache::{CachedAnswer, Lookup, SemanticCache, SemanticConfig, scope_key};

let cache: SemanticCache = SemanticCache::with_config(
    SemanticConfig::new(1024).with_threshold(0.92).with_ttl(Duration::from_secs(3600)),
);
let scope = scope_key(&["qwen3-8b", revision, &system_prompt_hash]);
match cache.lookup(&scope, &embedding)? {
    Lookup::Hit { value, score, .. } => { /* replay value.text; score is the cosine similarity */ }
    Lookup::Miss { .. } => { /* run the model, then: */
        cache.insert(&scope, &embedding, CachedAnswer::from(&answer))?;
    }
    _ => {}
}
```

## License

BSD-3-Clause.
