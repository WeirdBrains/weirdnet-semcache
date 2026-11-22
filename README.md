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

```rust
use weirdnet_semcache::{AnswerCache, CachedAnswer, ExactCache};

let cache = ExactCache::new(256);
if let Some(hit) = cache.get_rev("qwen3-8b", Some(revision), &request) {
    // replay hit.text; mark the answer as a cache hit (infer/2 `Meta::cache_hit`)
}
// after a finished answer:
cache.put_rev("qwen3-8b", Some(revision), &request, &CachedAnswer::from(&answer));
```

## License

BSD-3-Clause.
