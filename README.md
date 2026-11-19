# weirdnet-semcache

The answer cache for Weirdnet inference nodes. Before an inference node routes a prompt it asks
the cache; afterwards it offers the finished answer. Requests and answers are
[`weirdnet-agent`](https://crates.io/crates/weirdnet-agent) 0.4 infer/2 types.

- `AnswerCache`: the seam, `get(model, request)` and `put(model, request, answer)`.
- `CachedAnswer`: text, reasoning, tool calls, finish reason, usage and model; `From<&Answer>`
  for a `weirdnet_agent::remote::pool::Answer`.
- `AnswerStore`: where answers live, opaque bytes under 32-byte keys, so the store can be in
  memory, on disk or shared between nodes. `MemoryStore` is bounded and least-recently-used.
- `ExactCache`: answers only byte-identical **deterministic** requests (temperature 0, no
  tools, inline messages, same model, seed, response format and raw flag), and keeps only answers
  that finished with `stop` or `length`. The key is SHA-256 over the CBOR of those parts.
- `NoCache`: caches nothing.

```rust
use weirdnet_semcache::{AnswerCache, CachedAnswer, ExactCache};

let cache = ExactCache::new(256);
if let Some(hit) = cache.get("qwen3-8b", &request) {
    // replay hit.text; mark the answer as a cache hit (infer/2 `Meta::cache_hit`)
}
// after a finished answer:
cache.put("qwen3-8b", &request, &CachedAnswer::from(&answer));
```

Not here yet: semantic matching (nearest stored prompt by embedding, with a distance threshold)
and shared or persistent stores.

## License

BSD-3-Clause.
