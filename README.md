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

## Pool mode (shared cache)

Several inference nodes in one pool can share one cache (since 0.2.3). Each entry lives on a
**home** node and one **replica**, picked by rendezvous hashing of its scope (or exact key), so a
lookup is one call and nothing is coordinated. All nodes must use the same embedding model and
revision.

- `ShardRing` / `ShardPlan`: the members and, per scope, `plan(scope)` (home and replica) and
  `ranked(scope)`. When a node leaves, only its shards move, to their replica; when one joins,
  it takes over only the shards it now ranks first for, and their old home becomes the replica.
- `SharedCacheReq` / `SharedCacheResp`: the wire protocol on `PROTOCOL`
  (`/weirdnet-semcache/shared/1`), CBOR with a version: `lookup`, `insert` (with the entry's age,
  so a copy keeps its TTL), `remove_scope`, `exact_get` / `exact_put` / `exact_remove`, `ping`.
- `SharedCacheHost`: answers those requests from a node's own `SemanticCache` (and, optionally,
  an `AnswerStore` for exact entries), within `HostLimits` (scope length, dimensions, exact value
  size). It doesn't authorize; the serving side does.
- With the **`pool`** feature (no new dependencies beyond what `weirdnet-agent` already pulls in):
  - `pool::PooledSemanticCache`: one node's view. `lookup` tries a local L1 copy, then the home
    (its own shard when it is the home), then the replica when the home fails. `insert` writes
    its own shard when it holds the shard, the home (waited for, bounded) and the replica (in the
    background; `settle()` waits). `remove_scope` reaches every member. Every remote call has a
    timeout, a failing node is skipped for `down_for`, and when no owner answers a lookup is a
    `PoolLookup::Miss { degraded: true }`: **a cache outage never blocks inference.** Optional
    read repair (`ask_replica_on_miss`) copies a replica's hit back to a new home. `exact_get` /
    `exact_put` share exact entries the same way. `stats()` counts hits by source, misses,
    timeouts, errors and skips.
  - `pool::CacheTransport`: how a request reaches a node. `pool::DialerTransport` does it over a
    `weirdnet-agent` `Dialer`, and `pool::serve(channel, caller, auth, host)` answers on the other
    end after `auth` (for Weirdnet, the pool's `RosterAuth`) lets the caller in.

```rust
use std::sync::Arc;
use weirdnet_semcache::pool::{DialerTransport, PoolConfig, PoolLookup, PooledSemanticCache};
use weirdnet_semcache::{SemanticCache, SemanticConfig, scope_key};

let shard = Arc::new(SemanticCache::with_config(SemanticConfig::new(4096).with_threshold(0.92)));
let cache: PooledSemanticCache = PooledSemanticCache::new(my_peer_id, shard, Arc::new(DialerTransport::new(dialer)), PoolConfig::new());
cache.set_nodes(inference_node_peer_ids); // this node included
// serve cache.host() on weirdnet_semcache::PROTOCOL with weirdnet_semcache::pool::serve
let scope = scope_key(&["qwen3-8b", revision, &system_prompt_hash]);
match cache.lookup(&scope, &embedding).await? {
    PoolLookup::Hit { value, .. } => { /* replay value.text */ }
    _ => { /* run the model, then: */ cache.insert(&scope, &embedding, CachedAnswer::from(&answer)).await?; }
}
```

## License

BSD-3-Clause.
