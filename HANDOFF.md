# Handoff

Running log for [weirdnet-semcache](https://github.com/WeirdBrains/weirdnet-semcache), newest first. Shared rules: the collective `AGENTS.md` and this repo's [AGENTS.md](AGENTS.md).

## 2026-11-27: released 0.2.3

What changed: released with `scripts/weirdnet_release.py --patch` (additive; `cargo semver-checks` against 0.2.2 found no semver update required, default and all features). Release commit [`0da67b9`](https://github.com/WeirdBrains/weirdnet-semcache/commit/0da67b93bba96536c809a5252a560003581eeae3), tag [`v0.2.3`](https://github.com/WeirdBrains/weirdnet-semcache/releases/tag/v0.2.3), on [crates.io](https://crates.io/crates/weirdnet-semcache/0.2.3) (features `default` and `pool`) and [docs.rs](https://docs.rs/weirdnet-semcache/0.2.3). The [site](https://net.weirdbrains.com/crates/semcache) shows 0.2.3 with a pool-mode guide section.

Checks run: the release script's checks (fmt, clippy `-D warnings`, tests, docs), `cargo publish` verification, CI on the pool-mode commit and the release commit (green on Linux, macOS and Windows), the docs.rs page for `pool::PooledSemanticCache`.

Not tested: no real network, relay or live inference node.

What's left and next: Ymir M6 wires it (serve `PooledSemanticCache::host()` on `PROTOCOL` behind its roster auth, members from the pool's inference nodes).

## 2026-11-27: pool mode, a shared cache across inference nodes (for 0.2.3)

What changed: inference nodes in one pool can share one cache, as [Ymir](https://github.com/WeirdBrains/ymir) M6 asks (its `DESIGN.md` "Shared semantic cache": rendezvous hashing to a home node plus one replica, one call per lookup). In the default build: `ShardRing` / `ShardPlan` (rendezvous hashing over SHA-256, score pinned by a test), the CBOR wire protocol `SharedCacheReq` / `SharedCacheResp` on `PROTOCOL` = `/weirdnet-semcache/shared/1` with `WIRE_VERSION` 1, and `SharedCacheHost` / `HostLimits`. Behind the new opt-in `pool` feature: `pool::PooledSemanticCache` (L1, home, replica on failure, per-call timeouts, down-node skipping, background replica writes with `settle()`, optional read repair, exact entries, `PoolStats`), `pool::CacheTransport`, `pool::DialerTransport` and `pool::serve` over `weirdnet-agent` channels. The `pool` feature adds `tokio` and `async-trait`, both already in the tree through `weirdnet-agent`; the default build adds nothing. `ExactCache`, `SemanticCache` and snapshots are unchanged. Additive, so a patch release, 0.2.3.

Checks run: `cargo fmt --check`, `cargo clippy --all-targets` with and without `--all-features` (`-D warnings`), `cargo test` (default, `--all-features`, `--no-default-features`: 50 tests, 27 new, including 13 multi-node tests over in-process channels: sharing hits, replica failover when the home dies, writes while the home is down, scope isolation, TTL, eviction, stalled and failing peers, refused callers, removal on every member, read repair after a join, exact entries), `cargo doc --no-deps` with `-D warnings` (with and without features), `cargo package --locked`, `cargo semver-checks` against 0.2.2 (default and all features: no semver update required).

Not tested: no real network, relay or live inference node; no load test of lookup latency or of a large pool.

What's left and next: release 0.2.3 once CI on main is green, then add pool mode to the [site](https://net.weirdbrains.com/crates/semcache). Entries don't move when the pool changes (only read repair with `ask_replica_on_miss`); a bulk hand-off can be added later if Ymir needs it.

## 2026-11-26: released 0.2.2

What changed: released with `scripts/weirdnet_release.py --patch` (additive; `cargo semver-checks` found no semver update required). Release commit [`afcd9a7`](https://github.com/WeirdBrains/weirdnet-semcache/commit/afcd9a78847cd886cc1085827922c7c2b5e08634), tag [`v0.2.2`](https://github.com/WeirdBrains/weirdnet-semcache/releases/tag/v0.2.2), on [crates.io](https://crates.io/crates/weirdnet-semcache/0.2.2) and [docs.rs](https://docs.rs/weirdnet-semcache/0.2.2). The [site](https://net.weirdbrains.com/crates/semcache) shows 0.2.2.

Checks run: the release script's checks (fmt, clippy `-D warnings`, tests, docs), `cargo publish` verification, CI on the release commit (green on Linux, macOS and Windows), the docs.rs page for `SemanticCache`.

Not tested: no live node or real embedding model.

What's left and next: nothing for this release.

## 2026-11-26: `SemanticCache`, similarity lookup by embedding (for 0.2.2)

What changed: `SemanticCache<V = CachedAnswer>` ([`f20cdf4`](https://github.com/WeirdBrains/weirdnet-semcache/commit/f20cdf42621f6c7c124f8806b0ff61df33f20fdf)): the caller embeds the prompt; the nearest stored embedding in the same scope stands in when its cosine similarity reaches the threshold (default 0.95). `SemanticConfig` (max entries, threshold, optional TTL, optional fixed dimension), `Lookup::Hit { value, score, id }` / `Lookup::Miss { best }`, `SemanticError`, `SemanticStats`, least-recently-used eviction across scopes, `remove` / `remove_scope` / `clear`, and a CBOR `snapshot` / `restore`. Helpers `scope_key`, `cosine_similarity`, `normalize`. No new dependencies; `ExactCache` and every existing API are unchanged. Requested for [Ymir](https://github.com/WeirdBrains/ymir). CI now runs on Linux, macOS and Windows and builds docs with `-D warnings` ([`b6fbb6a`](https://github.com/WeirdBrains/weirdnet-semcache/commit/b6fbb6a64959b9c0bfbc524caf5686a1a2b6c84a)). Additive, so a patch release, 0.2.2.

Checks run: `cargo fmt --check`, `cargo clippy --all-targets` with and without `--all-features` (`-D warnings`), `cargo test` (default, `--all-features`, `--no-default-features`: 23 tests, 12 new), `cargo doc --no-deps` with `-D warnings`, `cargo semver-checks` (no semver update required), `cargo package --locked`.

Not tested: no live inference node or real embedding model; no benchmark of lookup cost (a linear scan per scope, meant for cache-sized entry counts).

What's left and next: release 0.2.2 once CI on main is green, check crates.io and docs.rs, and add `SemanticCache` to the [site](https://net.weirdbrains.com).

## 2026-11-22: released 0.2.1

What changed: released with `scripts/weirdnet_release.py` (approved by Michael, 2026-11-22). Release commit [`e482c3a`](https://github.com/WeirdBrains/weirdnet-semcache/commit/e482c3a4b78abd5846217116c5885f63b14ec872), tag [`v0.2.1`](https://github.com/WeirdBrains/weirdnet-semcache/releases/tag/v0.2.1), on [crates.io](https://crates.io/crates/weirdnet-semcache/0.2.1) and [docs.rs](https://docs.rs/weirdnet-semcache/0.2.1).

Checks run: the release script's checks (fmt, clippy `-D warnings`, tests, docs), `cargo publish` verification, CI on the release commit (green), the docs.rs pages for the new APIs.

Not tested: no live node.

What's left and next: nothing for this release; the [site pages](https://net.weirdbrains.com) are merged.

## 2026-11-22: ready for 0.2.1 (merge of `cache-mode`)

What changed: reverted the branch-only CI commit ([`748847c`](https://github.com/WeirdBrains/weirdnet-semcache/commit/748847c5bf76259fe8bca5a082ce6cca712d0216)) and moved to [weirdnet-agent 0.5.1](https://crates.io/crates/weirdnet-agent/0.5.1) from crates.io with no `[patch.crates-io]` ([`05fd535`](https://github.com/WeirdBrains/weirdnet-semcache/commit/05fd535db287bdbb41ed2081e70401f2ac76334c)); merged into main with `--no-ff`. Release approved by Michael (2026-11-22).

Checks run: `cargo fmt --check`, `cargo clippy --all-targets` with and without `--all-features` (`-D warnings`), `cargo test --all-features` (11 tests), `cargo test --no-default-features`, `cargo doc` with `-D warnings`, `cargo package --locked` (builds the packaged crate against crates.io).

Not tested: no live inference node.

What's left and next: release 0.2.1 from the merge commit once CI on main is green, then check crates.io and docs.rs.

## 2026-11-22: honour the request's cache mode; revision keys (branch `cache-mode`, for 0.2.1)

What changed ([`e1a0a2d`](https://github.com/WeirdBrains/weirdnet-semcache/commit/e1a0a2d4ae07011d79efcce7a1cfe1a5d540ab32)): `may_read(&Request)` and `may_store(&Request)` follow [weirdnet-agent](https://github.com/WeirdBrains/weirdnet-agent) 0.5.1's `Request::cache`; `ExactCache` never reads for `bypass` or `refresh` and never stores for `bypass`, so `refresh` overwrites the entry `use` reads. `AnswerCache::get_rev` / `put_rev` (default methods: they miss and keep nothing when a revision is given, so a revision-blind cache never crosses weights) and `exact_key_rev` key an answer on the model revision too. With no revision the key is byte-for-byte the 0.2.0 one (`v: 1`); with one it is `v: 2`. Requested for [Ymir](https://github.com/WeirdBrains/ymir)'s training evals. Additive (new functions and default trait methods), so a patch release, 0.2.1.

Branch only ([`18bf20f`](https://github.com/WeirdBrains/weirdnet-semcache/commit/18bf20ff2fc48c992271e7d1ef6c86c8fbd59ddb) and `Cargo.toml`): a `[patch.crates-io]` git dependency on weirdnet-agent `64fadb3`, CI on this branch, and `cargo package` skipped, because weirdnet-agent 0.5.1 isn't on crates.io yet.

Checks run: `cargo fmt --check`, `cargo clippy --all-targets` with and without `--all-features` (`-D warnings`), `cargo test --all-features` (11 tests), `cargo test --no-default-features`, `cargo doc` with `-D warnings`, all against the patched weirdnet-agent.

Not tested: `cargo package` (it ignores `[patch]` and needs weirdnet-agent 0.5.1 from crates.io).

What's left and next: after weirdnet-agent 0.5.1 is published, revert the CI commit, require `weirdnet-agent = "0.5.1"`, drop the `[patch.crates-io]` section, run the checks and `cargo package`, merge to main and release 0.2.1 once Michael OKs it.
