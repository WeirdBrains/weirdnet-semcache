# Handoff

Running log for [weirdnet-semcache](https://github.com/WeirdBrains/weirdnet-semcache), newest first. Shared rules: the collective `AGENTS.md` and this repo's [AGENTS.md](AGENTS.md).

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
