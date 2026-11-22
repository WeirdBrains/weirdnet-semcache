# Changelog

## [Unreleased]

### Added

- `may_read` and `may_store`: the request's `CacheMode` (weirdnet-agent 0.5.1); `ExactCache` never reads for `bypass` or `refresh` and never stores for `bypass`.
- `AnswerCache::get_rev` / `put_rev` (default methods) and `exact_key_rev`: keys that include the model revision; with no revision the key is unchanged.

### Changed

- Requires weirdnet-agent 0.5.1.

## [0.2.0] - 2026-11-20

### Changed

- Depends on `weirdnet-agent` `"0.5"` (content parts; `Message::content` is a `Content`). Breaking, because `AnswerCache` takes agent `Request`s: move to agent 0.5 with it. Requests with image or audio parts are keyed like text ones.

## [0.1.1] - 2026-11-19

### Changed

- Depends on `weirdnet-agent` `"0.4"` (any 0.4.x, 0.4.1 and later included) instead of exactly `=0.4.0`, so agent patch releases no longer need a weirdnet-semcache release. Raw prompts (agent 0.4.1 `Input::Prompt`) are not cached: `ExactCache` still takes inline messages only.

## [0.1.0] - 2026-11-19

### Added

- `AnswerCache` and `NoCache`, over `weirdnet-agent` 0.4 infer/2 `Request`s.
- `CachedAnswer`, from a pool `Answer`.
- `AnswerStore` and `MemoryStore` (bounded, least-recently-used).
- `ExactCache`: exact matches for deterministic requests (temperature 0, no tools, inline messages); keeps only answers that finished with `stop` or `length`.
