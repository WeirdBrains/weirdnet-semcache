# Changelog

## [Unreleased]

### Added

- `AnswerCache` and `NoCache`, over `weirdnet-agent` 0.4 infer/2 `Request`s.
- `CachedAnswer`, from a pool `Answer`.
- `AnswerStore` and `MemoryStore` (bounded, least-recently-used).
- `ExactCache`: exact matches for deterministic requests (temperature 0, no tools, inline messages); keeps only answers that finished with `stop` or `length`.
