# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0]

Correctness fixes, a smaller published package, and a much stricter build.

### Fixed

- **`close()` silently discarded all data.** `VectorDB.close()` and
  `PartitionedVectorDB.close()` set their closing flag before awaiting the final
  `save()`, and `save()` refused to run once that flag was set — so the "final
  save" never wrote a byte. Three separate layers were involved: the flag
  ordering, an inverted `if (this.isInitialized) return` guard in
  `PartitionedVectorDB.close()` that made closing a healthy database a no-op,
  and `save()` calling `_ensureInitialized()`, which throws while closing.
  `close({ save })` is new for callers that have already flushed state.
- **Filtered HNSW searches always returned no results.** The traversal invoked
  the caller's `filter` with the id only, so a metadata predicate received
  `undefined` and rejected every node. Metadata is now resolved through
  `VectorProvider.getMetadata`.
- **HNSW returned deleted and filtered-out vectors.** The entry point was
  seeded into the result set unconditionally. It is now only seeded when it is
  actually eligible; excluded nodes are still traversed, never returned.
- **HNSW's linear fallback returned soft-deleted vectors** and enforced an
  exact dimension match regardless of the `exactDimensions` option.
- **Deleting one vector destroyed its whole cluster.**
  `ClusteredVectorDB._removeVectorFromCluster` re-read the vector from storage
  after it had already been deleted, always got `undefined`, and took the
  "cluster is empty" branch — orphaning every sibling so they became
  permanently unreachable. `updateVector` also left the id a member of two
  clusters at once.
- **The search cache served results for the wrong filter.** The cache key
  flattened every filter to the string `'present'`, so one predicate's cached
  result was returned for any other. Filters now get distinct identities, and
  the vector hash covers the full vector instead of 16 sampled dimensions.
  The caller's `partitionIds` array is no longer sorted in place.
- **`searchTimeoutMs` was documented but ignored.** It was destructured and
  left as a TODO, so callers who set it got an unbounded search. It now
  applies, with the timer always cleared.
- **Async Express 4 handlers never reached the error middleware.** A throw
  produced an unhandled rejection and the request hung instead of returning
  500. All routes are wrapped.
- **`partition:indexProgress` was declared as `{}`** while the code emitted
  `{ id, progress, operation }`, so the payload was lost to consumers.
- **Cluster "pruning" was a no-op** — every cluster was scanned, making the
  search a full linear scan plus one distance per cluster. New `nprobe` option.
- **`bulkAdd` could spin forever** on a zero-batch-size path that retried
  without changing any state. It now throws.
- **Graph extraction could overflow the stack** (recursive DFS) and
  **stringified numeric ids**, so metadata lookups always missed. It is now
  iterative, keeps the original id type, and is bounded — requests over
  `maxGraphExtractionVectors` are rejected with 413.
- **`req.params.id` was never validated** in the vector routes; a missing id is
  now a 400.

### Added

- `VectorProvider.getMetadata?` so HNSW can evaluate metadata predicates.
- `VectorStoreSearchOptions` — one shared search-options type, replacing the
  `SearchOptions & { exactDimensions?: boolean }` intersections that were
  repeated across HNSW and the store classes.
- `VectorDB.readyPromise` and `ClusteredVectorDB.getStats({ includeClusterMembers })`.
- `SaveOptions` / `CloseOptions`, exposed on `save()` and `close()`.
- `nprobe` on `ClusteredVectorDBOptions`, and `maxGraphExtractionVectors` on
  the server options.
- `.env.example`.

### Changed

- The published tarball is **1.6 MB / 101 files**, down from 71.3 MB / 201
  files. It previously shipped `.env`, the whole test suite, all of `src/`,
  and 67 MB of Windows service daemon logs.
- The publish workflow now builds before publishing and asserts the tarball
  actually contains `dist/index.js` and `dist/index.d.ts`. `dist/` is
  gitignored, so without an explicit build the published `main`/`types`
  pointed at files that did not exist.
- `engines.node` is `>=20.11.0`, matching what the dependency tree actually
  requires. It claimed `>=14`.
- `.env` is no longer tracked in git.

### Removed

- The root `index.js`: dead (`main` points at `dist/index.js`), and it used
  ESM `export *` syntax in a CommonJS package, so running it threw.
- `MetadataIndex` — write-only from the store's perspective and never
  unindexed on delete or update, so it only ever grew.
- `utils/constants.ts` and `server/utils/id_helpers.ts` — unreferenced and not
  part of the public API.
- The `[key: string]: any` index signatures on `SearchResult`,
  `RerankingOptions` and `BatchSearchOptions`, which were hiding real fields.

### Build

- TypeScript 5.4.5 → 5.9.3, with `typescript-eslint` 8.28 → 8.71 and `typedoc`
  0.25 → 0.28. TypeScript 7 is not yet usable: both tools declare peers that
  exclude it.
- Enabled `noUncheckedIndexedAccess`, `exactOptionalPropertyTypes`,
  `noPropertyAccessFromIndexSignature`, `noImplicitOverride`,
  `useUnknownInCatchVariables`, `allowUnreachableCode: false` and
  `allowUnusedLabels: false`.
- ESLint runs the type-aware rules. These found two more unhandled promises:
  `Database.markAsReady()` is async but was never awaited, and
  `scheduleSaveConfigs()` had no rejection handler.
- `npm run lint` is no longer broken (`eslint .ts` was not a glob) and reports
  0 errors, down from 683.
- CI runs on pull requests across Node 20, 22 and 24.
- `npm audit`: 17 findings (2 critical, 8 high) → 0. `qs` was the only one
  reachable from production code, via express.
- `.gitattributes` pins `* text=auto eol=lf`; 49 files were renormalized so
  whole-file line-ending diffs no longer appear in `git status`.

### Testing

- 15 new tests covering `close()` persistence, HNSW filtering, cluster deletion
  and the packed-tarball smoke path.
- 135 passing (was 120).
- Test fixtures are written to `os.tmpdir()` instead of the repository, which
  unblocks `mocha --parallel`.

[Unreleased]: https://github.com/n2flowjs/nbase/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/n2flowjs/nbase/compare/v0.1.10...v0.2.0
