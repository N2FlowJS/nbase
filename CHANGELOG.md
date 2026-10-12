# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Five defects found reviewing the 0.3.0 release: two silent data-loss paths, one
process-killing event, one erased field, and a cache that served the wrong
payload. Each has a regression test that fails on 0.3.0 and passes here.

### Fixed

- **Evicting a partition deleted its unpersisted vectors.** The LRU disposer
  closed the partition with `save: false`, and `save()` only walks the *loaded*
  partitions — so once a partition left the cache nothing would ever write it
  again. On `partitionCapacity: 2, maxActivePartitions: 1` with 6 vectors, only
  **2 of 6** survived on disk. Eviction now persists before releasing, and
  `close()` awaits the write: lru-cache v11 never awaits a disposer
  (`Disposer` is typed `(value, key, reason) => void` and the return value is
  dropped), which is also what the old comment claimed otherwise.
- **A `partitionIds` search silently skipped the partitions it named.**
  `findNearest` filtered the caller's list against `loadedPartitions`, so a
  request for a non-resident partition returned results from whatever happened
  to be resident, or `[]` — indistinguishable from "no matches". An explicit
  list is now loaded and searched; unknown ids are reported and dropped. A
  search with no list still covers only the resident partitions, and now logs
  that it did so instead of passing it off as a full search.
- **`VectorDBMonitor` emitted `error` with no listener.** `recordError` emits
  `error`, which Node treats specially: with no listener the emit throws
  `ERR_UNHANDLED_ERROR`. Every `partition:error` is routed into it, so the
  first partition error threw out of whatever code path raised it. The monitor's
  errors are now forwarded to the `Database`'s own `error` event.
- **`updateVector` erased the vector's metadata.** `ClusteredVectorDB` updated
  by deleting and re-adding the vector, and `deleteVector` removes the metadata
  entry; the re-add did not supply it again. Every update silently dropped that
  vector's metadata, which then also excluded it from any metadata filter. It
  now updates in place, so metadata and dimension survive and an update no
  longer emits `vector:delete`.
- **The search cache ignored `includeMetadata` and shared its result objects.**
  The key was hand-listed and omitted the flag, and the cache is consulted
  *before* `Search` attaches metadata, so a query primed with
  `includeMetadata: false` satisfied a later `includeMetadata: true` query with
  metadata-free results. The key is now derived from the whole option object, so
  a newly added option is in the key by default, and both the stored entries and
  the returned results are copied — a caller mutating what it got back could
  rewrite the cached payload.

- **An absolute `persistence.dbPath` was rewritten into the project directory.**
  `Database` joined it onto `process.cwd()`, and `path.join(cwd, '/tmp/my-db')`
  is `<cwd>/tmp/my-db`, so an absolute path produced a directory inside the
  package instead of the one that was asked for. Now `path.resolve`, which
  honours an absolute path and still resolves a relative one against the cwd.

### Fixed

- **`efSearch` was never read.** `ef` was always `max(k, this.efSearch)`, so the
  documented accuracy dial did nothing: `efSearch: 1` and `efSearch: 500`
  returned the same list. It now sets the beam width — on a 4000x128 index,
  200 queries went from 0.37 ms to 32.35 ms across that range.
- **`distanceMetric` was ignored on the HNSW path.** The distance function is
  fixed when the index is built, so a `cosine` request was answered with
  Euclidean distances in Euclidean order: measured recall against the true
  cosine ranking was 0.80, and it is 1.00 now. The per-search function is
  threaded through the traversal rather than swapping shared state, so
  concurrent searches cannot change each other's metric.
- **`exactDimensions: true` returned `[]` after every `buildIndex()`.** The
  per-dimension entry point was only written by the `dimensionSet.size === 1`
  branch of `addVector`, and the build's dimension pre-scan had already filled
  every group before the first `addVector` ran. The pre-scan seeds it now.
- **A query could be answered from the wrong dimension group.** With
  `dimensionAware` on, the descent started at the global entry point, which
  belongs to whichever dimension was indexed first, so a query in another
  dimension returned confidently wrong-dimension hits whenever those were
  closer in raw distance. It now starts at that dimension's entry point.
- **An id of `0` disabled the index.** `if (!this.entryPointId)` and
  `if (dimEntryPoint)` are truthiness tests, and `0` is a valid id:
  `findNearest` silently answered from the linear fallback — which returns
  exact results, so nothing looked wrong — and the per-dimension entry point
  was never promoted past the level-0 node it started on.
- **Distance kernels are typed against `DistanceMetric`** instead of `string`,
  and the metric a per-search function implements is detected by comparison.
- **HNSW prunes with Algorithm 4 instead of nearest-M.** Keeping the `M` nearest
  neighbours builds a kNN graph — every node points inward at its own cluster,
  with no long-range links, which is the property a small-world graph needs to
  be navigable. Recall@10 over six seeded queries on 8000 vectors of dimension
  64, both variants built from the same data in the same process, reproduced
  over two runs: **0.917 average / 0.800 worst before, 0.967 / 0.900 after.**
  `keepPrunedConnections` is off; refilling with the rejected candidates
  re-introduces the clustering the heuristic removes. This is a benchmark
  result rather than a CI gate, for the `Math.random()` reason above — the suite
  pins the recall floor instead.
- **Tombstones survive a save/load.** `serialize()` omitted `deletedNodes`, so a
  reloaded index reported `deletedNodesCount: 0` while the graph still carried
  the deleted nodes and their edges, and every query kept paying to traverse
  them.

Recall@10 measured with the new oracle: 1.00 at 500 vectors of dimension 16
(every run), and 0.95-0.98 average / 0.90 worst at 2000x128.

### Known

- **`_randomLevel()` uses `Math.random()`, so two builds of the same data
  produce different graphs** and one build's recall@10 is not reproducible.
  Every figure quoted here is several seeded queries against one build, and
  comparing *builds* of two variants has to happen in the same process or the
  delta is noise. That is not hypothetical: the first measurement of the
  pruning heuristic below said it was worse, and only a same-process comparison
  showed the opposite.

### Added

- `test/test-helpers/search-oracle.ts` — a seeded PRNG, a float64 reference
  distance, a brute-force top-k and a recall metric. Every existing search
  assertion was `length > 0`, so an implementation returning `k` arbitrary
  vectors passed.
- `test/tests/search_oracle.test.ts` — the exact paths must match the reference
  id for id; the approximate ones must reach a stated recall; the library's
  kernels must agree with the reference; a deleted vector never comes back.
- `test/tests/hnsw_correctness.test.ts` — `efSearch`, the metric override, the
  per-dimension entry point, `exactDimensions` after a build, and id `0`.

### Security

Hardening of the optional REST API. `createServer` was the only code that
assembled the real Express app, and it had no tests, so none of this was
covered.

- **The server bound every interface.** `app.listen(PORT)` ignores the
  configured host, so a process configured for `localhost` — the default — was
  reachable from the network while logging `http://localhost:PORT`. The host is
  passed through now.
- **CORS is opt-in.** It was mounted as a bare `cors()`, answering every origin
  with `Access-Control-Allow-Origin: *`. That is not access control, and it
  defeated an api-key middleware added through `middleware` — a key held by a
  browser app is readable by any site that can reach the API. Pass
  `cors: ['https://app.example']`, or `cors: true` to restore the old
  behaviour.
- **API keys (opt-in).** `auth: { apiKeys: [...] }` protects `/api/*` via
  `Authorization: Bearer` or `x-api-key`, compared with `timingSafeEqual` over
  SHA-256 digests. With no keys configured the API stays open and the server logs
  a warning saying so; `auth: { requireAuth: true }` makes it refuse to start
  instead.
- **A 500 no longer leaks internal messages.** Every route answered
  `res.status(500).json({ error: error.message })`, and those strings come from
  the storage layer — `ENOENT: /home/user/.../partitions/vec.bin` is what a
  client got, which is a map of the host. Outside `NODE_ENV=development` the
  client now gets `Internal server error`.
- **`k` is validated.** `k: "abc"` produced `NaN`, HNSW derives its beam width as
  `Math.max(k, efSearch)`, and `results.size >= NaN` is never true — so one
  request scanned the whole index. `k: 1e9` did the same directly. `k`,
  `efSearch`, `limit` and `rerankLambda` are now bounded integers/numbers on
  `/search`, `/similar` and `/search/metadata`.
- **`threshold` is capped on the O(n^2) endpoints.** `/relationships` and
  `/communities` only checked `> 0`; a threshold of 1e9 makes every pair related,
  so 10k vectors meant 50M result objects and an OOM kill — which also loses
  whatever the database had not saved yet.
- **Vector payloads are checked.** `new Float32Array(["1", null])` stored NaN and
  answered 200, poisoning every later distance for that vector.
- **Ids are checked.** An object or `null` id became a `Map` key across
  `memoryStorage`, `metadata` and `vectorDimensions`, and because
  `typeof id === 'number'` was false the auto-id counter never advanced, so the
  next generated id silently overwrote it.
- **`__proto__` is stripped from metadata** before it is persisted and echoed.
- **A metadata criteria no longer matches inherited keys.** `field in meta` walks
  the prototype chain, so `{"criteria": "constructor"}` matched every stored
  record. `Object.hasOwn` now.
- **`/stats` is rate-limited.** It is mounted at `/`, outside the `/api` limiter,
  and it is an O(all vectors) call that stringifies every metadata object.
- **`limit` is honoured by the metadata search.** Only the object-criteria branch
  applied it; the string/array branch returned every match from every loaded
  partition.
- **Middleware order.** The 404 handler now runs before the error handler, so a
  throw after `createServer` returns reaches the app's handler instead of
  Express's default one, which answers with a stack trace. A malformed JSON body
  is reported as 400 instead of 500.

### Added

- `test/api/server_defaults.test.ts` — the assembled app: headers, CORS, auth,
  error handling, payload validation, limiter scope.
- `test/api/validation.test.ts` — the validators, including the NaN and
  out-of-range cases that were denial of service.
- `src/server/utils/validation.ts` and `src/server/utils/http_error.ts`.
- `trustProxy` option; without it the rate limiter trusts a client-supplied
  `X-Forwarded-For` by default, which was the case before it existed.

- `test/database/database.test.ts` — the `Database` facade had no tests at all;
  nothing in the suite ever constructed one, and the coverage tool reported it
  as 100%.
- `test/tests/partitioned_vector_db.eviction.test.ts` — eviction durability,
  restart, and `partitionIds` coverage.

## [0.3.0] - 2026-10-02

Breaking renames plus the removal of every explicit `any` from `src/`. The
published package was still 0.1.10, so everything in 0.2.0 and this release
arrives together.

### Changed

- **BREAKING: `UnifiedSearch` is now `Search`**, in `src/search/search.ts`
  (was `src/search/unified_search.ts`). The public export, the class name and
  every JSDoc reference changed with it.
- **BREAKING: `UnifiedSearchOptions` is now `SearchOptions`.** The old
  `SearchOptions` was a thin wrapper over it, so the two were merged: the
  backward-compatibility aliases `limit`, `offset` and `stopEarly` moved into
  the base interface. `HybridSearchOptions` and `BatchQuery.options` follow
  the merged type.
- **BREAKING: `UnifiedSearchPartitionedStats` is now `SearchStats`**, and
  `Database.getUnifiedSearch()` is now `Database.getSearch()`.
- **BREAKING: metadata is `Record<string, unknown>`**, not
  `Record<string, any>`, everywhere on the public surface (`SearchResult`,
  `addVector`/`updateMetadata`/`getMetadata`, filter predicates, the server
  routes, `RerankingOptions.metadata`/`metadataMap`). Reading a metadata value
  now requires narrowing, which is the point: the values came from `JSON.parse`
  and were never checked.
- Query helpers take `values?: unknown` instead of `any | any[]`, and
  `FilterConfig.value` is `unknown`.
- `PartitionedVectorDBInterface` is fully typed: `search`/`findNearest` take
  `SearchOptions`, `createPartition` takes the new `CreatePartitionOptions`,
  `getActivePartition()` returns `ClusteredVectorDB | null` and
  `getPartitionConfigs()` returns `PartitionConfig[]`.
- `TypedEventEmitter`'s constraint is `object` rather than
  `Record<string, unknown>`, which interfaces without an index signature
  cannot satisfy.
- The `search:complete` payload has one shape (`SearchCompleteEvent`) shared by
  the engine and `Database`, instead of two that disagreed.
- The `Search` engine's `search:complete` event no longer carries `timeMs`: the
  underlying `partition:indexed` event never had it, so it was always
  `undefined`.

### Fixed

- **The monitor recorded the search method as `undefined`.** `Database`
  read `data.dbMethodUsed` off the engine's payload, which is called `method`,
  so every `methodUsage` entry was keyed `"undefined"`.
- **Filters compared across types.** `$gt`/`$gte`/`$lt`/`$lte` used the raw
  `>` operator, so a string metadata value was coerced to a number and compared
  against a numeric filter. Comparisons are now type-bracketed (both numbers or
  both strings), the way MongoDB does it; mixed types never match.
- **`recordError()` spread a primitive `extraData`** character by character
  into the event payload. Only objects are spread now.
- **`partition:indexProgress.progress` is documented as a fraction (0-1)**, not
  a percentage, which is what HNSW reports.
- The HNSW worker restored its state through `as any`; it now assigns the
  private fields directly, and the compiler checks the field names.

### Removed

- `UnifiedSearchStats` — superseded by `SearchStats` and referenced nowhere.
- 227 `any` occurrences across 19 files, and the 2 unused
  `eslint-disable @typescript-eslint/no-unused-vars` directives. `npm run lint`
  now reports 0 problems, down from 229 warnings.
- Dead code: `VectorDB.getStats()` re-checked for a `clusters.dimensions` field
  its own return type requires, `HybridEngineSearch` re-derived `progress`/`id`/
  `error` from payloads that are already typed, and
  `getMetadataWithFieldAcrossPartitions` called `getVector(null)`.

### Added

- Plain repository/npm/stars/forks/issues/license links in place of the five
  shields.io badges. npmjs.com rendered the badge images as alt text on the
  package page, so every one of them was a broken image for anyone reading the
  package there.
- `utils/errors.ts` — `toError`, `errorMessage` and `errorCode`, so
  `useUnknownInCatchVariables` no longer forces an `any` annotation on every
  catch block (`error.code === 'ENOENT'` in particular).
- `LICENSE` (MIT, 2025 n2flowjs). `package.json` declared `"license": "MIT"`
  and shipped `LICENSE` in `files`, and the README links to it, but the file
  was never committed — npm warns on publish and TypeDoc could not resolve the
  link.

### Fixed

- **The release workflow could not publish.** `.github/workflows/npm.yml` ran
  its "Check npm authentication" step under `set -euo pipefail` and echoed
  `$GITHUB_WORKFLOW_PATH` inside the diagnostic message. That variable does not
  exist in the Actions environment, so `set -u` aborted the step on its own
  error output, before `npm publish` was ever reached. Build, tests and the
  tarball checks all passed first. The message now uses `$GITHUB_WORKFLOW` and
  an explicit `WORKFLOW_FILE`.
- `npm run docs` reported 21 warnings, now 0: 17 uses of the unsupported
  `@fires` block tag (TypeDoc's tag is `@event`, so the emitted-event list was
  silently dropped from the API docs), plus an `@description` block tag, a
  `@constructor` tag, and a `@param options.force` that did not match the
  `save(options: SaveOptions)` signature.
- 64 tracked files carried CRLF in the working tree while `.gitattributes`
  pins `* text=auto eol=lf` and every committed blob is LF, so `git status` and
  `npm run format:check` disagreed with the repository. Working tree
  normalized to LF; `.prettierrc` was committed with CRLF and is now LF too.
- `npm run format:check` now passes. 42 files had drifted from the prettier
  config in `.prettierrc` — quote style, wrapping, trailing commas — and
  nothing caught it because CI never runs the check. Formatting only, no
  behaviour change.

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
