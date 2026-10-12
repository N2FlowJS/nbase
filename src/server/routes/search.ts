import { Router, Request, Response } from 'express';
import { ApiContext, DistanceMetric, RerankingMethod, SearchResult, SearchOptions } from '../../types';
import { asyncHandler } from '../utils/async_handler';
import { publicErrorMessage } from '../utils/http_error';
import { MAX_EF_SEARCH, MAX_GRAPH_THRESHOLD, MAX_K, MAX_LIMIT, InvalidRequestError, requireInteger, requireNumber, requireVector } from '../utils/validation';

/**
 * Configures and returns the search-related routes for the API.
 *
 * @param context - The API context containing dependencies such as the database, timer, and filter creation function.
 *
 * @returns An Express router with the following endpoints:
 *
 * ### POST `/`
 * Performs a nearest neighbor search using the provided query vector(s) and options.
 *
 * Request Body:
 * - `query` (array): The query vector(s) to search for. **Required**.
 * - `k` (number): The number of nearest neighbors to retrieve.
 * - `method` (string): The search method to use (`hnsw` or `clustered`).
 * - `partitionIds` (array): Optional array of partition IDs to restrict the search.
 * - `efSearch` (number): The efSearch parameter for HNSW search.
 * - `distanceMetric` (string): The distance metric to use for the search.
 * - `rerank` (boolean): Whether to apply reranking to the results.
 * - `rerankingMethod` (string): The method to use for reranking.
 * - `rerankLambda` (number): The lambda parameter for reranking.
 * - `filters` (object | array): Filters to apply to the search results.
 * - `includeMetadata` (boolean): Whether to include metadata in the results.
 * - `includeVectors` (boolean): Whether to include vectors in the results.
 * - `skipCache` (boolean): Whether to skip the cache for this search.
 * - `searchTimeoutMs` (number): Timeout for the search operation in milliseconds.
 *
 * Response:
 * - `results` (array): The search results.
 * - `count` (number): The number of results returned.
 * - `duration` (number): The time taken to perform the search in milliseconds.
 * - `searchOptions` (object): Details about the search options used.
 *
 * ### POST `/metadata`
 * Searches for metadata entries that match specific criteria across all loaded partitions.
 *
 * Request Body:
 * - `criteria` (string | array | object): The criteria to match. **Required**.
 * - `values` (any): Optional values to match (used with string/array criteria).
 * - `includeVectors` (boolean): Whether to include vectors in the response.
 * - `limit` (number): Optional limit on the number of results to return.
 *
 * Response:
 * - `results` (array): The metadata entries matching the criteria.
 * - `count` (number): The number of results returned.
 * - `duration` (number): The time taken to perform the search in milliseconds.
 *
 * Errors:
 * - Returns a 400 status code for invalid requests (e.g., missing or invalid `query` or `criteria`).
 * - Returns a 500 status code for internal server errors, with optional stack trace in development mode.
 *
 * ### POST `/relationships`
 * Finds relationships between vectors based on a distance threshold.
 *
 * Request Body:
 * - `threshold` (number): The maximum distance between vectors to consider them related. **Required**.
 * - `metric` (string): Distance metric to use (e.g., 'cosine', 'euclidean'). Default depends on database implementation.
 * - `partitionIds` (array): Optional array of partition IDs to restrict the relationship search.
 *
 * Response:
 * - `relationships` (array): Array of relationships between vectors.
 * - `count` (number): The number of relationships found.
 * - `duration` (number): The time taken to extract relationships in milliseconds.
 *
 * Errors:
 * - Returns a 400 status code for invalid requests (e.g., missing or invalid `threshold`).
 * - Returns a 500 status code for internal server errors, with optional stack trace in development mode.
 */
/** Narrows an untrusted value to the metrics the store supports. */
function isDistanceMetric(value: unknown): value is DistanceMetric {
  return value === 'euclidean' || value === 'cosine';
}

/** Narrows an untrusted value to the reranking methods `Search` implements. */
function isRerankingMethod(value: unknown): value is RerankingMethod {
  return value === 'diversity' || value === 'standard' || value === 'weighted';
}

export function searchRoutes(context: ApiContext) {
  const router = Router();
  const { database, createTimer, createFilterFunction, maxGraphExtractionVectors } = context;
  /**
   * Nearest neighbor search endpoint
   *
   * Searches for the k-nearest neighbors of a given query vector across all loaded partitions.
   *
   * Request body:
   * - query: Array of query vectors (required)
   * - k: Number of nearest neighbors to retrieve
   * - method: Search method to use (e.g., 'hnsw' or 'clustered')
   * - partitionIds: Optional array of partition IDs to restrict the search
   * - efSearch: efSearch parameter for HNSW search
   * - distanceMetric: Distance metric to use for the search
   * - rerank: Whether to apply reranking to the results
   * - rerankingMethod: Method to use for reranking
   * - rerankLambda: Lambda parameter for reranking
   * - filters: Filters to apply to the search results (optional)
   * - includeMetadata: Whether to include metadata in the results
   * - includeVectors: Whether to include vectors in the results
   * - skipCache: Whether to skip the cache for this search
   * - searchTimeoutMs: Timeout for the search operation in milliseconds
   * * Example:
   * ```json
   * {
   *   "query": [[0.1, 0.2, 0.3]],
   *  "k": 5,
   *  "method": "hnsw",
   *  "partitionIds": ["partition1", "partition2"],
   *  "efSearch": 200,
   * "distanceMetric": "euclidean",
   * "rerank": true,
   * "rerankingMethod": "cosine",
   * "rerankLambda": 0.5,
   * "filters": { "field": "value" },
   * "includeMetadata": true,
   * "includeVectors": false,
   * "skipCache": false,
   * "searchTimeoutMs": 5000
   * }
   * @returns Array of nearest neighbors with metadata and vector IDs
   * * Example response:
   * ```json
   * {
   *   "results": [
   *                {
   *                  "partitionId": "partition1",
   *                  "vectorId": 123,
   *                  "distance": 0.1,
   *                  "metadata": { "field": "value" }
   *                },
   *                {
   *                  "partitionId": "partition2",
   *                  "vectorId": 456,
   *                  "distance": 0.2,
   *                  "metadata": { "field": "value" }
   *                }
   *              ],
   *    "count": 2,
   *    "duration": 1234,
   *    "searchOptions": {
   *        "k": 5,
   *        "method": "hnsw",
   *        "partitionsSearched": ["partition1", "partition2"],
   *        "rerankApplied": true,
   *        "cacheUsed": true,
   *        "filtersApplied": true
   *     }
   *
   */
  router.post(
    '/',
    asyncHandler(async function (req: Request, res: Response) {
      const timer = createTimer();
      timer.start('search');

      const { query, method, partitionIds, distanceMetric, rerank, rerankingMethod, filters = {}, includeMetadata, includeVectors, skipCache } = req.body;

      if (!query || !Array.isArray(query)) {
        res.status(400).json({
          error: 'Invalid request: query vector array is required',
        });
        return;
      }

      // `k` and `efSearch` are forwarded to the search engine, and HNSW derives
      // its beam width as `Math.max(k, efSearch)`. A non-integer `k` (or one of
      // a billion) used to turn every query into a full-index scan, so both are
      // bounded here rather than trusted.
      let k: number | undefined;
      let efSearch: number | undefined;
      let rerankLambda: number | undefined;
      let searchTimeoutMs: number | undefined;
      try {
        requireVector(query, 'query');
        k = requireInteger(req.body['k'], 'k', 1, MAX_K);
        efSearch = requireInteger(req.body['efSearch'], 'efSearch', 1, MAX_EF_SEARCH);
        rerankLambda = requireNumber(req.body['rerankLambda'], 'rerankLambda', 0, 1);
        searchTimeoutMs = requireInteger(req.body['searchTimeoutMs'], 'searchTimeoutMs', 1, 600_000);
      } catch (error) {
        if (error instanceof InvalidRequestError) {
          res.status(error.status).json({ error: error.message });
          return;
        }
        throw error;
      }

      try {
        const hasFilters = Array.isArray(filters) ? filters.length > 0 : Object.keys(filters).length > 0;

        // Build comprehensive search options. Optional keys are spread in only
        // when present: an explicit `undefined` is not the same as an absent key
        // under exactOptionalPropertyTypes, and `Search` defaults a missing key
        // while an explicit undefined would reach the engine as a real value.
        const searchOptions: SearchOptions = {
          useHNSW: method === 'hnsw',
          // Only include filter if filters are present
          filter: hasFilters ? createFilterFunction(filters) : undefined,
          ...(k !== undefined ? { k } : {}),
          ...(efSearch !== undefined ? { efSearch } : {}),
          ...(rerankLambda !== undefined ? { rerankLambda } : {}),
          ...(searchTimeoutMs !== undefined ? { searchTimeoutMs } : {}),
          ...(includeMetadata !== undefined ? { includeMetadata: includeMetadata === true } : {}),
          ...(includeVectors !== undefined ? { includeVectors: includeVectors === true } : {}),
          ...(skipCache !== undefined ? { skipCache: skipCache === true } : {}),
          ...(rerank !== undefined ? { rerank: rerank === true } : {}),
          ...(isDistanceMetric(distanceMetric) ? { distanceMetric } : {}),
          ...(isRerankingMethod(rerankingMethod) ? { rerankingMethod } : {}),
          ...(Array.isArray(partitionIds) && partitionIds.every((id) => typeof id === 'string') ? { partitionIds: partitionIds as string[] } : {}),
        };

        const results: SearchResult[] = await database.findNearest(query, k, searchOptions);

        const duration = timer.stop('search').total;

        // Return enhanced response with details
        res.json({
          results,
          count: results.length,
          duration,
          searchOptions: {
            k,
            method: method === 'hnsw' ? 'hnsw' : 'clustered',
            partitionsSearched: partitionIds?.length || 'all',
            rerankApplied: rerank,
            cacheUsed: !skipCache,
            filtersApplied: hasFilters,
          },
        });
        return;
      } catch (error) {
        const duration = timer.stop('search').total;
        console.error('Search error:', error);

        // Provide detailed error response
        res.status(500).json({
          error: publicErrorMessage(error),
          stack: process.env['NODE_ENV'] === 'development' ? (error as Error).stack : undefined,
          duration,
        });
        return;
      }
    }),
  );
  /**
   * Metadata search endpoint
   *
   * Searches for metadata entries that match specific criteria across all loaded partitions.
   *
   * Request body:
   * - criteria: String, array of strings, or object with field-value pairs
   * - values: Optional values to match (used with string/array criteria)
   * - includeVectors: Whether to include vectors in the response
   * - limit: Optional limit on the number of results to return
   * Example
   * - Get all metadata entries with 'category' field across partitions
   * POST body
   * ```json
   * {
   *   "criteria":"category"
   * }
   * ```
   *
   * - Get entries where 'status' equals 'active' across partitions
   * POST body
   * ```json
   * {
   *   "criteria":"category",
   *   "values":"active"
   * }
   * ```
   *
   * - Get entries with both 'author' and 'title' fields across partitions
   * POST body

   * ```json
   * {
   *   "criteria":["author", "title"]
   * }
   * ```
   * - Get entries where 'type' is 'book' AND 'published' is true across partitions
   * POST body
   * ```json
   * {
   *   "criteria":["type", "published"],
   *   "values": ["book", true]
   * }
   * - Using object syntax (recommended): type='book' AND published=true
   * const publishedBooks = await db.getMetadataWithFieldAcrossPartitions({ type: 'book', published: true });
   * POST body
   * ```json
   * {
   *   "criteria":{ "type": "book", "published": true },
   * }
   * @returns Array of metadata entries matching the criteria with partition and vector IDs
   */
  router.post(
    '/metadata',
    asyncHandler(async function (req: Request, res: Response) {
      const timer = createTimer();
      timer.start('metadata_search');

      const { criteria, values, includeVectors = false } = req.body;

      // Validate that criteria is provided and is of the right type
      if (criteria === undefined || (typeof criteria !== 'string' && !Array.isArray(criteria) && (typeof criteria !== 'object' || criteria === null))) {
        res.status(400).json({
          error: 'Invalid request: criteria must be a string, array of strings, or object with field-value pairs',
        });
        return;
      }

      // `limit` was forwarded unvalidated, and the legacy string/array branch of
      // `getMetadataWithField` ignores it entirely — so `{"criteria":"x"}`
      // returned every matching record across every loaded partition.
      let limit: number | undefined;
      try {
        limit = requireInteger(req.body['limit'], 'limit', 1, MAX_LIMIT);
      } catch (error) {
        if (error instanceof InvalidRequestError) {
          res.status(error.status).json({ error: error.message });
          return;
        }
        throw error;
      }

      try {
        // Call the database method to get metadata with matching fields
        const results = await database.getMetadataWithField(criteria, values, limit !== undefined ? { limit } : {});

        // If includeVectors is true, fetch vectors for each result
        if (includeVectors) {
          const resultsWithVectors = await Promise.all(
            results.map(async (item: { partitionId: string; vectorId: number | string; metadata: Record<string, unknown> }) => {
              const vectorData = await database.getVector(item.vectorId);
              return {
                ...item,
                vector: vectorData ? vectorData.vector : null,
              };
            }),
          );

          const duration = timer.stop('metadata_search').total;
          res.json({
            results: resultsWithVectors,
            count: resultsWithVectors.length,
            duration,
          });
          return;
        }

        const duration = timer.stop('metadata_search').total;
        res.json({
          results,
          count: results.length,
          duration,
        });
      } catch (error) {
        const duration = timer.stop('metadata_search').total;
        console.error('Metadata search error:', error);

        res.status(500).json({
          error: publicErrorMessage(error),
          stack: process.env['NODE_ENV'] === 'development' ? (error as Error).stack : undefined,
          duration,
        });
      }
    }),
  );

  /**
   * Vector relationships extraction endpoint
   *
   * Finds relationships between vectors based on a distance threshold.
   *
   * Request body:
   * - threshold: The maximum distance between vectors to consider them related (required)
   * - metric: Distance metric to use (e.g., 'cosine', 'euclidean')
   * - partitionIds: Optional array of partition IDs to restrict the relationship search
   * - `includeMetadata` (boolean): Whether to include metadata for each vector in the results.
   *
   * Example:
   * ```json
   * {
   *   "threshold": 0.3,
   *   "metric": "cosine",
   *   "partitionIds": ["partition1", "partition2"]
   *   "includeMetadata": true
   * }
   * ```
   * @returns Array of relationships between vectors
   *
   * Example response:
   * ```json
   * {
   *   "relationships": [
   *     {
   *       "vector1": { "id": 123, "partitionId": "partition1" },
   *       "vector2": { "id": 456, "partitionId": "partition1" },
   *       "distance": 0.25
   *     },
   *     {
   *       "vector1": { "id": 789, "partitionId": "partition2" },
   *       "vector2": { "id": 101, "partitionId": "partition2" },
   *       "distance": 0.15
   *     }
   *   ],
   *   "count": 2,
   *   "duration": 345
   * }
   * ```
   */
  router.post(
    '/relationships',
    asyncHandler(async function (req: Request, res: Response) {
      const timer = createTimer();
      timer.start('extract_relationships');

      const { threshold, metric, partitionIds, includeMetadata = true } = req.body;

      // A large threshold makes every pair related, so the guard on the scan
      // budget below is not enough on its own: 10k vectors still means 50M
      // result objects. The threshold is capped as well.
      let safeThreshold: number | undefined;
      try {
        safeThreshold = requireNumber(threshold, 'threshold', Number.MIN_VALUE, MAX_GRAPH_THRESHOLD);
      } catch (error) {
        if (error instanceof InvalidRequestError) {
          res.status(error.status).json({ error: error.message });
          return;
        }
        throw error;
      }
      if (safeThreshold === undefined) {
        res.status(400).json({
          error: 'Invalid request: threshold must be a positive number',
        });
        return;
      }

      // Both graph endpoints compare every pair of vectors (O(n^2)). Reject a
      // request that would exceed the configured scan budget rather than letting
      // a single HTTP call stall the event loop for minutes.
      const totalVectors = (await database.getStats()).database?.vectors.totalInMemory ?? 0;
      if (totalVectors > maxGraphExtractionVectors) {
        res.status(413).json({
          error: `Request would scan ${totalVectors} vectors, above the limit of ${maxGraphExtractionVectors}. ` + `Narrow it with 'partitionIds' or raise the server's maxGraphExtractionVectors.`,
        });
        return;
      }

      try {
        const relationships = await database.extractRelationships(safeThreshold, {
          metric,
          partitionIds,
          includeMetadata,
        });

        const duration = timer.stop('extract_relationships').total;

        res.json({
          relationships,
          count: relationships.length,
          duration,
        });
      } catch (error) {
        const duration = timer.stop('extract_relationships').total;
        console.error('Relationship extraction error:', error);

        res.status(500).json({
          error: publicErrorMessage(error),
          stack: process.env['NODE_ENV'] === 'development' ? (error as Error).stack : undefined,
          duration,
        });
      }
    }),
  );

  /**
   * Vector communities extraction endpoint
   *
   * Finds communities (clusters) of vectors based on a distance threshold across loaded partitions.
   * A community is a group of vectors where each vector is related to at least one other vector in the group.
   *
   * Request body:
   * - `threshold` (number): The maximum distance between vectors to consider them related. **Required**.
   * - `metric` (string): Distance metric to use (e.g., 'cosine', 'euclidean'). Default depends on database implementation.
   * - `partitionIds` (array): Optional array of partition IDs to restrict the community extraction.
   * - `includeMetadata` (boolean): Whether to include metadata for each vector in the results.
   *
   * Example:
   * ```json
   * {
   *   "threshold": 0.3,
   *   "metric": "cosine",
   *   "partitionIds": ["partition1", "partition2"],
   *   "includeMetadata": true
   * }
   * ```
   * @returns Array of communities, where each community is an array of related vectors
   *
   * Example response:
   * ```json
   * {
   *   "communities": [
   *     [
   *       { "id": 123, "partitionId": "partition1", "metadata": { "label": "doc1" } },
   *       { "id": 456, "partitionId": "partition1", "metadata": { "label": "doc2" } }
   *     ],
   *     [
   *       { "id": 789, "partitionId": "partition2", "metadata": { "label": "doc3" } },
   *       { "id": 101, "partitionId": "partition2", "metadata": { "label": "doc4" } },
   *       { "id": 102, "partitionId": "partition2", "metadata": { "label": "doc5" } }
   *     ]
   *   ],
   *   "count": 2,
   *   "totalVectors": 5,
   *   "duration": 345
   * }
   * ```
   */
  router.post(
    '/communities',
    asyncHandler(async function (req: Request, res: Response) {
      const timer = createTimer();
      timer.start('extract_communities');

      const { threshold, metric, partitionIds, includeMetadata = true } = req.body;

      // A large threshold makes every pair related, so the guard on the scan
      // budget below is not enough on its own: 10k vectors still means 50M
      // result objects. The threshold is capped as well.
      let safeThreshold: number | undefined;
      try {
        safeThreshold = requireNumber(threshold, 'threshold', Number.MIN_VALUE, MAX_GRAPH_THRESHOLD);
      } catch (error) {
        if (error instanceof InvalidRequestError) {
          res.status(error.status).json({ error: error.message });
          return;
        }
        throw error;
      }
      if (safeThreshold === undefined) {
        res.status(400).json({
          error: 'Invalid request: threshold must be a positive number',
        });
        return;
      }

      // Both graph endpoints compare every pair of vectors (O(n^2)). Reject a
      // request that would exceed the configured scan budget rather than letting
      // a single HTTP call stall the event loop for minutes.
      const totalVectors = (await database.getStats()).database?.vectors.totalInMemory ?? 0;
      if (totalVectors > maxGraphExtractionVectors) {
        res.status(413).json({
          error: `Request would scan ${totalVectors} vectors, above the limit of ${maxGraphExtractionVectors}. ` + `Narrow it with 'partitionIds' or raise the server's maxGraphExtractionVectors.`,
        });
        return;
      }

      try {
        const communities = await database.extractCommunities(safeThreshold, {
          metric,
          partitionIds,
          includeMetadata,
        });

        const duration = timer.stop('extract_communities').total;

        res.json({
          communities,
          count: communities.length,
          totalVectors: communities.reduce((sum, community) => sum + community.length, 0),
          duration,
        });
      } catch (error) {
        const duration = timer.stop('extract_communities').total;
        console.error('Community extraction error:', error);

        res.status(500).json({
          error: publicErrorMessage(error),
          stack: process.env['NODE_ENV'] === 'development' ? (error as Error).stack : undefined,
          duration,
        });
      }
    }),
  );

  return router;
}
