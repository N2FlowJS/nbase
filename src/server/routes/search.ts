import { Router, Request, Response } from 'express';
import { ApiContext, UnifiedSearchOptions } from '../../types';

/**
 * Creates and configures Express router for vector search functionality.
 *
 * This function sets up a POST endpoint that performs vector similarity search
 * against the database with configurable parameters for search behavior and performance.
 *
 * @param context - The API context object containing dependencies
 * @param context.database - Database interface for vector search operations
 * @param context.timer - Utility for measuring operation duration
 * @param context.createFilterFunction - Factory function to create filters for search results
 *
 * @returns An Express router configured with vector search endpoint
 *
 * @remarks
 * The search endpoint accepts the following parameters in the request body:
 * - query: Required vector array for similarity search
 * - k: Number of results to return
 * - method: Search method ("hnsw" or default clustered search)
 * - partitionIds: Specific partitions to search
 * - probes: Number of probes for HNSW search
 * - efSearch: Exploration factor for HNSW search
 * - distanceMetric: Distance metric to use
 * - rerank: Whether to rerank results
 * - rerankingMethod: Method to use for reranking
 * - rerankLambda: Lambda parameter for reranking
 * - filters: Metadata filters to apply to search results
 * - includeMetadata: Whether to include metadata in results
 * - includeVectors: Whether to include vectors in results
 * - skipCache: Whether to bypass the cache
 * - searchTimeoutMs: Search timeout in milliseconds
 *
 * The endpoint returns search results with metadata including duration and search options used.
 */
export function searchRoutes(context: ApiContext) {
  const router = Router();
  const { database, timer, createFilterFunction } = context;

  /**
   * Search endpoint
   */
  router.post(
    '/',

    async (req: Request, res: Response) => {
      timer.start('search');

      const { query, k, method, partitionIds, probes, efSearch, distanceMetric, rerank, rerankingMethod, rerankLambda, filters = {}, includeMetadata, includeVectors, skipCache, searchTimeoutMs } = req.body;

      if (!query || !Array.isArray(query)) {
        res.status(400).json({
          error: 'Invalid request: query vector array is required',
        });
        return;
      }

      try {
        const hasFilters = Array.isArray(filters) ? filters.length > 0 : Object.keys(filters).length > 0;

        // Build comprehensive search options
        const searchOptions: UnifiedSearchOptions = {
          k,
          useHNSW: method === 'hnsw',
          // Only include filter if filters are present
          filter: hasFilters ? createFilterFunction(filters) : undefined,
          includeMetadata,
          includeVectors,
          skipCache,
          partitionIds,
          probes,
          efSearch,
          distanceMetric,
          rerank,
          rerankingMethod,
          rerankLambda,
          searchTimeoutMs,
        };

        // Remove undefined options to avoid overriding defaults
        Object.keys(searchOptions).forEach((key) => {
          if (searchOptions[key as keyof UnifiedSearchOptions] === undefined) {
            delete searchOptions[key as keyof UnifiedSearchOptions];
          }
        });

        const results = await database.findNearest(query, k, searchOptions);

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
          error: (error as Error).message,
          stack: process.env.NODE_ENV === 'development' ? (error as Error).stack : undefined,
          duration,
        });
        return;
      }
    }
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
  router.post('/metadata', async (req: Request, res: Response) => {
    timer.start('metadata_search');

    const { criteria, values, includeVectors = false, limit } = req.body;

    // Validate that criteria is provided and is of the right type
    if (criteria === undefined || (typeof criteria !== 'string' && !Array.isArray(criteria) && (typeof criteria !== 'object' || criteria === null))) {
      res.status(400).json({
        error: 'Invalid request: criteria must be a string, array of strings, or object with field-value pairs',
      });
      return;
    }

    try {
      // Call the database method to get metadata with matching fields
      const results = await database.getMetadataWithField(criteria, values, { limit });

      // If includeVectors is true, fetch vectors for each result
      if (includeVectors) {
        const resultsWithVectors = await Promise.all(
          results.map(async (item: { partitionId: string; vectorId: number | string; metadata: Record<string, any> }) => {
            const vectorData = await database.getVector(item.vectorId);
            return {
              ...item,
              vector: vectorData ? vectorData.vector : null,
            };
          })
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
        error: (error as Error).message,
        stack: process.env.NODE_ENV === 'development' ? (error as Error).stack : undefined,
        duration,
      });
    }
  });

  return router;
}
