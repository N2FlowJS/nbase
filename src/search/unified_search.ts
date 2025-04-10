// --- START OF FILE unified_search.ts ---

import { EventEmitter } from 'events';
import { Vector, SearchResult, PartitionedVectorDBInterface, UnifiedSearchOptions, BaseSearchOptions, SearchExecutionOptions, RerankingOptions, UnifiedSearchPartitionedStats, PartitionedDBStats } from '../types';
import { SearchReranker } from './reranking';
import { createTimer } from '../utils/profiling';

/**
 * UnifiedSearch provides a consistent search interface, now leveraging PartitionedVectorDB
 * for scalability with large datasets, using refined type definitions.
 */
/**
 * A unified search interface that provides search capabilities across partitioned vector databases.
 *
 * @class UnifiedSearch
 * @extends {EventEmitter}
 * @description
 * UnifiedSearch wraps a partitioned vector database to provide a unified search API
 * with advanced features like search method selection (HNSW/clustered), reranking,
 * metadata fetching, and performance tracking.
 *
 * The class handles:
 * - Vector similarity search using partitioned vector databases
 * - Automatic method selection between HNSW and clustered search
 * - Optional result reranking for diversity or other criteria
 * - Metadata fetching and inclusion in results
 * - Performance metrics and statistics
 *
 * @fires UnifiedSearch#search:complete - Emitted when a search completes successfully
 * @fires UnifiedSearch#search:error - Emitted when a search encounters an error
 * @fires UnifiedSearch#search:closed - Emitted when the search engine is closed
 *
 * @example
 * ```typescript
 * // Create a UnifiedSearch instance with a partitioned vector database
 * const search = new UnifiedSearch(vectorDb, { debug: true });
 *
 * // Perform a search with unified options
 * const results = await search.search(queryVector, {
 *   k: 20,
 *   rerank: true,
 *   rerankingMethod: 'diversity',
 *   includeMetadata: true
 * });
 * ```
 */
export class UnifiedSearch extends EventEmitter {
  private db: PartitionedVectorDBInterface;
  public reranker: SearchReranker | null = null;
  private debug: boolean = false;
  private searchStats: UnifiedSearchPartitionedStats['search'];
  private timer: ReturnType<typeof createTimer>;

  constructor(
    db: PartitionedVectorDBInterface, // Nhận instance DB đã được cấu hình
    options: { debug?: boolean } = {}
  ) {
    super();
    this.db = db;
    this.debug = options.debug || false;
    this.timer = createTimer();

    // Initialize reranker
    this.reranker = new SearchReranker();

    // Initialize search stats according to the new structure
    this.searchStats = {
      calls: 0,
      totalTime: 0,
      avgTime: 0,
      methodCounts: {
        'partitioned-hnsw': 0,
        'partitioned-clustered': 0,
      },
      lastSearchTime: 0,
      errors: 0,
    };

    // (Optional) Forward relevant events from the PartitionedVectorDB instance
    // Example: if the db emits 'partition:loaded' or 'partition:error', we can forward them
    // if (this.db instanceof EventEmitter) {
    //   this.db.on('partition:loaded', (data: PartitionedDBEventData['partition:loaded']) => this.emit('partition:loaded', data));
    //   this.db.on('partition:error', (data: PartitionedDBEventData['partition:error']) => this.emit('partition:error', data));
    //   // ... forward other necessary events
    // }
  }

  // Helper function in UnifiedSearch
  private async _getVectorsForResults(ids: (number | string)[]): Promise<Map<number | string, Float32Array>> {
    const vectorsMap = new Map<number | string, Float32Array>();
    if (ids.length === 0 || typeof this.db.getVector !== 'function') {
      return vectorsMap;
    }
    // Fetch vectors concurrently
    const promises = ids.map(async (id) => {
      try {
        const result = await this.db.getVector(id); // Assumes getVector returns { partitionId, vector } | null
        if (result?.vector) {
          vectorsMap.set(id, result.vector);
        } else {
          if (this.debug) console.warn(`Vector not found for ID ${id} during rerank fetch.`);
        }
      } catch (error) {
        if (this.debug) console.error(`Failed to get vector for ID ${id}:`, error);
      }
    });
    await Promise.all(promises);
    return vectorsMap;
  }
  /**
   * Search for nearest neighbors using PartitionedVectorDB with unified options.
   */
  async search(
    query: Vector,
    // Sử dụng UnifiedSearchOptions đã được tối ưu
    options: UnifiedSearchOptions = {}
  ): Promise<SearchResult[]> {
    const operationTimer = this.timer; // Use the class-level timer
    operationTimer.start('unified_search_total');

    // Destructure options with defaults, separating base, execution, and unified options
    const {
      // BaseSearchOptions
      k = 10, // TODO: Consider getting default K from DB/config if possible
      filter,
      includeMetadata = false,
      distanceMetric, // Can override DB's default metric for this query

      // SearchExecutionOptions
      partitionIds,
      efSearch, // For HNSW search

      // UnifiedSearchOptions specific
      useHNSW = true, // Default preference for HNSW
      rerank = false,
      rerankingMethod = 'diversity',
      searchTimeoutMs, // Optional timeout for the search operation
    } = options;

    if (this.debug) {
      console.log('UnifiedSearch options received:', options);
    }

    let results: SearchResult[] = [];
    let methodUsed = 'unknown';
    const searchStartTime = Date.now();

    try {
      // --- 1. Database Search ---
      operationTimer.start('db_search');
      const dbSearchOptions: BaseSearchOptions & SearchExecutionOptions = {
        k,
        filter,
        includeMetadata: false, // Don't include metadata yet, fetch later if needed
        distanceMetric,
        partitionIds,
        efSearch,
      };

      // Decide search method (HNSW preferred if enabled and available)
      let searchPromise: Promise<SearchResult[]>;
      const canUseHNSW = useHNSW && typeof this.db.findNearestHNSW === 'function';

      if (canUseHNSW) {
        methodUsed = 'partitioned-hnsw';
        if (this.debug) console.log(`Using ${methodUsed} search with efSearch=${efSearch}...`);
        // Pass only relevant options for HNSW
        const hnswOptions: BaseSearchOptions & SearchExecutionOptions = {
          ...dbSearchOptions,
        };
        searchPromise = this.db.findNearestHNSW(query, k, hnswOptions);
      } else if (typeof this.db.findNearest === 'function') {
        methodUsed = 'partitioned-clustered';
        if (this.debug) console.log(`Using ${methodUsed} search..`);
        // Pass only relevant options for Clustered/findNearest
        const clusteredOptions: BaseSearchOptions & SearchExecutionOptions = {
          ...dbSearchOptions,
        };
        delete clusteredOptions.efSearch; // efSearch is not for clustered
        searchPromise = this.db.findNearest(query, k, clusteredOptions);
      } else {
        throw new Error('No suitable search method (findNearestHNSW or findNearest) available in the database.');
      }

      // TODO: Implement timeout if searchTimeoutMs is provided
      // searchPromise = await Promise.race([
      //   searchPromise,
      //   new Promise((_, reject) => setTimeout(() => reject(new Error('Search timed out')), searchTimeoutMs))
      // ]);

      results = await searchPromise;
      const dbSearchTime = operationTimer.stop('db_search').total ?? 0; // Get duration
      if (this.debug) console.log(`${methodUsed} search completed in ${dbSearchTime}ms, found ${results.length} raw results.`);

      // --- 2. Reranking (Optional) ---
      let rerankTime = 0;
      let finalResults = results; // Results after potential reranking

      // Fetch metadata *before* reranking only if needed for weighted rerank or final output
      const needMetadataForRerankOrOutput = includeMetadata || (rerank && rerankingMethod === 'weighted');
      let metadataMap: Map<number | string, any> | undefined;

      if (rerank && this.reranker && results.length > 1) {
        operationTimer.start('fetch_vectors_for_rerank');
        if (this.debug) console.log(`Fetching ${results.length} vectors for diversity reranking...`);
        // Fetch vectors corresponding to the initial results
        const vectorsMap = await this._getVectorsForResults(results.map((r) => r.id));

        if (this.debug) console.log(`Fetched ${vectorsMap.size} vectors.`);
        operationTimer.stop('fetch_vectors_for_rerank');

        operationTimer.start('rerank');
        if (needMetadataForRerankOrOutput) {
          if (this.debug) console.log('Fetching metadata for reranking/output...');
          metadataMap = await this._getMetadataForResults(results.map((r) => r.id));
          if (this.debug) console.log(`Fetched metadata for ${metadataMap.size} IDs.`);
        }

        const rerankOptions: RerankingOptions = {
          method: rerankingMethod,
          k: k,
          queryVector: query, // Pass the original query vector
          vectorsMap: vectorsMap, // Pass the fetched vectors
          lambda: options.rerankLambda ?? 0.7, // Get lambda from UnifiedSearchOptions or default
          distanceMetric: distanceMetric ?? 'euclidean', // Use the query's distance metric
          // metadataMap: metadataMap, // If weighted rerank also considered
        };
        finalResults = this.reranker.rerank(results, rerankOptions);
        rerankTime = operationTimer.stop('rerank').total ?? 0;
        if (this.debug) console.log(`Reranking completed in ${rerankTime}ms. Results after rerank: ${finalResults.length}`);
      } else {
        // If not reranking, ensure results are capped at k
        finalResults = results.slice(0, k);
      }

      // --- 3. Add Metadata (if requested and not already fetched) ---
      if (includeMetadata) {
        const firstResultNeedsMeta = finalResults.length > 0 && !finalResults[0]?.metadata;

        if (firstResultNeedsMeta) {
          operationTimer.start('fetch_metadata');
          if (this.debug) console.log('Fetching metadata for final output...');
          // Fetch metadata only if it wasn't already fetched for reranking
          const finalMetadataMap = metadataMap ?? (await this._getMetadataForResults(finalResults.map((r) => r.id)));
          if (this.debug) console.log(`Fetched metadata for ${finalMetadataMap.size} IDs.`);

          for (const result of finalResults) {
            const meta = finalMetadataMap.get(result.id);
            if (meta) {
              result.metadata = meta;
            }
          }
          operationTimer.stop('fetch_metadata');
        } else if (finalResults.length > 0 && finalResults[0]?.metadata) {
          if (this.debug) console.log('Metadata already present in results (likely from reranking fetch).');
        }
      }

      // --- 4. Finalize Stats and Emit Event ---
      const totalSearchTime = operationTimer.stop('unified_search_total').total ?? 0;

      this.searchStats.calls++;
      this.searchStats.methodCounts[methodUsed] = (this.searchStats.methodCounts[methodUsed] || 0) + 1;
      this.searchStats.totalTime += totalSearchTime;
      this.searchStats.avgTime = this.searchStats.totalTime / this.searchStats.calls;
      this.searchStats.lastSearchTime = totalSearchTime;
      this.searchStats.lastSearchTimestamp = new Date();

      this.emit('search:complete', {
        method: methodUsed,
        searchOnlyTime: dbSearchTime,
        rerankTime,
        totalTime: totalSearchTime,
        resultCount: finalResults.length,
        kRequested: k,
        optionsUsed: options, // Include original options for context
      });

      if (this.debug) {
        console.log(`UnifiedSearch completed in ${totalSearchTime}ms (DB: ${dbSearchTime}ms, Rerank: ${rerankTime}ms). Method: ${methodUsed}. Returning ${finalResults.length} results.`);
      }

      return finalResults;
    } catch (error: unknown) {
      const totalSearchTimeOnError = operationTimer.stop('unified_search_total').total ?? Date.now() - searchStartTime; // Ensure timer stops
      const errorMessage = error instanceof Error ? error.message : String(error);
      console.error(`UnifiedSearch error after ${totalSearchTimeOnError}ms using method ${methodUsed}:`, errorMessage, error);

      this.searchStats.errors++;
      this.searchStats.lastError = error instanceof Error ? error : new Error(errorMessage);

      this.emit('search:error', {
        error: this.searchStats.lastError,
        method: methodUsed,
        options,
        totalTime: totalSearchTimeOnError,
      });

      // Re-throw the error so the caller can handle it
      throw error;
    }
  }

  /**
   * Helper to fetch metadata for a list of result IDs.
   * Assumes `this.db` has a `getMetadata(id)` method adhering to the interface.
   * @private
   */
  private async _getMetadataForResults(ids: (number | string)[]): Promise<Map<number | string, any>> {
    const metadataMap = new Map<number | string, any>();
    if (ids.length === 0 || typeof this.db.getMetadata !== 'function') {
      return metadataMap;
    }

    operationTimer.start('fetch_metadata_batch'); // Start timer for batch metadata fetch
    if (this.debug) console.log(`Fetching metadata for ${ids.length} IDs...`);

    // Fetch metadata concurrently
    const promises = ids.map(async (id) => {
      try {
        // Assumes getMetadata returns { partitionId: string; metadata: Record<string, any> } | null
        const result = await this.db.getMetadata(id);
        if (result?.metadata !== undefined) {
          // Check if metadata exists in the result
          metadataMap.set(id, result.metadata);
        } else {
          if (this.debug) console.warn(`Metadata not found for ID ${id}.`);
        }
      } catch (error) {
        if (this.debug) console.error(`Failed to get metadata for ID ${id}:`, error);
        // Optionally log the error but continue fetching others
      }
    });

    await Promise.all(promises);
    operationTimer.stop('fetch_metadata_batch'); // Stop timer
    if (this.debug) console.log(`Metadata fetch batch completed in ${operationTimer.getElapsed('fetch_metadata_batch')}ms`);
    return metadataMap;
  }

  /**
   * Get search engine statistics, including stats from PartitionedVectorDB.
   * @returns Object containing search statistics according to UnifiedSearchPartitionedStats
   */
  async getStats(): Promise<UnifiedSearchPartitionedStats> {
    let dbStats: PartitionedDBStats = {} as PartitionedDBStats;
    try {
      if (typeof this.db.getStats === 'function') {
        dbStats = await this.db.getStats();
      } else {
        console.warn('Database instance does not provide a getStats() method.');
      }
    } catch (error) {
      console.error('Failed to get stats from database:', error);
    }

    // Construct the stats object based on the defined interface
    const stats: UnifiedSearchPartitionedStats = {
      search: { ...this.searchStats }, // Copy current search stats
      database: dbStats, // Embed the stats received from the DB
      reranker: {
        available: this.reranker !== null,
      },
      // Add other sections if UnifiedSearchPartitionedStats defines them
    };
    return stats;
  }

  /**
   * Close and clean up resources, including closing the PartitionedVectorDB.
   */
  async close(): Promise<void> {
    if (this.debug) console.log('Closing UnifiedSearch...');

    // Close the underlying database instance
    if (typeof this.db.close === 'function') {
      await this.db.close();
    } else {
      console.warn('Database instance does not provide a close() method.');
    }

    this.emit('search:closed');
    if (this.debug) console.log('UnifiedSearch closed.');
  }
}

// Add a global timer instance for helper functions like _getMetadataForResults
// This is a simple approach; a more robust solution might inject the timer or use a separate instance.
const operationTimer = createTimer();

// --- END OF FILE unified_search.ts ---
