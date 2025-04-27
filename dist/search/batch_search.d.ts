import { BatchSearchConfiguration, BatchSearchQuery, PartitionedVectorDBInterface, SearchResult } from '../types';
/**
 * BatchEngineSearch processes multiple vector queries in parallel using a PartitionedVectorDBInterface.
 * This version uses the optimized type definitions.
 */
/**
 * Manages efficient processing of multiple vector search queries in batches.
 *
 * The BatchEngineSearch class provides functionality to handle concurrent vector
 * search operations against a partitioned vector database. It automatically manages
 * large batches by splitting them into smaller chunks, processes queries in parallel,
 * and handles timeout and error scenarios gracefully.
 *
 * Key features:
 * - Efficient batching of multiple vector search queries
 * - Automatic chunking of large query batches
 * - Concurrent query execution with configurable timeouts
 * - Support for preserving original query order in results
 * - Integration with different search methods (HNSW or clustered)
 * - Error handling and graceful degradation
 *
 * @example
 * ```typescript
 * const searchEngine = new PartitionedVectorDB(...);
 * const batchSearch = new BatchEngineSearch(searchEngine, {
 *   maxBatchSize: 32,
 *   defaultSearchTimeoutMs: 10000
 * });
 *
 * const queries = [
 *   { query: vectorA, k: 5, options: { useHNSW: true } },
 *   { query: vectorB, k: 10, options: { filter: { category: "tech" } } }
 * ];
 *
 * const results = await batchSearch.searchBatch(queries);
 * ```
 */
export declare class BatchEngineSearch {
    private searchEngine;
    private options;
    constructor(searchEngine: PartitionedVectorDBInterface, options?: BatchSearchConfiguration);
    /**
     * Process multiple search queries in a batch.
     * @param queries - Array of search queries using the new BatchSearchQuery interface.
     * @returns Results for each query.
     */
    searchBatch(queries: BatchSearchQuery[]): Promise<SearchResult[][]>;
    /**
     * Process batch queries using the PartitionedVectorDB directly.
     * Leverages Promise.all for concurrency across queries and relies on the
     * PartitionedVectorDB's internal parallelism.
     */
    private _processBatch;
    /**
     * Placeholder for grouping similar queries.
     * @private
     */
    private _groupSimilarQueries;
    /**
     * Clean up resources (no-op in this version).
     */
    shutdown(): void;
}
