/// <reference types="node" />
import { EventEmitter } from 'events';
import { Vector, SearchResult, PartitionedVectorDBInterface, UnifiedSearchOptions, UnifiedSearchPartitionedStats } from '../types';
import { SearchReranker } from './reranking';
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
export declare class UnifiedSearch extends EventEmitter {
    private db;
    reranker: SearchReranker | null;
    private debug;
    private searchStats;
    private timer;
    constructor(db: PartitionedVectorDBInterface, // Nhận instance DB đã được cấu hình
    options?: {
        debug?: boolean;
    });
    private _getVectorsForResults;
    /**
     * Search for nearest neighbors using PartitionedVectorDB with unified options.
     */
    search(query: Vector, options?: UnifiedSearchOptions): Promise<SearchResult[]>;
    /**
     * Helper to fetch metadata for a list of result IDs.
     * Assumes `this.db` has a `getMetadata(id)` method adhering to the interface.
     * @private
     */
    private _getMetadataForResults;
    /**
     * Get search engine statistics, including stats from PartitionedVectorDB.
     * @returns Object containing search statistics according to UnifiedSearchPartitionedStats
     */
    getStats(): Promise<UnifiedSearchPartitionedStats>;
    /**
     * Close and clean up resources, including closing the PartitionedVectorDB.
     */
    close(): Promise<void>;
}
