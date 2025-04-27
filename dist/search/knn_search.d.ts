import { PartitionedVectorDB } from "../vector/partitioned_vector_db";
import { Vector, SearchResult, SearchOptions, KNNOptionsPartitioned, KNNStatsPartitioned } from "../types";
/**
 * KNNEngineSearch using PartitionedVectorDB.
 * Note: Search will only be performed on partitions that are loaded into memory (LRU cache).
 */
/**
 * A class that performs K-nearest neighbors (KNN) search operations using a partitioned vector database.
 *
 * KNNEngineSearch provides an efficient way to find similar vectors in a high-dimensional space
 * by using a partitioned database architecture. It supports different distance metrics, result caching,
 * and maintains performance statistics.
 *
 * Features:
 * - Works with partitioned vector databases for scalable search operations
 * - Caches search results to improve performance for repeated queries
 * - Maintains statistics for performance monitoring
 * - Supports various distance metrics (euclidean by default)
 *
 * @example
 * ```typescript
 * const db = new PartitionedVectorDB(...);
 * const knnSearch = new KNNEngineSearch(db, { metric: 'cosine' });
 *
 * // Perform a search
 * const results = await knnSearch.findNearest(queryVector, 10, { filter: myFilter });
 *
 * // Get performance stats
 * const stats = knnSearch.getStats();
 * ```
 */
export declare class KNNEngineSearch {
    private db;
    private options;
    private distanceFunc;
    private timer;
    private resultCache;
    private stats;
    constructor(db: PartitionedVectorDB, // Accept PartitionedVectorDB
    options?: KNNOptionsPartitioned);
    /**
     * Find k-nearest neighbors for the query vector.
     * Search is performed on partitions currently loaded in PartitionedVectorDB.
     */
    findNearest(query: Vector, k?: number, options?: SearchOptions): Promise<SearchResult[]>;
    /**
     * Create cache key (keeping original logic)
     * @private
     */
    private _getCacheKey;
    /**
     * Get statistics about KNN search (simplified version)
     */
    getStats(): KNNStatsPartitioned;
    /**
     * Clear result cache
     */
    clearCache(): void;
    /**
     * Release resources (mainly cache)
     */
    close(): void;
}
