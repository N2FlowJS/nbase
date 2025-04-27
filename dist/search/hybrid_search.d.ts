import { BaseSearchOptions, BuildIndexHNSWOptions, DistanceMetric, HybridSearchEvents, PartitionedVectorDBInterface, SearchExecutionOptions, SearchResult, // For DB events
TypedEventEmitter, Vector } from '../types';
declare const HybridEngineSearch_base: new () => TypedEventEmitter<HybridSearchEvents>;
/**
 * A high-level search engine that provides hybrid vector search functionality.
 *
 * This class wraps a partitioned vector database and provides:
 * - HNSW index building and management
 * - Vector similarity search with multiple algorithms
 * - Event forwarding from the underlying database
 * - Performance timing and error handling
 *
 * The class extends EventEmitter to provide typed events for search operations
 * and indexing status updates.
 *
 * @example
 * ```
 * const db = new PartitionedVectorDB(...);
 * const searchEngine = new HybridEngineSearch(db, {
 *   defaultK: 5,
 *   defaultDistanceMetric: 'cosine'
 * });
 *
 * // Listen for events
 * searchEngine.on('indexing:progress', (data) => console.log(data.percentage));
 * searchEngine.on('search:complete', (data) => console.log(`Search took ${data.totalTime}ms`));
 *
 * // Build indexes
 * await searchEngine.buildIndexes();
 *
 * // Perform search
 * const results = await searchEngine.findNearest(queryVector, {
 *   k: 10,
 *   useHNSW: true,
 *   includeMetadata: true
 * });
 *
 * // Clean up when done
 * searchEngine.close();
 * ```
 *
 * @fires HybridEngineSearch#indexing:start - When index building starts
 * @fires HybridEngineSearch#indexing:progress - During index building with progress updates
 * @fires HybridEngineSearch#indexing:complete - When index building completes
 * @fires HybridEngineSearch#indexing:error - If an error occurs during indexing
 * @fires HybridEngineSearch#search:complete - When a search operation completes
 * @fires HybridEngineSearch#search:error - If an error occurs during search
 */
export declare class HybridEngineSearch extends HybridEngineSearch_base {
    private db;
    private timer;
    private isBuildingIndex;
    private defaultK;
    private defaultDistanceMetric;
    constructor(db: PartitionedVectorDBInterface, options?: {
        defaultK?: number;
        defaultDistanceMetric?: DistanceMetric;
    });
    private handleIndexProgress;
    private handleIndexComplete;
    private handlePartitionError;
    private _setupEventForwarding;
    close(): void;
    /**
     * Instructs the PartitionedVectorDB to build its HNSW indexes.
     * Adapts the progress callback signature.
     * @param options Options for the build process, accepting a structured progress callback.
     */
    buildIndexes(options?: BuildIndexHNSWOptions): Promise<void>;
    findNearest(query: Vector, options?: BaseSearchOptions & SearchExecutionOptions & {
        useHNSW?: boolean;
    }): Promise<SearchResult[]>;
    getStats(): Promise<Record<string, any>>;
}
export {};
