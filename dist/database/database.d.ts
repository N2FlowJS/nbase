import { UnifiedSearch } from '../search/unified_search';
import { BuildIndexHNSWOptions, DatabaseEvents, DatabaseOptions, DatabaseStats, DistanceMetric, PartitionedVectorDBInterface, SearchResult, TypedEventEmitter, UnifiedSearchOptions, Vector, VectorData } from '../types';
declare const Database_base: new () => TypedEventEmitter<DatabaseEvents>;
/**
 * High-level Database class using PartitionedVectorDB for scalable vector storage and search.
 * Provides unified search, caching, auto-save, monitoring, and a simplified API.
 *
 * NOTE: File system backup of the partitions directory is recommended as an external process.
 */
/**
 * The `Database` class provides a high-level interface for managing a partitioned vector database
 * with support for vector addition, deletion, metadata management, nearest neighbor search,
 * and background tasks such as auto-saving and monitoring. It integrates with `PartitionedVectorDB`
 * and `UnifiedSearch` for efficient vector storage and search operations.
 *
 * ### Features:
 * - Asynchronous initialization with event-based readiness notifications.
 * - Partitioned vector storage with configurable options for clustering, indexing, and persistence.
 * - Unified search engine for nearest neighbor queries with caching and concurrency control.
 * - Background tasks for auto-saving and monitoring database performance.
 * - Event-driven architecture for tracking database operations and errors.
 *
 * ### Usage:
 * - Instantiate the class with `DatabaseOptions`.
 * - Await the `ready()` promise or listen for the `'ready'` event before performing operations.
 * - Use public methods like `addVector`, `findNearest`, `deleteVector`, etc., to interact with the database.
 * - Call `close()` to gracefully shut down the database and release resources.
 *
 * ### Events:
 * - `initializing`: Emitted when the database starts initializing.
 * - `ready`: Emitted when the database is fully initialized and ready for operations.
 * - `error`: Emitted when an error occurs during operations or background tasks.
 * - `close`: Emitted when the database is closed.
 * - Various search and indexing-related events such as `search:start`, `search:complete`, `index:progress`, etc.
 *
 * ### Example:
 * ```typescript
 * const db = new Database({
 *   vectorSize: 128,
 *   cacheSize: 1000,
 *   partitioning: { partitionsDir: './data/partitions' },
 *   indexing: { buildOnStart: true },
 * });
 *
 * db.on('ready', async () => {
 *   console.log('Database is ready!');
 *   await db.addVector('id1', [0.1, 0.2, 0.3], { label: 'example' });
 *   const results = await db.findNearest([0.1, 0.2, 0.3], 5);
 *   console.log('Search results:', results);
 * });
 *
 * db.on('error', (err) => {
 *   console.error('Database error:', err);
 * });
 * ```
 *
 * ### Notes:
 * - Ensure proper error handling for asynchronous operations.
 * - Use the `getStats()` method to retrieve detailed information about the database state and performance.
 * - The database must be closed using `close()` to ensure all resources are released and state is saved.
 */
export declare class Database extends Database_base {
    private vectorDB;
    private unifiedSearch;
    private readonly searchCache;
    private readonly monitor;
    private readonly timer;
    private readonly options;
    private isClosed;
    private isReady;
    initializationPromise: Promise<void>;
    private autoSaveTimer;
    private readonly activeSearchPromises;
    private metrics;
    /**
     * Creates a new Database instance. Initialization is asynchronous.
     * Listen for the 'ready' event or await the ready() promise before use.
     * @param options Configuration options.
     */
    constructor(options: DatabaseOptions);
    /**
     * Performs the main asynchronous initialization sequence.
     */
    private _initialize;
    /** Ensures a directory exists. */
    private _ensureDirectoryExists;
    /** Handles loading and/or building indices during startup based on options. */
    private _handleInitialIndexing;
    /** Sets up internal event listeners for PartitionedVectorDB events. */
    private _setupEventListeners;
    /** Sets up listeners for UnifiedSearch events. */
    private _setupUnifiedSearchListeners;
    /** Pushes current DB stats to the monitor asynchronously. */
    private _updateMonitorDbMetrics;
    /** Sets up the auto-save interval timer. */
    private _setupAutoSave;
    /** Generates a cache key for search results. */
    private _getCacheKey;
    /** Simple, fast vector hash (not cryptographically secure). */
    private _hashVector;
    /** Checks if the database is initialized and ready for operations. */
    IsReady(): boolean;
    /** Throws an error if the database is not ready or closed. */
    private _assertReady;
    /**
     * Adds a vector to the appropriate partition.
     * @returns An object containing the partitionId and the vectorId.
     */
    addVector(id: number | string | undefined, vector: Vector, metadata?: Record<string, any>): Promise<{
        partitionId: string;
        vectorId: number | string;
    }>;
    /**
     * Bulk adds vectors, handling partitioning automatically.
     */
    bulkAdd(vectors: VectorData[]): Promise<{
        count: number;
        partitionIds: string[];
    }>;
    /** Deletes a vector from the partition it resides in. */
    deleteVector(id: number | string): Promise<boolean>;
    /** Checks if a vector exists in any loaded partition. */
    hasVector(id: number | string): Promise<boolean>;
    /** Retrieves a vector by searching loaded partitions. */
    getVector(id: number | string): Promise<{
        partitionId: string;
        vector: Float32Array;
    } | null>;
    /** Adds or updates metadata for a vector. Requires finding the vector first. */
    addMetadata(id: number | string, metadata: Record<string, any>): Promise<boolean>;
    /** Updates metadata using a callback or merging. Requires finding the vector first. */
    updateMetadata(id: number | string, metadataUpdate: Record<string, any> | ((existing: Record<string, any> | null) => Record<string, any>)): Promise<boolean>;
    /** Retrieves metadata by searching loaded partitions. */
    getMetadata(id: number | string): Promise<{
        partitionId: string;
        metadata: Record<string, any>;
    } | null>;
    /**
     * Searches for metadata entries that match specific criteria across all loaded partitions.
     *
     * @param criteria Can be:
     *   - A string: field name to check for existence
     *   - An array of strings: multiple field names to check for existence
     *   - An object: key-value pairs where each key must exist and match the specified value
     * @param values Optional value(s) to match against the field(s) when using string/array input
     * @returns Array of objects with partitionId, vectorId, and metadata
     *
     * @example
     * ```typescript
     * // Get all entries with 'category' field
     * const withCategory = await db.getMetadataWithField('category');
     *
     * // Get entries where type is 'article'
     * const articles = await db.getMetadataWithField('type', 'article');
     *
     * // Get entries with both 'author' and 'publishDate' fields
     * const authorsWithDates = await db.getMetadataWithField(['author', 'publishDate']);
     *
     * // Get entries where type='book' AND published=true (using arrays)
     * const publishedBooks = await db.getMetadataWithField(['type', 'published'], ['book', true]);
     *
     * // Get entries where type='book' AND published=true (using object)
     * const publishedBooks = await db.getMetadataWithField({ type: 'book', published: true });
     * ```
     */
    getMetadataWithField(criteria: string | string[] | Record<string, any>, values?: any | any[], option?: {
        limit: number;
    }): Promise<Array<{
        partitionId: string;
        vectorId: number | string;
        metadata: Record<string, any>;
    }>>;
    /**
     * Performs a nearest neighbor search using the UnifiedSearch engine.
     * Handles caching and concurrency limits.
     */
    findNearest(query: Vector, k?: number, options?: UnifiedSearchOptions): Promise<SearchResult[]>;
    /** Alias for findNearest */
    search(query: Vector, options?: UnifiedSearchOptions): Promise<SearchResult[]>;
    /**
     * Saves the current state of the database (delegated to PartitionedVectorDB).
     * This includes partition configurations, loaded partition data, and loaded HNSW indices.
     */
    save(): Promise<void>;
    /**
     * Builds HNSW indexes. Delegates to PartitionedVectorDB.
     */
    buildIndexes(partitionId?: string, options?: BuildIndexHNSWOptions): Promise<void>;
    /** Closes the database, saves state, stops background tasks, and releases resources. */
    close(): Promise<void>;
    /** Gets combined statistics about the database state, components, and system. */
    getStats(): Promise<DatabaseStats>;
    /** Gets the underlying PartitionedVectorDB instance. Use with caution. */
    getVectorDB(): PartitionedVectorDBInterface;
    /** Gets the UnifiedSearch instance. */
    getUnifiedSearch(): UnifiedSearch;
    /** Gets the total count of vectors across all configured partitions. */
    getTotalVectorCount(): Promise<number>;
    /** Gets the count of vectors currently loaded in memory. */
    getInMemoryVectorCount(): Promise<number>;
    /** Gets the number of partitions currently loaded in memory. */
    getLoadedPartitionCount(): Promise<number>;
    /** Gets the IDs of the partitions currently loaded in memory. */
    getLoadedPartitionIds(): Promise<string[]>;
    /**
     * Extract relationships between vectors based on distance threshold across all loaded partitions.
     *
     * @param threshold - The maximum distance between vectors to consider them related
     * @param options - Options including distance metric, partition filtering, and metadata inclusion
     * @returns An array of relationships with vectorIds, partitionIds, optional metadata, and distances
     */
    extractRelationships(threshold: number, options?: {
        metric?: DistanceMetric;
        partitionIds?: string[];
        includeMetadata?: boolean;
    }): Promise<Array<{
        vector1: {
            id: number | string;
            partitionId: string;
            metadata?: Record<string, any>;
        };
        vector2: {
            id: number | string;
            partitionId: string;
            metadata?: Record<string, any>;
        };
        distance: number;
    }>>;
    /**
     * Extract communities of related vectors based on distance threshold across all loaded partitions.
     * A community is a group of vectors where each vector is related to at least one other vector in the group.
     *
     * @param threshold - The maximum distance between vectors to consider them related
     * @param options - Options including distance metric and partition filtering
     * @returns An array of communities, where each community is an array of related vector information
     */
    extractCommunities(threshold: number, options?: {
        metric?: DistanceMetric;
        partitionIds?: string[];
        includeMetadata?: boolean;
    }): Promise<Array<Array<{
        id: number | string;
        partitionId: string;
        metadata?: Record<string, any>;
    }>>>;
}
export {};
