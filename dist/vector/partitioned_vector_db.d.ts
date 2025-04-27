import { BuildIndexHNSWOptions, DistanceMetric, HNSWStats, PartitionConfig, PartitionedDBEventData, PartitionedDBStats, PartitionedVectorDBInterface, PartitionedVectorDBOptions, SearchOptions, SearchResult, TypedEventEmitter, Vector, VectorData } from '../types';
import { ClusteredVectorDB } from './clustered_vector_db';
declare const PartitionedVectorDB_base: new () => TypedEventEmitter<PartitionedDBEventData>;
/**
 * PartitionedVectorDB distributes vectors across multiple ClusteredVectorDB partitions
 * for improved scalability and performance with very large datasets.
 * It uses an LRU cache to manage loaded partitions in memory and integrates HNSW index persistence.
 *
 * Storage strategy:
 * - Uses ClusteredVectorDB as the partition implementation
 * - Each partition uses optimized binary storage for vectors and JSON for metadata (handled by ClusteredVectorDB)
 * - HNSW indices are stored separately per partition.
 * - Partitions are stored in separate directories with their own config files
 */
/**
 * The `PartitionedVectorDB` class provides a partitioned, in-memory vector database
 * with support for clustering, HNSW indexing, and LRU-based partition management.
 * It is designed to handle large-scale vector data by dividing it into manageable
 * partitions, each with its own configuration and storage.

 * ### Features:
 * - **Partition Management**: Automatically manages partitions with configurable capacity.
 * - **LRU Cache**: Keeps a limited number of partitions in memory for efficient access.
 * - **HNSW Indexing**: Supports approximate nearest neighbor search using HNSW indices.
 * - **Auto-Partitioning**: Automatically creates and activates new partitions when needed.
 * - **Persistence**: Saves and loads partition configurations and data to/from disk.
 * - **Event-Driven**: Emits events for lifecycle operations like initialization, partition loading, and errors.

 * ### Usage:
 * 1. Create an instance of `PartitionedVectorDB` with desired options.
 * 2. Use methods like `addVector`, `bulkAdd`, `findNearest`, and `findNearestHNSW` to interact with the database.
 * 3. Manage partitions using methods like `createPartition`, `setActivePartition`, and `getPartition`.
 * 4. Save and load the database state using `save` and `load`.

 * ### Events:
 * - `db:initialized`: Emitted when the database is fully initialized.
 * - `partition:loaded`: Emitted when a partition is loaded into memory.
 * - `partition:unloaded`: Emitted when a partition is evicted from memory.
 * - `partition:error`: Emitted when an error occurs during partition operations.
 * - `vector:add`: Emitted when a vector is added to a partition.
 * - `vector:delete`: Emitted when a vector is deleted from a partition.
 * - `db:close`: Emitted when the database is closed.

 * ### Example:
 * ```typescript
 * const db = new PartitionedVectorDB({
 *   partitionsDir: './data/partitions',
 *   partitionCapacity: 1000,
 *   maxActivePartitions: 5,
 *   autoCreatePartitions: true,
 *   vectorSize: 128,
 * });
 *
 * await db.initializationPromise; // Wait for initialization
 *
 * // Add a vector
 * const { partitionId, vectorId } = await db.addVector(undefined, [0.1, 0.2, 0.3], { label: 'example' });
 *
 * // Search for nearest neighbors
 * const results = await db.findNearest([0.1, 0.2, 0.3], 5);
 *
 * // Save the database state
 * await db.save();
 *
 * // Close the database
 * await db.close();
 * ```

 * ### Constructor Options:
 * - `partitionsDir`: Directory where partition data is stored.
 * - `partitionCapacity`: Maximum number of vectors per partition.
 * - `maxActivePartitions`: Maximum number of partitions to keep in memory.
 * - `autoCreatePartitions`: Whether to automatically create new partitions when needed.
 * - `vectorSize`: Suggested size of vectors (optional).
 * - `useCompression`: Whether to enable compression for partition data.
 * - `clusterOptions`: Default options for clustered vector databases.
 * - `autoLoadHNSW`: Whether to automatically load HNSW indices.

 * ### Methods:
 * - `addVector`: Adds a single vector to the active partition.
 * - `bulkAdd`: Adds multiple vectors across partitions.
 * - `findNearest`: Finds nearest neighbors using standard search.
 * - `findNearestHNSW`: Finds nearest neighbors using HNSW indices.
 * - `createPartition`: Creates a new partition.
 * - `setActivePartition`: Sets the active partition.
 * - `getPartition`: Loads and retrieves a specific partition.
 * - `getActivePartition`: Retrieves the currently active partition.
 * - `save`: Saves the database state, including partitions and indices.
 * - `load`: Loads the database state from disk.
 * - `close`: Closes the database, saving state and releasing resources.
 * - `buildIndexHNSW`: Builds HNSW indices for specified or all loaded partitions.
 * - `saveHNSWIndices`: Saves HNSW indices for specified or all loaded partitions.
 * - `loadHNSWIndices`: Loads HNSW indices for specified or all loaded partitions.
 * - `getStats`: Retrieves database statistics.
 * - `getVector`: Retrieves a vector by ID.
 * - `getMetadata`: Retrieves metadata for a vector by ID.
 * - `deleteVector`: Deletes a vector by ID.
 * - `updateMetadata`: Updates metadata for a vector by ID.

 * ### Internal Methods:
 * - `_initialize`: Handles asynchronous initialization of the database.
 * - `_loadPartition`: Loads a specific partition into memory.
 * - `_saveHNSWIndex`: Saves the HNSW index for a partition.
 * - `_loadHNSWIndex`: Loads the HNSW index for a partition.
 * - `_ensureActivePartitionHasCapacity`: Ensures the active partition has enough capacity.
 * - `_saveSinglePartitionConfig`: Saves a single partition configuration to disk.
 * - `_loadPartitionConfigs`: Loads all partition configurations from disk.

 * ### Notes:
 * - This class is designed for scenarios where vector data is too large to fit into memory at once.
 * - It relies on partitioning and LRU caching to manage memory usage efficiently.
 * - HNSW indexing provides fast approximate nearest neighbor search but requires additional memory.
 */
export declare class PartitionedVectorDB extends PartitionedVectorDB_base implements PartitionedVectorDBInterface {
    private readonly partitionsDir;
    private readonly partitionCapacity;
    private readonly maxActivePartitions;
    private readonly autoCreatePartitions;
    private readonly vectorSize;
    private readonly useCompression;
    private readonly defaultClusterOptions;
    private readonly autoLoadHNSW;
    private readonly runKMeansOnLoad;
    private partitionConfigs;
    private loadedPartitions;
    private hnswIndices;
    private activePartitionId;
    private isInitialized;
    initializationPromise: Promise<void>;
    private saveConfigPromise;
    private isClosing;
    constructor(options?: PartitionedVectorDBOptions);
    /** Checks if the database is initialized and ready for operations. */
    IsReady(): boolean;
    /**
     * Ensure initialization is complete before performing operations.
     */
    private _ensureInitialized;
    /**
     * Asynchronous initialization: Loads configs and potentially active partitions & indices.
     */
    private _initialize;
    /**
     * Load all partition configuration files from the directory.
     * Finds the active partition or sets one if needed.
     */
    private _loadPartitionConfigs;
    /**
     * Loads a specific partition's DB instance into the LRU cache if not already present.
     * Optionally loads the HNSW index as well.
     * Returns the loaded DB instance or null on failure.
     */
    private _loadPartition;
    /** Loads the HNSW index for a given partition ID if it exists. */
    private _loadHNSWIndex;
    /** Saves the HNSW index for a given partition ID. */
    private _saveHNSWIndex;
    /**
     * Get a partition instance by ID. Loads it (and its index if configured) if necessary.
     */
    getPartition(id: string): Promise<ClusteredVectorDB | null>;
    /**
     * Get the currently active partition instance. Loads it if necessary.
     */
    getActivePartition(): Promise<ClusteredVectorDB | null>;
    /**
     * Explicitly save the entire state: configs, loaded partition data, and loaded HNSW indices.
     */
    save(): Promise<void>;
    /**
     * Loads partition configurations and optionally pre-loads data/indices.
     * This is typically called during initialization but can be called manually.
     */
    load(): Promise<void>;
    /**
     * Build HNSW indices for specified or all loaded partitions
     * Ensures partition is loaded before building.
     */
    buildIndexHNSW(partitionId?: string, options?: BuildIndexHNSWOptions): Promise<void>;
    /**
     * Find nearest neighbors using HNSW indices across specified or all *loaded* partitions.
     * Optimized for parallel search. Loads partitions/indices if needed.
     */
    findNearestHNSW(query: Vector, k?: number, options?: SearchOptions & {
        partitionIds?: string[];
        exactDimensions?: boolean;
    }): Promise<SearchResult[]>;
    /**
     * Explicitly save HNSW indices for specified or all *loaded* partitions.
     */
    saveHNSWIndices(partitionId?: string): Promise<void>;
    /**
     * Explicitly load HNSW indices for specified or all *loaded* partitions.
     * Requires the partition DB to be loaded first.
     */
    loadHNSWIndices(partitionId?: string): Promise<void>;
    /** Get HNSW stats */
    getHNSWStats(partitionId: string): HNSWStats | null;
    /**
     * Close the partitioned database, saving state and releasing resources.
     */
    close(): Promise<void>;
    /** Saves all partition configurations (debounced). */
    savePartitionConfigs(): Promise<void>;
    /** Schedules a config save if one isn't already pending. */
    private scheduleSaveConfigs;
    /** Save a single partition configuration file. */
    private _saveSinglePartitionConfig;
    /** Create a new partition. */
    createPartition(id: string, name: string, options?: {
        description?: string;
        properties?: Record<string, any>;
        setActive?: boolean;
        clusterSize?: number;
        skipInitializationCheck?: boolean;
    }): Promise<string>;
    /** Set the active partition. Handles loading and updating config states. */
    setActivePartition(id: string, force?: boolean): Promise<void>;
    /** Ensures active partition has capacity, creates/activates new one if needed. */
    private _ensureActivePartitionHasCapacity;
    /** Add a single vector */
    addVector(id: number | string | undefined, vector: Vector, metadata?: Record<string, any>): Promise<{
        partitionId: string;
        vectorId: number | string;
    }>;
    /** Bulk add vectors */
    bulkAdd(vectors: VectorData[]): Promise<{
        count: number;
        partitionIds: string[];
    }>;
    /** Get a vector by ID. Searches loaded partitions only. */
    getVector(id: number | string): Promise<{
        partitionId: string;
        vector: Float32Array;
    } | null>;
    /** Get metadata by ID. Searches loaded partitions only. */
    getMetadata(id: number | string): Promise<{
        partitionId: string;
        metadata: Record<string, any>;
    } | null>;
    /** Delete a vector by ID. Searches loaded partitions. */
    deleteVector(id: number | string): Promise<boolean>;
    /** Update metadata for a vector by ID. Searches loaded partitions only. */
    updateMetadata(id: number | string, data: Record<string, any> | ((current: Record<string, any> | null) => Record<string, any>)): Promise<boolean>;
    updateVector(id: number | string, vector: Vector): Promise<boolean>;
    /** Find nearest neighbors (standard search). Searches across specified or all loaded partitions. */
    findNearest(query: Vector, k?: number, options?: SearchOptions): Promise<SearchResult[]>;
    /** Get database statistics. */
    getStats(): Promise<PartitionedDBStats>;
    /**
     * Gets metadata entries that match specified criteria across all loaded partitions.
     *
     * @param criteria Can be:
     *   - A string: field name to check for existence
     *   - An array of strings: multiple field names to check for existence
     *   - An object: key-value pairs where each key must exist and match the specified value
     * @param values Optional value(s) to match against the field(s) when using string/array input
     * @returns Array of {partitionId, vectorId, metadata} objects from all loaded partitions
     *
     * @example
     * ```typescript
     * // Get all metadata entries with 'category' field across partitions
     * const results = await db.getMetadataWithFieldAcrossPartitions('category');
     *
     * // Get entries where 'status' equals 'active' across partitions
     * const active = await db.getMetadataWithFieldAcrossPartitions('status', 'active');
     *
     * // Get entries with both 'author' and 'title' fields across partitions
     * const authoredContent = await db.getMetadataWithFieldAcrossPartitions(['author', 'title']);
     *
     * // Get entries where 'type' is 'book' AND 'published' is true across partitions
     * const publishedBooks = await db.getMetadataWithFieldAcrossPartitions(['type', 'published'], ['book', true]);
     *
     * // Using object syntax (recommended): type='book' AND published=true
     * const publishedBooks = await db.getMetadataWithFieldAcrossPartitions({ type: 'book', published: true });
     * ```
     */
    getMetadataWithFieldAcrossPartitions(criteria: string | string[] | Record<string, any>, values?: any | any[], option?: {
        limit: number;
    }): Promise<Array<{
        partitionId: string;
        vectorId: number | string;
        metadata: Record<string, any>;
    }>>;
    /** Get partition configurations */
    getPartitionConfigs(): PartitionConfig[];
    /**
     * Extract communities of related vectors based on distance threshold across specified partitions.
     * A community is a group of vectors where each vector is related to at least one other vector in the group.
     *
     * @param threshold - The maximum distance between vectors to consider them related
     * @param options - Options including distance metric, partition IDs, and metadata inclusion
     * @returns Array of communities, where each community is an array of related vector information
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
    /**
     * Extract relationships between vectors based on distance threshold across specified partitions.
     *
     * @param threshold - The maximum distance between vectors to consider them related
     * @param options - Options including distance metric, partition filtering, and metadata inclusion
     * @returns Array of relationships with vectorIds, partitionIds, optional metadata, and distances
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
}
export {};
