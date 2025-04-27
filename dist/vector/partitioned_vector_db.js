"use strict";
// --- START OF FILE partitioned_vector_db.ts ---
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.PartitionedVectorDB = void 0;
// partitioned_db_optimized.ts
const events_1 = require("events");
const fs_1 = require("fs");
const lru_cache_1 = require("lru-cache"); // Using a robust LRU cache library
const path_1 = __importDefault(require("path"));
const hnsw_1 = __importDefault(require("../ann/hnsw")); // Assuming HNSW is a class for the clustering algorithm
const log_1 = require("../utils/log");
const config_1 = __importDefault(require("../config"));
const clustered_vector_db_1 = require("./clustered_vector_db");
// --- Types ---
const DEFAULT_PARTITION_CAPACITY = 100000;
const DEFAULT_MAX_ACTIVE_PARTITIONS = 3; // Keep a few partitions warm
const HNSW_INDEX_DIR_NAME = 'hnsw';
const HNSW_INDEX_FILE_NAME = 'hnsw_index.json'; // Using binary for HNSW potentially
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
class PartitionedVectorDB extends events_1.EventEmitter {
    constructor(options = {}) {
        super();
        this.isInitialized = false;
        this.saveConfigPromise = null;
        this.isClosing = false; // Flag to prevent operations during close
        (0, log_1.log)('info', '[PartitionedVectorDB] Initializing with options:', JSON.stringify(options, null, 2));
        this.partitionsDir = options.partitionsDir || path_1.default.join(process.cwd(), 'database', 'partitions');
        this.partitionCapacity = options.partitionCapacity || DEFAULT_PARTITION_CAPACITY;
        this.maxActivePartitions = options.maxActivePartitions || DEFAULT_MAX_ACTIVE_PARTITIONS;
        this.autoCreatePartitions = options.autoCreatePartitions !== false; // Default true
        this.vectorSize = options.vectorSize ?? config_1.default.defaults.vectorSize;
        this.useCompression = options.useCompression ?? false; // Default false
        this.defaultClusterOptions = options.clusterOptions ?? {};
        this.autoLoadHNSW = options.autoLoadHNSW ?? true; // Default true
        this.runKMeansOnLoad = options.runKMeansOnLoad ?? config_1.default.indexing.runKMeansOnLoad; // Default false
        (0, log_1.log)('info', `[PartitionedVectorDB] Configuration:
      - partitionsDir: ${this.partitionsDir}
      - partitionCapacity: ${this.partitionCapacity}
      - maxActivePartitions: ${this.maxActivePartitions}
      - autoCreatePartitions: ${this.autoCreatePartitions}
      - vectorSize: ${this.vectorSize ?? 'not specified'}
      - useCompression: ${this.useCompression}
      - autoLoadHNSW: ${this.autoLoadHNSW}
      - runKMeansOnLoad: ${this.runKMeansOnLoad}`);
        this.partitionConfigs = new Map();
        this.hnswIndices = new Map();
        this.activePartitionId = null;
        // --- Initialize LRU Cache ---
        this.loadedPartitions = new lru_cache_1.LRUCache({
            max: this.maxActivePartitions,
            // Dispose function called when an item is removed (evicted)
            dispose: async (dbInstance, partitionId, reason) => {
                (0, log_1.log)('info', `[PartitionedVectorDB] Disposing partition ${partitionId} from memory (Reason: ${reason}).`);
                // Save is handled by the main save() method or explicitly before eviction if needed.
                // Close the DB instance to release resources.
                const hnswIndex = this.hnswIndices.get(partitionId);
                if (hnswIndex) {
                    // Decide if HNSW index should be saved on eviction - maybe not, rely on explicit save?
                    // await this._saveHNSWIndex(partitionId); // Optional: save index on eviction
                    this.hnswIndices.delete(partitionId); // Remove from memory map
                    (0, log_1.log)('info', `[PartitionedVectorDB] Unloaded HNSW index for evicted partition ${partitionId}`);
                }
                try {
                    // Close partition DB (releases file handles, etc., but VectorDB.close might save if path set - review VectorDB.close)
                    // Ideally, saving is orchestrated explicitly via PartitionedVectorDB.save()
                    await dbInstance.close();
                    this.emit('partition:unloaded', { id: partitionId });
                }
                catch (error) {
                    (0, log_1.log)('error', `[PartitionedVectorDB] Error closing partition ${partitionId} during dispose:`, error);
                    this.emit('partition:error', {
                        id: partitionId,
                        error,
                        operation: 'dispose',
                    });
                }
            },
        });
        // Ensure partitions directory exists
        try {
            if (!(0, fs_1.existsSync)(this.partitionsDir)) {
                (0, fs_1.mkdirSync)(this.partitionsDir, { recursive: true });
            }
        }
        catch (err) {
            // Fatal if we cannot ensure the base directory exists
            throw new Error(`FATAL: Could not create or access partitions directory: ${this.partitionsDir} - ${err.message}`);
        }
        // Defer actual loading to an async method
        this.initializationPromise = this._initialize(options.autoLoadPartitions !== false);
    }
    /** Checks if the database is initialized and ready for operations. */
    IsReady() {
        return this.isInitialized && !this.isClosing;
    }
    /**
     * Ensure initialization is complete before performing operations.
     */
    async _ensureInitialized(force = false) {
        if (this.isClosing)
            throw new Error('Database is closing or closed.');
        if (!this.isInitialized && !force) {
            await this.initializationPromise;
        }
    }
    /**
     * Asynchronous initialization: Loads configs and potentially active partitions & indices.
     */
    async _initialize(autoLoad) {
        if (this.isInitialized)
            return;
        (0, log_1.log)('info', `[PartitionedVectorDB] Starting initialization (autoLoad: ${autoLoad})`);
        try {
            // 1. Load all partition configurations first
            await this._loadPartitionConfigs();
            (0, log_1.log)('info', `[PartitionedVectorDB] Loaded ${this.partitionConfigs.size} partition configurations.`);
            // 2. Determine which partitions to load initially (e.g., active one)
            const partitionsToLoad = [];
            if (autoLoad && this.activePartitionId) {
                partitionsToLoad.push(this.activePartitionId);
                // Optionally load more based on LRU or other criteria if needed
            }
            // 3. Load partitions and potentially their HNSW indices in parallel
            if (partitionsToLoad.length > 0) {
                (0, log_1.log)('info', `[PartitionedVectorDB] Auto-loading initial partitions: [${partitionsToLoad.join(', ')}]`);
                await Promise.all(partitionsToLoad.map((id) => this._loadPartition(id, this.autoLoadHNSW)));
                (0, log_1.log)('info', `[PartitionedVectorDB] Initial partitions loaded (${this.loadedPartitions.size} in memory, ${this.hnswIndices.size} HNSW indices loaded).`);
            }
            else {
                (0, log_1.log)('info', '[PartitionedVectorDB] No initial partitions specified for auto-loading.');
            }
            this.isInitialized = true;
            (0, log_1.log)('info', `[PartitionedVectorDB] Initialization complete. Active: ${this.activePartitionId ?? 'None'}`);
            this.emit('db:initialized', {
                partitionCount: this.partitionConfigs.size,
                loadedCount: this.loadedPartitions.size,
                activeId: this.activePartitionId,
            });
        }
        catch (err) {
            (0, log_1.log)('error', `[PartitionedVectorDB] FATAL: Error during initialization:`, err);
            this.emit('partition:error', { error: err, operation: 'initialize' });
            // Potentially set a flag indicating failed initialization?
            throw err; // Re-throw to signal failure
        }
    }
    /**
     * Load all partition configuration files from the directory.
     * Finds the active partition or sets one if needed.
     */
    async _loadPartitionConfigs() {
        (0, log_1.log)('info', `[PartitionedVectorDB] Loading partition configurations from ${this.partitionsDir}`);
        this.partitionConfigs.clear();
        let foundActiveId = null;
        const configsRead = [];
        try {
            const entries = await fs_1.promises.readdir(this.partitionsDir, {
                withFileTypes: true,
            });
            const partitionDirs = entries.filter((e) => e.isDirectory());
            (0, log_1.log)('info', `[PartitionedVectorDB] Found ${partitionDirs.length} potential partition directories.`);
            for (const dir of partitionDirs) {
                const configPath = path_1.default.join(this.partitionsDir, dir.name, `${dir.name}.config.json`);
                if ((0, fs_1.existsSync)(configPath)) {
                    (0, log_1.log)('info', `[PartitionedVectorDB] Attempting to load config: ${configPath}`);
                    try {
                        const content = await fs_1.promises.readFile(configPath, 'utf8');
                        const config = JSON.parse(content);
                        // Basic validation
                        if (config.id && config.dbDirName === dir.name) {
                            this.partitionConfigs.set(config.id, config);
                            configsRead.push(config);
                            (0, log_1.log)('info', `[PartitionedVectorDB] Loaded config for partition: ${config.id} (Dir: ${dir.name}, Active: ${config.active}, Vectors: ${config.vectorCount})`);
                            if (config.active) {
                                if (foundActiveId && foundActiveId !== config.id) {
                                    (0, log_1.log)('warn', `[PartitionedVectorDB] Multiple active partitions defined! Found ${config.id} after ${foundActiveId}. Deactivating ${config.id}.`);
                                    config.active = false;
                                    // Schedule a save to fix the inconsistency?
                                    this.scheduleSaveConfigs();
                                }
                                else {
                                    foundActiveId = config.id;
                                }
                            }
                        }
                        else {
                            (0, log_1.log)('warn', `[PartitionedVectorDB] Invalid partition config format or mismatched ID/DirName: ${configPath}`);
                        }
                    }
                    catch (e) {
                        (0, log_1.log)('warn', `[PartitionedVectorDB] Error reading/parsing partition config ${configPath}:`, e);
                    }
                }
                else {
                    (0, log_1.log)('info', `[PartitionedVectorDB] No config file found in directory: ${dir.name}`);
                }
            }
            this.activePartitionId = foundActiveId;
            (0, log_1.log)('info', `[PartitionedVectorDB] Active partition ID after scan: ${this.activePartitionId ?? 'None'}`);
            // If no active partition found, try to set one or create the first one
            if (!this.activePartitionId && this.partitionConfigs.size > 0) {
                // Find the first config (order might not be guaranteed, consider sorting by name/ID if needed)
                const firstConfig = this.partitionConfigs.values().next().value;
                if (firstConfig) {
                    (0, log_1.log)('info', `[PartitionedVectorDB] No active partition found, activating first available: ${firstConfig.id}`);
                    firstConfig.active = true;
                    this.activePartitionId = firstConfig.id;
                    this.scheduleSaveConfigs(); // Save the change
                }
            }
            else if (!this.activePartitionId && this.autoCreatePartitions) {
                (0, log_1.log)('info', '[PartitionedVectorDB] No partitions found, creating initial partition.');
                // Call createPartition but skip initialization check within it
                await this.createPartition(`p-${Date.now()}`, 'Initial Partition', {
                    setActive: true,
                    skipInitializationCheck: true,
                });
                // Re-fetch active ID potentially set by createPartition
                this.activePartitionId = Array.from(this.partitionConfigs.values()).find((c) => c.active)?.id ?? null;
            }
            this.emit('partitions:loaded', {
                count: this.partitionConfigs.size,
                active: this.activePartitionId,
            });
        }
        catch (error) {
            if (error.code === 'ENOENT' && !(0, fs_1.existsSync)(this.partitionsDir)) {
                (0, log_1.log)('warn', `[PartitionedVectorDB] Partitions directory ${this.partitionsDir} not found. It will be created when needed.`);
                // If autoCreate is on, the first partition creation will handle it.
            }
            else {
                (0, log_1.log)('error', '[PartitionedVectorDB] Error listing or reading partition configs:', error);
                throw error; // Propagate other errors
            }
        }
    }
    /**
     * Loads a specific partition's DB instance into the LRU cache if not already present.
     * Optionally loads the HNSW index as well.
     * Returns the loaded DB instance or null on failure.
     */
    async _loadPartition(partitionId, loadHNSW = this.autoLoadHNSW // Use instance default
    ) {
        if (this.isClosing)
            return null; // Prevent loading during close
        const cachedDb = this.loadedPartitions.get(partitionId);
        if (cachedDb) {
            // If DB is already loaded, ensure HNSW is loaded if requested and not already loaded
            if (loadHNSW && !this.hnswIndices.has(partitionId)) {
                await this._loadHNSWIndex(partitionId, cachedDb); // Pass the DB instance
            }
            return cachedDb;
        }
        const config = this.partitionConfigs.get(partitionId);
        if (!config) {
            (0, log_1.log)('warn', `[PartitionedVectorDB] Partition config not found for ID: ${partitionId}. Cannot load.`);
            return null;
        }
        // Construct paths relative to the main partitions directory
        const partitionDirPath = path_1.default.join(this.partitionsDir, config.dbDirName);
        const dbBasePath = path_1.default.join(partitionDirPath, 'data');
        (0, log_1.log)('info', `[PartitionedVectorDB] Loading partition ${partitionId} DB from base path: ${dbBasePath}`);
        try {
            // Ensure the specific partition directory exists
            if (!(0, fs_1.existsSync)(partitionDirPath)) {
                await fs_1.promises.mkdir(partitionDirPath, { recursive: true });
                (0, log_1.log)('info', `[PartitionedVectorDB] Created directory for partition ${partitionId}: ${partitionDirPath}`);
            }
            // Also ensure the data directory exists for a new partition
            const dataDir = path_1.default.dirname(dbBasePath);
            if (!(0, fs_1.existsSync)(dataDir)) {
                await fs_1.promises.mkdir(dataDir, { recursive: true });
                (0, log_1.log)('info', `[PartitionedVectorDB] Created data directory for partition ${partitionId}: ${dataDir}`);
            }
            const metaFilePath = path_1.default.join(dbBasePath, 'meta.json');
            const vectorFilePath = path_1.default.join(dbBasePath, 'vec.bin');
            const clusterFilePath = path_1.default.join(dbBasePath, 'cluster.json');
            if (!(0, fs_1.existsSync)(metaFilePath)) {
                (0, log_1.log)('info', '`[PartitionedVectorDB] Meta file not found, creating new one.`');
                await fs_1.promises.writeFile(metaFilePath, JSON.stringify({}), 'utf8');
            }
            if (!(0, fs_1.existsSync)(vectorFilePath)) {
                (0, log_1.log)('info', '`[PartitionedVectorDB] Vector file not found, creating new one.`');
                await fs_1.promises.writeFile(vectorFilePath, Buffer.alloc(0));
            }
            if (!(0, fs_1.existsSync)(clusterFilePath)) {
                (0, log_1.log)('info', '`[PartitionedVectorDB] Vector file not found, creating new one.`');
                await fs_1.promises.writeFile(clusterFilePath, JSON.stringify({}), 'utf8');
            }
            const hnswIndexDir = path_1.default.join(partitionDirPath, HNSW_INDEX_DIR_NAME);
            const hnswIndexPath = path_1.default.join(hnswIndexDir, HNSW_INDEX_FILE_NAME);
            if (!(0, fs_1.existsSync)(hnswIndexDir)) {
                (0, log_1.log)('info', `[PartitionedVectorDB] HNSW index directory not found, creating new one.`);
                await fs_1.promises.mkdir(hnswIndexDir, { recursive: true });
            }
            if (!(0, fs_1.existsSync)(hnswIndexPath)) {
                (0, log_1.log)('info', `[PartitionedVectorDB] HNSW index file not found, creating new one.`);
                await fs_1.promises.writeFile(hnswIndexPath, JSON.stringify(config_1.default.indexing.hnsw), 'utf8');
            }
            // --- Load the ClusteredVectorDB ---
            const clusterDbOptions = {
                ...this.defaultClusterOptions,
                clusterSize: config.clusterSize, // Use specific or default
                useCompression: this.useCompression, // Pass down compression setting
                runKMeansOnLoad: this.runKMeansOnLoad, // Pass down K-Means option
            };
            const vectorDB = new clustered_vector_db_1.ClusteredVectorDB(this.vectorSize, // Pass the suggested vector size
            dbBasePath, // Pass the base path for data files
            clusterDbOptions);
            await vectorDB.load(); // Wait for initialization
            // Successfully loaded the DB, add to LRU cache
            this.loadedPartitions.set(partitionId, vectorDB);
            (0, log_1.log)('info', `[PartitionedVectorDB] Partition DB ${partitionId} loaded. Vector count: ${vectorDB.getVectorCount()}`);
            // --- Optionally Load HNSW Index ---
            (0, log_1.log)('info', `[PartitionedVectorDB] Loading HNSW index for partition ${partitionId}`);
            if (loadHNSW) {
                await this._loadHNSWIndex(partitionId, vectorDB);
            }
            (0, log_1.log)('info', `[PartitionedVectorDB] HNSW index loaded for partition ${partitionId}`);
            this.emit('partition:loaded', {
                id: partitionId,
                name: config.name,
                vectorCount: vectorDB.getVectorCount(),
                hnswLoaded: this.hnswIndices.has(partitionId),
            });
            // --- Sync vector count ---
            (0, log_1.log)('info', `[PartitionedVectorDB] Syncing vector count for partition ${partitionId}`);
            const loadedCount = vectorDB.getVectorCount();
            (0, log_1.log)('info', `[PartitionedVectorDB] Loaded vector count: ${loadedCount}`);
            if (config.vectorCount !== loadedCount) {
                (0, log_1.log)('warn', `[PartitionedVectorDB] Partition ${partitionId}: Config count (${config.vectorCount}) differs from loaded DB count (${loadedCount}). Updating config.`);
                config.vectorCount = loadedCount;
                this.scheduleSaveConfigs(); // Save updated count later
            }
            return vectorDB;
        }
        catch (error) {
            (0, log_1.log)('error', `[PartitionedVectorDB] Error loading partition DB ${partitionId} from ${dbBasePath}:`, error);
            // Clean up potentially partially loaded state? Remove from cache if added?
            this.loadedPartitions.delete(partitionId);
            this.hnswIndices.delete(partitionId); // Ensure HNSW is also removed if DB load failed
            this.emit('partition:error', {
                id: partitionId,
                error,
                operation: 'loadPartitionDB',
            });
            return null;
        }
    }
    /** Loads the HNSW index for a given partition ID if it exists. */
    async _loadHNSWIndex(partitionId, dbInstance) {
        (0, log_1.log)('info', `[PartitionedVectorDB] Loading HNSW index for partition ${partitionId}`);
        if (this.hnswIndices.has(partitionId)) {
            (0, log_1.log)('info', `[PartitionedVectorDB] HNSW index for ${partitionId} already loaded.`);
            return true; // Already loaded
        }
        if (this.isClosing)
            return false;
        const config = this.partitionConfigs.get(partitionId);
        if (!config) {
            (0, log_1.log)('warn', `[PartitionedVectorDB] Cannot load HNSW index: Config not found for ${partitionId}`);
            return false;
        }
        const indexDir = path_1.default.join(this.partitionsDir, config.dbDirName, HNSW_INDEX_DIR_NAME);
        const indexPath = path_1.default.join(indexDir, HNSW_INDEX_FILE_NAME);
        if ((0, fs_1.existsSync)(indexPath)) {
            (0, log_1.log)('info', `[PartitionedVectorDB] Loading HNSW index for partition ${partitionId} from ${indexPath}`);
            try {
                const hnswIndex = await hnsw_1.default.loadIndex(indexPath, dbInstance);
                this.hnswIndices.set(partitionId, hnswIndex);
                (0, log_1.log)('info', `[PartitionedVectorDB] Successfully loaded HNSW index for ${partitionId}. Nodes: ${hnswIndex.getNodeCount()}`);
                this.emit('partition:indexLoaded', {
                    id: partitionId,
                    indexType: 'hnsw',
                    path: indexPath,
                });
                return true;
            }
            catch (error) {
                (0, log_1.log)('error', `[PartitionedVectorDB] Error loading HNSW index for partition ${partitionId} from ${indexPath}:`, error.message || error);
                this.emit('partition:error', {
                    id: partitionId,
                    error,
                    operation: 'loadHNSWIndex',
                });
                return false;
            }
        }
        else {
            (0, log_1.log)('info', `[PartitionedVectorDB] HNSW index file not found for partition ${partitionId} at ${indexPath}. Index not loaded.`);
            return false; // Index file doesn't exist
        }
    }
    /** Saves the HNSW index for a given partition ID. */
    async _saveHNSWIndex(partitionId) {
        (0, log_1.log)('info', `[PartitionedVectorDB] Saving HNSW index for partition ${partitionId}`);
        const hnswIndex = this.hnswIndices.get(partitionId);
        const config = this.partitionConfigs.get(partitionId);
        if (!hnswIndex) {
            (0, log_1.log)('info', `[PartitionedVectorDB] No HNSW index instance found in memory for partition ${partitionId}. Skipping save.`);
            return false;
        }
        if (!config) {
            (0, log_1.log)('warn', `[PartitionedVectorDB] Cannot save HNSW index: Config not found for ${partitionId}`);
            return false;
        }
        if (this.isClosing) {
            (0, log_1.log)('warn', `[PartitionedVectorDB] Skipping HNSW index save for ${partitionId} during close operation (already handled or closing).`);
            return false;
        }
        const indexDir = path_1.default.join(this.partitionsDir, config.dbDirName, HNSW_INDEX_DIR_NAME);
        const indexPath = path_1.default.join(indexDir, HNSW_INDEX_FILE_NAME);
        (0, log_1.log)('info', `[PartitionedVectorDB] Saving HNSW index for partition ${partitionId} to ${indexPath}`);
        try {
            // Ensure directory exists
            if (!(0, fs_1.existsSync)(indexDir)) {
                await fs_1.promises.mkdir(indexDir, { recursive: true });
            }
            await hnswIndex.saveIndex(indexPath); // HNSW handles the actual saving
            (0, log_1.log)('info', `[PartitionedVectorDB] Successfully saved HNSW index for ${partitionId}.`);
            this.emit('partition:indexSaved', {
                id: partitionId,
                indexType: 'hnsw',
                path: indexPath,
            });
            return true;
        }
        catch (error) {
            (0, log_1.log)('error', `[PartitionedVectorDB] Error saving HNSW index for partition ${partitionId} to ${indexPath}:`, error);
            this.emit('partition:error', {
                id: partitionId,
                error,
                operation: 'saveHNSWIndex',
                path: indexPath,
            });
            return false;
        }
    }
    /**
     * Get a partition instance by ID. Loads it (and its index if configured) if necessary.
     */
    async getPartition(id) {
        (0, log_1.log)('info', `[PartitionedVectorDB] Getting partition ${id}...`);
        await this._ensureInitialized();
        // _loadPartition handles cache checking, loading DB, and potentially HNSW index
        return this._loadPartition(id); // Uses instance default for loading HNSW
    }
    /**
     * Get the currently active partition instance. Loads it if necessary.
     */
    async getActivePartition() {
        (0, log_1.log)('info', `[PartitionedVectorDB] Getting active partition...`);
        await this._ensureInitialized();
        if (!this.activePartitionId) {
            (0, log_1.log)('warn', '[PartitionedVectorDB] No active partition is set.');
            return null;
        }
        return this._loadPartition(this.activePartitionId); // Loads DB and potentially HNSW
    }
    // =====================================================================
    // Public API Methods (Add, Search, Delete, Stats, etc.)
    // =====================================================================
    /**
     * Explicitly save the entire state: configs, loaded partition data, and loaded HNSW indices.
     */
    async save() {
        await this._ensureInitialized();
        if (this.isClosing) {
            (0, log_1.log)('warn', '[PartitionedVectorDB] Attempted to save while closing.');
            return;
        }
        (0, log_1.log)('info', '[PartitionedVectorDB] Starting comprehensive save...');
        // 1. Save all configurations (ensures counts, active status, etc., are up-to-date)
        // Use await on the debounced save to ensure it finishes before proceeding
        await this.savePartitionConfigs();
        (0, log_1.log)('info', `[PartitionedVectorDB] Partition configurations saved. Active partition: ${this.activePartitionId}`);
        // Ensure the save promise is resolved before proceeding
        if (this.saveConfigPromise)
            await this.saveConfigPromise; // Ensure pending config save finishes
        // 2. Save data for all *loaded* partitions in parallel
        const loadedPartitionIds = Array.from(this.loadedPartitions.keys());
        (0, log_1.log)('info', `[PartitionedVectorDB] Saving data for ${loadedPartitionIds.length} loaded partitions...`);
        const partitionSavePromises = loadedPartitionIds.map(async (id) => {
            const partition = this.loadedPartitions.peek(id); // Use peek to avoid altering LRU order
            if (partition) {
                try {
                    // Check if the underlying DB instance exists and has a save method
                    if (typeof partition.save === 'function') {
                        await partition.save(); // Call the save method of ClusteredVectorDB/VectorDB
                        (0, log_1.log)('info', `[PartitionedVectorDB] Saved data for partition ${id}`);
                        return true;
                    }
                    else {
                        (0, log_1.log)('warn', `[PartitionedVectorDB] Partition ${id} instance cannot be saved (missing save method or wrong type).`);
                        return false;
                    }
                }
                catch (error) {
                    (0, log_1.log)('error', `[PartitionedVectorDB] Error saving data for partition ${id}:`, error);
                    this.emit('partition:error', {
                        id,
                        error,
                        operation: 'savePartitionData',
                    });
                    return false; // Indicate failure for this partition
                }
            }
            return true; // Partition not found in cache (shouldn't happen with keys()), consider it success?
        });
        // 3. Save all *loaded* HNSW indices in parallel
        const loadedHnswIds = Array.from(this.hnswIndices.keys());
        (0, log_1.log)('info', `[PartitionedVectorDB] Saving ${loadedHnswIds.length} loaded HNSW indices...`);
        const hnswSavePromises = loadedHnswIds.map((id) => this._saveHNSWIndex(id));
        // Wait for all saves to complete
        const [partitionResults, hnswResults] = await Promise.all([Promise.all(partitionSavePromises), Promise.all(hnswSavePromises)]);
        const successfulPartitions = partitionResults.filter((r) => r).length;
        const successfulHnsw = hnswResults.filter((r) => r).length;
        (0, log_1.log)('info', `[PartitionedVectorDB] Comprehensive save complete. Partitions saved: ${successfulPartitions}/${loadedPartitionIds.length}. HNSW indices saved: ${successfulHnsw}/${loadedHnswIds.length}.`);
        this.emit('db:saved', {
            partitionsSaved: successfulPartitions,
            indicesSaved: successfulHnsw,
        });
    }
    /**
     * Loads partition configurations and optionally pre-loads data/indices.
     * This is typically called during initialization but can be called manually.
     */
    async load() {
        if (this.isInitialized && !this.isClosing) {
            (0, log_1.log)('warn', '[PartitionedVectorDB] Database already initialized. Call close() before loading again.');
            return;
        }
        this.isClosing = false; // Reset closing flag if re-loading
        this.isInitialized = false; // Reset initialization flag
        // Reset internal state before loading
        this.loadedPartitions.clear();
        this.hnswIndices.clear();
        this.partitionConfigs.clear();
        this.activePartitionId = null;
        (0, log_1.log)('info', '[PartitionedVectorDB] Starting manual load process...');
        // Re-run the initialization logic, including loading configs and initial partitions/indices
        this.initializationPromise = this._initialize(this.autoLoadHNSW); // Use constructor options
        await this.initializationPromise;
        (0, log_1.log)('info', '[PartitionedVectorDB] Manual load process finished.');
        this.emit('db:loaded', {
            partitionCount: this.partitionConfigs.size,
            loadedCount: this.loadedPartitions.size,
            activeId: this.activePartitionId,
        });
    }
    /**
     * Build HNSW indices for specified or all loaded partitions
     * Ensures partition is loaded before building.
     */
    async buildIndexHNSW(partitionId, options) {
        await this._ensureInitialized(options?.force);
        const buildSingleIndex = async (id) => {
            (0, log_1.log)('info', `[PartitionedVectorDB] Building HNSW index for partition ${id}...`);
            const partition = await this.getPartition(id); // Ensures partition DB is loaded
            if (!partition) {
                (0, log_1.log)('error', `[PartitionedVectorDB] Cannot build HNSW index: Partition ${id} not found or could not be loaded.`);
                return;
            }
            let hnswIndex = this.hnswIndices.get(id);
            if (!hnswIndex) {
                (0, log_1.log)('info', `[PartitionedVectorDB] Creating new HNSW index instance for partition ${id} before building.`);
                hnswIndex = new hnsw_1.default(partition); // Pass the loaded partition DB
                this.hnswIndices.set(id, hnswIndex);
            }
            (0, log_1.log)('info', `[PartitionedVectorDB] Building HNSW index for partition ${id}...`);
            try {
                await hnswIndex.buildIndex({
                    ...options,
                    // Wrap progress callback to emit event
                    progressCallback: (progress) => {
                        options?.progressCallback?.(progress); // Call original callback if provided
                        this.emit('partition:indexProgress', {
                            id,
                            progress,
                            operation: 'buildHNSW',
                        });
                    },
                });
                (0, log_1.log)('info', `[PartitionedVectorDB] HNSW index built successfully for partition ${id}.`);
                this.emit('partition:indexed', { id, indexType: 'hnsw' });
            }
            catch (error) {
                (0, log_1.log)('error', `[PartitionedVectorDB] Error building HNSW index for partition ${id}:`, error);
                this.emit('partition:error', {
                    id,
                    error,
                    operation: 'buildHNSWIndex',
                });
            }
        };
        if (partitionId) {
            await buildSingleIndex(partitionId);
        }
        else {
            // Build for all currently *loaded* partitions in parallel
            const partitionIds = Array.from(this.loadedPartitions.keys());
            (0, log_1.log)('info', `[PartitionedVectorDB] Building HNSW indices for ${partitionIds.length} loaded partitions in parallel...`);
            await Promise.all(partitionIds.map((id) => buildSingleIndex(id)));
            (0, log_1.log)('info', `[PartitionedVectorDB] Finished building HNSW indices for loaded partitions.`);
        }
    }
    /**
     * Find nearest neighbors using HNSW indices across specified or all *loaded* partitions.
     * Optimized for parallel search. Loads partitions/indices if needed.
     */
    async findNearestHNSW(query, k = 10, options = {}) {
        await this._ensureInitialized();
        const queryVector = query instanceof Float32Array ? query : new Float32Array(query);
        // Determine target partitions: provided list OR all configured partitions (load on demand)
        // Decide whether to search *all* configured or just *currently loaded*
        // Let's search specified OR all *loaded* by default for performance.
        // If you need to search *all* partitions (loading unloaded ones), adjust the logic.
        const targetPartitionIds = options.partitionIds
            ? options.partitionIds.filter((id) => this.partitionConfigs.has(id)) // Filter valid provided IDs
            : Array.from(this.loadedPartitions.keys()); // Default to currently loaded
        if (targetPartitionIds.length === 0) {
            (0, log_1.log)('warn', '[PartitionedVectorDB] No valid partitions specified or loaded to search with HNSW.');
            return [];
        }
        (0, log_1.log)('info', `[PartitionedVectorDB] Performing HNSW search on partitions: [${targetPartitionIds.join(', ')}]`);
        // Perform search in parallel
        const searchResultsNested = await Promise.all(targetPartitionIds.map(async (partitionId) => {
            try {
                // 1. Ensure Partition DB is loaded
                const partition = await this._loadPartition(partitionId, false); // Load DB only first
                if (!partition) {
                    (0, log_1.log)('warn', `[PartitionedVectorDB] Skipping HNSW search on partition ${partitionId}: Could not load DB.`);
                    return [];
                }
                // 2. Ensure HNSW Index is loaded (or try loading it)
                let hnswIndex = this.hnswIndices.get(partitionId);
                if (!hnswIndex) {
                    const loaded = await this._loadHNSWIndex(partitionId, partition);
                    if (loaded) {
                        hnswIndex = this.hnswIndices.get(partitionId);
                    }
                    else {
                        // Optional: Build index on the fly if not found? Risky for performance.
                        // log('info', `[PartitionedVectorDB] HNSW index for ${partitionId} not found. Building on-the-fly for search.`);
                        // hnswIndex = new HNSW(partition);
                        // await hnswIndex.buildIndex(); // Consider build options
                        // this.hnswIndices.set(partitionId, hnswIndex);
                        (0, log_1.log)('warn', `[PartitionedVectorDB] Skipping HNSW search on partition ${partitionId}: Index not loaded and not found.`);
                        return []; // Skip if index cannot be loaded/created
                    }
                }
                // 3. Perform the search on the loaded index
                if (hnswIndex) {
                    return await hnswIndex.findNearest(queryVector, k, {
                        ...options,
                        filter: options.filter, // Pass down filter
                    });
                }
                else {
                    return []; // Should not happen if logic above is correct
                }
            }
            catch (error) {
                (0, log_1.log)('error', `[PartitionedVectorDB] Error during HNSW search for partition ${partitionId}:`, error);
                this.emit('partition:error', {
                    id: partitionId,
                    error,
                    operation: 'searchHNSW',
                });
                return []; // Return empty results for this partition on error
            }
        }));
        // Flatten results, sort by distance, and take top k
        const mergedResults = searchResultsNested.flat();
        mergedResults.sort((a, b) => a.dist - b.dist);
        return mergedResults.slice(0, k);
    }
    /**
     * Explicitly save HNSW indices for specified or all *loaded* partitions.
     */
    async saveHNSWIndices(partitionId) {
        await this._ensureInitialized();
        const idsToSave = partitionId ? [partitionId] : Array.from(this.hnswIndices.keys()); // Save only loaded indices
        if (idsToSave.length === 0) {
            (0, log_1.log)('info', '[PartitionedVectorDB] No HNSW indices loaded or specified to save.');
            return;
        }
        (0, log_1.log)('info', `[PartitionedVectorDB] Saving HNSW indices for partitions: [${idsToSave.join(', ')}]`);
        await Promise.all(idsToSave.map((id) => this._saveHNSWIndex(id)));
        (0, log_1.log)('info', '[PartitionedVectorDB] Finished saving HNSW indices.');
    }
    /**
     * Explicitly load HNSW indices for specified or all *loaded* partitions.
     * Requires the partition DB to be loaded first.
     */
    async loadHNSWIndices(partitionId) {
        await this._ensureInitialized();
        const loadIndexForPartition = async (id) => {
            const partition = this.loadedPartitions.peek(id); // Check if DB is loaded without changing LRU order
            if (!partition) {
                (0, log_1.log)('warn', `[PartitionedVectorDB] Cannot load HNSW index for ${id}: Partition DB not loaded.`);
                // Optionally load the DB first: await this._loadPartition(id, false);
                return;
            }
            if (this.hnswIndices.has(id)) {
                (0, log_1.log)('info', `[PartitionedVectorDB] HNSW index for ${id} is already loaded.`);
                return;
            }
            await this._loadHNSWIndex(id, partition); // Attempt to load
        };
        const idsToLoad = partitionId ? [partitionId] : Array.from(this.loadedPartitions.keys()); // Try loading for all loaded partitions
        if (idsToLoad.length === 0) {
            (0, log_1.log)('info', '[PartitionedVectorDB] No partitions loaded or specified to load HNSW indices for.');
            return;
        }
        (0, log_1.log)('info', `[PartitionedVectorDB] Loading HNSW indices for partitions: [${idsToLoad.join(', ')}]`);
        await Promise.all(idsToLoad.map((id) => loadIndexForPartition(id)));
        (0, log_1.log)('info', `[PartitionedVectorDB] Finished loading HNSW indices. Indices in memory: ${this.hnswIndices.size}`);
    }
    /** Get HNSW stats */
    getHNSWStats(partitionId) {
        if (!this.isInitialized)
            return null;
        const hnswIndex = this.hnswIndices.get(partitionId);
        return hnswIndex ? hnswIndex.getStats() : null;
    }
    /**
     * Close the partitioned database, saving state and releasing resources.
     */
    async close() {
        if (this.isInitialized) {
            (0, log_1.log)('warn', '[PartitionedVectorDB] Close operation called before initialization.');
            return;
        }
        if (this.isClosing) {
            (0, log_1.log)('warn', '[PartitionedVectorDB] Close operation already in progress.');
            return;
        }
        (0, log_1.log)('info', '[PartitionedVectorDB] Closing database...');
        this.isClosing = true;
        // 1. Ensure initialization finished (to avoid race conditions)
        // We might be closing before initialization fully completed
        try {
            await this.initializationPromise;
        }
        catch (initError) {
            (0, log_1.log)('warn', '[PartitionedVectorDB] Initialization failed, proceeding with close anyway:', initError);
        }
        // 2. Perform final save of everything loaded
        try {
            await this.save(); // Comprehensive save of configs, partitions, indices
        }
        catch (saveError) {
            (0, log_1.log)('error', '[PartitionedVectorDB] Error during final save operation:', saveError);
            // Continue closing even if save fails
        }
        // 3. Clear the LRU cache - this triggers dispose which calls close() on individual DBs
        // Dispose should NOT save again, just release resources.
        this.loadedPartitions.clear();
        // 4. Clear HNSW index map (dispose might have already removed some)
        this.hnswIndices.clear();
        // 5. Clear partition configs
        this.partitionConfigs.clear();
        // 6. Reset state
        this.activePartitionId = null;
        this.isInitialized = false; // Mark as not initialized
        // Keep isClosing = true
        this.emit('db:close', undefined);
        (0, log_1.log)('info', '[PartitionedVectorDB] Database closed.');
    }
    // --- Configuration Saving ---
    /** Saves all partition configurations (debounced). */
    async savePartitionConfigs() {
        if (this.isClosing)
            return; // Don't save during close triggered by 'save' itself
        if (!this.saveConfigPromise) {
            this.saveConfigPromise = (async () => {
                // await new Promise((resolve) => setTimeout(resolve, 500)); // Simple debounce delay
                (0, log_1.log)('info', '[PartitionedVectorDB] Debounced saving of partition configurations...');
                const configsToSave = Array.from(this.partitionConfigs.values());
                try {
                    const savePromises = configsToSave.map((config) => this._saveSinglePartitionConfig(config));
                    await Promise.all(savePromises);
                    (0, log_1.log)('info', `[PartitionedVectorDB] Saved ${configsToSave.length} partition configurations.`);
                    this.emit('config:saved', undefined);
                }
                catch (error) {
                    (0, log_1.log)('error', '[PartitionedVectorDB] Error saving one or more partition configs:', error);
                    // Emit specific error?
                }
                finally {
                    this.saveConfigPromise = null; // Release lock
                }
            })();
        }
        return this.saveConfigPromise;
    }
    /** Schedules a config save if one isn't already pending. */
    scheduleSaveConfigs() {
        if (!this.saveConfigPromise && !this.isClosing) {
            this.savePartitionConfigs();
        }
    }
    /** Save a single partition configuration file. */
    async _saveSinglePartitionConfig(config) {
        if (this.isClosing)
            return; // Prevent saving during close
        const partitionDir = path_1.default.join(this.partitionsDir, config.dbDirName);
        const configPath = path_1.default.join(partitionDir, `${config.id}.config.json`); // Store config inside partition dir
        try {
            // Ensure directory exists before writing config
            if (!(0, fs_1.existsSync)(partitionDir)) {
                await fs_1.promises.mkdir(partitionDir, { recursive: true });
            }
            await fs_1.promises.writeFile(configPath, JSON.stringify(config, null, 2), 'utf8');
        }
        catch (error) {
            (0, log_1.log)('error', `[PartitionedVectorDB] Error saving config ${config.id} to ${configPath}:`, error);
            this.emit('partition:error', {
                id: config.id,
                error,
                operation: 'saveConfig',
                path: configPath,
            });
            throw error; // Re-throw
        }
    }
    /** Create a new partition. */
    async createPartition(id, name, options = {}) {
        // Allow skipping check only for internal calls during initial setup
        if (!options.skipInitializationCheck) {
            await this._ensureInitialized();
        }
        if (!/^[a-zA-Z0-9._-]+$/.test(id)) {
            throw new Error('Partition ID must contain only alphanumeric, underscore, hyphen, or dot characters');
        }
        if (this.partitionConfigs.has(id)) {
            throw new Error(`Partition with ID ${id} already exists`);
        }
        const dirName = id; // Use the ID as the directory name for simplicity and uniqueness
        const partitionDataDir = path_1.default.join(this.partitionsDir, dirName);
        const partitionDataDirData = path_1.default.join(partitionDataDir, 'data');
        (0, log_1.log)('info', `[PartitionedVectorDB] Creating new partition '${name}' (ID: ${id}) in directory: ${partitionDataDir}`);
        try {
            if (!(0, fs_1.existsSync)(partitionDataDir)) {
                await fs_1.promises.mkdir(partitionDataDir, { recursive: true });
            }
            if (!(0, fs_1.existsSync)(partitionDataDirData)) {
                await fs_1.promises.mkdir(partitionDataDirData, { recursive: true });
            }
        }
        catch (error) {
            (0, log_1.log)('error', `[PartitionedVectorDB] Failed to create directory for new partition ${id}: ${partitionDataDir}`, error);
            this.emit('partition:error', {
                id,
                error,
                operation: 'createDir',
                path: partitionDataDir,
            });
            throw new Error(`Failed to create directory for partition ${id}: ${error.message}`);
        }
        const newConfig = {
            id,
            name,
            dbDirName: dirName,
            active: false, // Activation handled later
            vectorCount: 0,
            description: options.description,
            properties: options.properties,
            clusterSize: options.clusterSize, // Use specific or let underlying DB use default
        };
        // Add to in-memory map *before* saving and loading
        this.partitionConfigs.set(id, newConfig);
        // Save the new config immediately (important!)
        try {
            await this._saveSinglePartitionConfig(newConfig);
        }
        catch (saveError) {
            // If saving config fails, rollback the creation?
            this.partitionConfigs.delete(id); // Remove from memory
            (0, log_1.log)('error', `[PartitionedVectorDB] Failed to save config for new partition ${id}. Rolling back creation.`);
            // Optionally try to delete the created directory?
            throw saveError;
        }
        // Ensure all required files are created
        try {
            const clusterDbOptions = {
                clusterSize: options.clusterSize,
                useCompression: this.useCompression,
                runKMeansOnLoad: this.runKMeansOnLoad, // Pass down K-Means option
            };
            const vectorDB = new clustered_vector_db_1.ClusteredVectorDB(this.vectorSize, path_1.default.join(partitionDataDir, 'data'), // Base path for data files
            clusterDbOptions);
            // Save the initial state of the database
            await vectorDB.save();
            // Create an empty HNSW index file
            const hnswIndexDir = path_1.default.join(partitionDataDir, HNSW_INDEX_DIR_NAME);
            const hnswIndexPath = path_1.default.join(hnswIndexDir, HNSW_INDEX_FILE_NAME);
            if (!(0, fs_1.existsSync)(hnswIndexDir)) {
                await fs_1.promises.mkdir(hnswIndexDir, { recursive: true });
            }
            await fs_1.promises.writeFile(hnswIndexPath, JSON.stringify(config_1.default.indexing.hnsw), 'utf8');
        }
        catch (error) {
            (0, log_1.log)('error', `[PartitionedVectorDB] Failed to initialize files for new partition ${id}:`, error);
            throw error;
        }
        // Load the new partition into memory (will trigger LRU if needed)
        // Don't load HNSW index yet, it doesn't exist
        (0, log_1.log)('info', `[PartitionedVectorDB] Loading new partition ${id}...`);
        const loadedDB = await this._loadPartition(id, false);
        (0, log_1.log)('info', `[PartitionedVectorDB] Loaded partition ${id}`);
        if (!loadedDB) {
            // If loading fails immediately after creation, this is problematic
            this.partitionConfigs.delete(id); // Remove from memory
            // Config file might still exist, manual cleanup needed?
            (0, log_1.log)('error', `[PartitionedVectorDB] Failed to load partition ${id} immediately after creation. Config saved but DB unusable.`);
            throw new Error(`Failed to load newly created partition ${id}.`);
        }
        // Handle activation
        if (options.setActive === true || (!this.activePartitionId && this.partitionConfigs.size === 1)) {
            (0, log_1.log)('info', `[PartitionedVectorDB] Activating new partition ${id}...`);
            // Activate if requested OR if it's the very first partition
            await this.setActivePartition(id, true); // This saves config changes
        }
        (0, log_1.log)('info', `[PartitionedVectorDB] Partition ${id} created and loaded.`);
        this.emit('partition:created', {
            id,
            name,
            active: this.partitionConfigs.get(id)?.active ?? false, // Get current active state
        });
        (0, log_1.log)('info', `[PartitionedVectorDB] Successfully created and loaded partition: ${id}`);
        return id;
    }
    /** Set the active partition. Handles loading and updating config states. */
    async setActivePartition(id, force = false) {
        await this._ensureInitialized(force);
        const newActiveConfig = this.partitionConfigs.get(id);
        if (!newActiveConfig)
            throw new Error(`Partition with ID ${id} not found`);
        if (this.activePartitionId === id)
            return; // Already active
        // Ensure target partition is loaded (marks as recently used)
        const db = await this._loadPartition(id); // Load DB & potentially HNSW
        if (!db)
            throw new Error(`Failed to load partition ${id} to activate it.`);
        const previousActiveId = this.activePartitionId;
        let configChanged = false;
        // Deactivate previous
        if (previousActiveId) {
            const prevActiveConfig = this.partitionConfigs.get(previousActiveId);
            if (prevActiveConfig && prevActiveConfig.active) {
                prevActiveConfig.active = false;
                configChanged = true;
                // No need to save individually, scheduleSaveConfigs handles it
            }
        }
        // Activate new
        if (!newActiveConfig.active) {
            newActiveConfig.active = true;
            configChanged = true;
            // No need to save individually
        }
        this.activePartitionId = id;
        // Save configs if state changed (debounced)
        if (configChanged) {
            this.scheduleSaveConfigs();
        }
        this.emit('partition:activated', { id });
        (0, log_1.log)('info', `[PartitionedVectorDB] Activated partition: ${id}`);
    }
    /** Ensures active partition has capacity, creates/activates new one if needed. */
    async _ensureActivePartitionHasCapacity(neededCapacity = 1) {
        // Rely on getActivePartition to load the current active one if needed
        (0, log_1.log)('info', `[PartitionedVectorDB] Ensuring active partition has capacity...`);
        let activePartition = await this.getActivePartition(); // Loads/returns active partition
        (0, log_1.log)('info', `[PartitionedVectorDB] Active partition: ${this.activePartitionId}`);
        let currentActiveId = this.activePartitionId; // Store current ID
        // Handle case where there is no active partition initially or after creation fails
        if (!activePartition || !currentActiveId) {
            if (this.autoCreatePartitions) {
                (0, log_1.log)('warn', '[PartitionedVectorDB] No usable active partition found. Attempting to create a new one.');
                // Ensure uniqueness and avoid collisions during rapid calls
                const newId = `p-${Date.now()}-${Math.random().toString(36).substring(2, 7)}`;
                await this.createPartition(newId, `Auto Partition ${this.partitionConfigs.size + 1}`, { setActive: true });
                activePartition = await this.getActivePartition(); // Fetch the newly created active partition
                currentActiveId = this.activePartitionId; // Update the active ID
                if (!activePartition || !currentActiveId) {
                    // Check again after creation attempt
                    throw new Error('[PartitionedVectorDB] Failed to create or load a new active partition automatically.');
                }
            }
            else {
                throw new Error('[PartitionedVectorDB] No active partition available and autoCreatePartitions is disabled.');
            }
        }
        // Now we are sure activePartition and currentActiveId are valid
        const activeConfig = this.partitionConfigs.get(currentActiveId); // Config must exist
        // Check capacity against the config's count
        if (activeConfig.vectorCount + neededCapacity > this.partitionCapacity) {
            if (this.autoCreatePartitions) {
                (0, log_1.log)('info', `[PartitionedVectorDB] Active partition ${currentActiveId} nearing capacity (${activeConfig.vectorCount}/${this.partitionCapacity}). Creating and activating new partition.`);
                const newId = `p-${Date.now()}-${Math.random().toString(36).substring(2, 7)}`;
                await this.createPartition(newId, `Auto Partition ${this.partitionConfigs.size + 1}`, { setActive: true });
                // Re-fetch the *new* active partition
                activePartition = await this.getActivePartition();
                if (!activePartition) {
                    throw new Error(`[PartitionedVectorDB] Failed to load the newly created active partition ${this.activePartitionId}.`);
                }
                (0, log_1.log)('info', `[PartitionedVectorDB] Switched to new active partition: ${this.activePartitionId}`);
            }
            else {
                throw new Error(`[PartitionedVectorDB] Active partition ${currentActiveId} has insufficient capacity (${activeConfig.vectorCount}/${this.partitionCapacity}) and autoCreatePartitions is disabled.`);
            }
        }
        // Return the partition guaranteed to have capacity (either original or newly created)
        return activePartition;
    }
    /** Add a single vector */
    async addVector(id, vector, metadata) {
        await this._ensureInitialized();
        const partition = await this._ensureActivePartitionHasCapacity(1);
        const partitionId = this.activePartitionId; // Must be set after ensureCapacity
        const vectorId = partition.addVector(id, vector, metadata);
        // Update count in config (in memory) and schedule save
        const config = this.partitionConfigs.get(partitionId);
        config.vectorCount++;
        this.scheduleSaveConfigs(); // Debounced save
        // Update HNSW Index (if loaded) incrementally
        const hnswIndex = this.hnswIndices.get(partitionId);
        const addedVector = partition.getVector(vectorId); // Get the Float32Array vector
        if (hnswIndex && addedVector) {
            try {
                hnswIndex.addPoint(addedVector, vectorId); // Assumes HNSW has addPoint method
                (0, log_1.log)('info', `[PartitionedVectorDB] Added point ${vectorId} to HNSW index for partition ${partitionId}`);
            }
            catch (error) {
                (0, log_1.log)('error', `[PartitionedVectorDB] Error adding point ${vectorId} to HNSW index for ${partitionId}:`, error);
                // Should we invalidate the index? Mark for rebuild?
                this.emit('partition:error', {
                    id: partitionId,
                    error,
                    operation: 'addPointHNSW',
                });
            }
        }
        this.emit('vector:add', { partitionId, vectorId, metadata });
        return { partitionId, vectorId };
    }
    /** Bulk add vectors */
    async bulkAdd(vectors) {
        await this._ensureInitialized();
        if (vectors.length === 0)
            return { count: 0, partitionIds: [] };
        (0, log_1.log)('info', `[PartitionedVectorDB] Starting bulk add of ${vectors.length} vectors...`);
        let totalAddedCount = 0;
        const partitionIdsUsed = new Set();
        let remainingVectors = vectors;
        while (remainingVectors.length > 0) {
            // Determine needed capacity for the next chunk (could be all remaining)
            const needed = remainingVectors.length;
            const partition = await this._ensureActivePartitionHasCapacity(1); // Check for at least 1 slot
            const partitionId = this.activePartitionId;
            const config = this.partitionConfigs.get(partitionId);
            const availableCapacity = this.partitionCapacity - config.vectorCount;
            const batchSize = Math.min(remainingVectors.length, availableCapacity);
            if (batchSize <= 0) {
                // This indicates ensureCapacity might have created a new partition, but we still hit the condition somehow
                // Or the partition is genuinely full and autoCreate is off.
                if (!this.autoCreatePartitions) {
                    throw new Error(`Partition ${partitionId} is full (${config.vectorCount}/${this.partitionCapacity}), cannot add remaining ${remainingVectors.length} vectors (autoCreatePartitions is off).`);
                }
                else {
                    // If autoCreate is ON, ensureCapacity should have switched partitions.
                    // This case implies a potential logic error or race condition. Let's log and retry the loop.
                    (0, log_1.log)('warn', `[PartitionedVectorDB] Bulk add loop detected zero batch size for partition ${partitionId} despite capacity check. Retrying capacity check.`);
                    continue; // Retry the loop, hoping ensureCapacity fixes it
                }
            }
            const batchToAdd = remainingVectors.slice(0, batchSize);
            (0, log_1.log)('info', `[PartitionedVectorDB] Adding batch of ${batchToAdd.length} vectors to partition ${partitionId} (Capacity: ${config.vectorCount}/${this.partitionCapacity})`);
            const countInBatch = partition.bulkAdd(batchToAdd); // Underlying bulkAdd
            // --- Update HNSW Index incrementally for the batch ---
            const hnswIndex = this.hnswIndices.get(partitionId);
            if (hnswIndex && countInBatch > 0) {
                (0, log_1.log)('info', `[PartitionedVectorDB] Incrementally updating HNSW index for partition ${partitionId} with ${countInBatch} vectors...`);
                const vectorsForIndex = [];
                // We need the actual IDs assigned by bulkAdd if they weren't provided
                // This requires bulkAdd to return the added items or query them back - potentially inefficient.
                // For now, let's assume bulkAdd uses the provided IDs or we requery.
                // Requerying is safer but slower. Let's assume HNSW addPoints handles this if needed.
                for (const item of batchToAdd.slice(0, countInBatch)) {
                    // Process only successfully added items
                    const actualId = item.id ?? null; // How to get the real ID if it was auto-generated? This is a limitation.
                    const vectorData = partition.getVector(actualId); // Re-fetch vector data - INEFFICIENT
                    if (vectorData && actualId !== null) {
                        vectorsForIndex.push({ vector: vectorData, id: actualId });
                    }
                    else if (!item.id) {
                        (0, log_1.log)('warn', `[PartitionedVectorDB] Cannot update HNSW incrementally for auto-generated ID during bulk add.`);
                        // Mark index as potentially stale?
                    }
                }
                if (vectorsForIndex.length > 0) {
                    try {
                        // Assuming hnswIndex has a bulk add method like addPoints(vectors: {vector: Float32Array, id: number|string}[])
                        // hnswIndex.addPoints(vectorsForIndex);
                        // If not, add one by one (less efficient)
                        for (const { vector, id } of vectorsForIndex) {
                            hnswIndex.addPoint(vector, id);
                        }
                        (0, log_1.log)('info', `[PartitionedVectorDB] Finished updating HNSW index for partition ${partitionId} batch.`);
                    }
                    catch (error) {
                        (0, log_1.log)('error', `[PartitionedVectorDB] Error bulk adding points to HNSW index for ${partitionId}:`, error);
                        this.emit('partition:error', {
                            id: partitionId,
                            error,
                            operation: 'bulkAddPointHNSW',
                        });
                    }
                }
            }
            // --- End HNSW Update ---
            totalAddedCount += countInBatch;
            config.vectorCount += countInBatch; // Update config count
            partitionIdsUsed.add(partitionId);
            remainingVectors = remainingVectors.slice(batchSize);
            (0, log_1.log)('info', `[PartitionedVectorDB] Added ${countInBatch} vectors to ${partitionId}. Total added: ${totalAddedCount}. Remaining: ${remainingVectors.length}. Partition size: ${config.vectorCount}`);
            // No explicit check needed here for next loop, _ensureActivePartitionHasCapacity will handle it.
        }
        // Schedule config save if counts were updated
        if (totalAddedCount > 0) {
            this.scheduleSaveConfigs();
        }
        (0, log_1.log)('info', `[PartitionedVectorDB] Bulk add complete. Added ${totalAddedCount} vectors across partitions: [${Array.from(partitionIdsUsed).join(', ')}]`);
        this.emit('vectors:bulkAdd', {
            count: totalAddedCount,
            partitionIds: Array.from(partitionIdsUsed),
        });
        return {
            count: totalAddedCount,
            partitionIds: Array.from(partitionIdsUsed),
        };
    }
    /** Get a vector by ID. Searches loaded partitions only. */
    async getVector(id) {
        await this._ensureInitialized();
        for (const partitionId of this.loadedPartitions.keys()) {
            const partition = this.loadedPartitions.peek(partitionId);
            if (partition) {
                const vector = partition.getVector(id); // Use the base DB's getVector
                if (vector) {
                    this.loadedPartitions.get(partitionId); // Mark as recently used
                    return { partitionId, vector };
                }
            }
        }
        return null; // Not found in loaded partitions
    }
    /** Get metadata by ID. Searches loaded partitions only. */
    async getMetadata(id) {
        await this._ensureInitialized();
        for (const partitionId of this.loadedPartitions.keys()) {
            const partition = this.loadedPartitions.peek(partitionId);
            if (partition) {
                const metadata = partition.getMetadata(id); // Use the base DB's getMetadata
                if (metadata !== null && metadata !== undefined) {
                    this.loadedPartitions.get(partitionId); // Mark as recently used
                    return { partitionId, metadata };
                }
            }
        }
        return null; // Not found
    }
    /** Delete a vector by ID. Searches loaded partitions. */
    async deleteVector(id) {
        await this._ensureInitialized();
        let deleted = false;
        let partitionIdFound = null;
        for (const partitionId of this.loadedPartitions.keys()) {
            const partition = this.loadedPartitions.peek(partitionId);
            if (partition?.hasVector(id)) {
                // Check existence first
                const deletedLocally = partition.deleteVector(id); // Use base DB's delete
                if (deletedLocally) {
                    this.loadedPartitions.get(partitionId); // Mark as used
                    deleted = true;
                    partitionIdFound = partitionId;
                    // --- Remove from HNSW Index ---
                    const hnswIndex = this.hnswIndices.get(partitionId);
                    if (hnswIndex) {
                        try {
                            hnswIndex.markDelete(id); // Assumes HNSW has markDelete or similar
                            (0, log_1.log)('info', `[PartitionedVectorDB] Marked point ${id} for deletion in HNSW index for partition ${partitionId}`);
                            // Note: Actual removal might happen during maintenance/compaction in HNSW
                        }
                        catch (error) {
                            (0, log_1.log)('error', `[PartitionedVectorDB] Error marking point ${id} for deletion in HNSW index for ${partitionId}:`, error);
                            this.emit('partition:error', {
                                id: partitionId,
                                error,
                                operation: 'deletePointHNSW',
                            });
                        }
                    }
                    // --- End HNSW Update ---
                    break; // Assume unique IDs, stop searching
                }
            }
        }
        // Update config count if deleted
        if (deleted && partitionIdFound) {
            const config = this.partitionConfigs.get(partitionIdFound);
            if (config) {
                config.vectorCount = Math.max(0, config.vectorCount - 1);
                this.scheduleSaveConfigs(); // Schedule config save
            }
            this.emit('vector:delete', {
                partitionId: partitionIdFound,
                vectorId: id,
            });
            (0, log_1.log)('info', `[PartitionedVectorDB] Deleted vector ${id} from partition ${partitionIdFound}. New count: ${config?.vectorCount}`);
        }
        return deleted;
    }
    /** Update metadata for a vector by ID. Searches loaded partitions only. */
    async updateMetadata(id, data) {
        await this._ensureInitialized();
        for (const partitionId of this.loadedPartitions.keys()) {
            const partition = this.loadedPartitions.peek(partitionId);
            if (partition?.hasVector(id)) {
                // Vector found, update its metadata
                const updated = partition.updateMetadata(id, data);
                if (updated) {
                    this.loadedPartitions.get(partitionId); // Mark as recently used
                    this.emit('vector:metadataUpdate', {
                        partitionId,
                        vectorId: id,
                    });
                    (0, log_1.log)('info', `[PartitionedVectorDB] Updated metadata for vector ${id} in partition ${partitionId}`);
                }
                return updated;
            }
        }
        (0, log_1.log)('warn', `[PartitionedVectorDB] Could not update metadata: Vector ${id} not found in any loaded partition`);
        return false; // Vector not found or update failed
    }
    async updateVector(id, vector) {
        await this._ensureInitialized();
        for (const partitionId of this.loadedPartitions.keys()) {
            const partition = this.loadedPartitions.peek(partitionId);
            if (partition?.hasVector(id)) {
                // Vector found, update it
                const updated = partition.updateVector(id, vector);
                if (updated) {
                    this.loadedPartitions.get(partitionId); // Mark as recently used
                    (0, log_1.log)('info', `[PartitionedVectorDB] Updated vector ${id} in partition ${partitionId}`);
                    return true;
                }
            }
        }
        (0, log_1.log)('warn', `[PartitionedVectorDB] Could not update vector: Vector ${id} not found in any loaded partition`);
        return false; // Vector not found or update failed
    }
    /** Find nearest neighbors (standard search). Searches across specified or all loaded partitions. */
    async findNearest(query, k = 10, options = {} // Uses SearchOptions now
    ) {
        await this._ensureInitialized();
        const queryVector = query instanceof Float32Array ? query : new Float32Array(query);
        // Determine partitions to search
        const partitionIdsToSearch = options.partitionIds
            ? options.partitionIds.filter((id) => this.loadedPartitions.has(id)) // Search specified *loaded* partitions
            : Array.from(this.loadedPartitions.keys()); // Default: search all *loaded* partitions
        if (partitionIdsToSearch.length === 0) {
            (0, log_1.log)('warn', '[PartitionedVectorDB] No valid partitions specified or loaded to search.');
            return [];
        }
        (0, log_1.log)('info', `[PartitionedVectorDB] Performing standard search on partitions: [${partitionIdsToSearch.join(', ')}]`);
        // Perform search in parallel
        const searchPromises = partitionIdsToSearch.map(async (partitionId) => {
            const partition = this.loadedPartitions.get(partitionId); // Get (marks as used)
            if (partition) {
                try {
                    // Pass options down to the underlying ClusteredVectorDB's findNearest
                    return partition.findNearest(queryVector, k, {
                        filter: options.filter,
                        metric: options.distanceMetric,
                    });
                }
                catch (err) {
                    (0, log_1.log)('error', `[PartitionedVectorDB] Error searching partition ${partitionId}:`, err);
                    this.emit('partition:error', {
                        id: partitionId,
                        error: err,
                        operation: 'search',
                    });
                    return [];
                }
            }
            else {
                return []; // Should not happen if using loadedPartitions.keys()
            }
        });
        const allResultsNested = await Promise.all(searchPromises);
        const allResultsFlat = allResultsNested.flat();
        // Sort combined results and take top k
        allResultsFlat.sort((a, b) => a.dist - b.dist);
        return allResultsFlat.slice(0, k);
    }
    /** Get database statistics. */
    async getStats() {
        // Ensure initialization is complete, but don't throw if called before fully ready
        if (!this.isInitialized) {
            (0, log_1.log)('warn', '[PartitionedVectorDB] getStats called before initialization complete. Stats might be incomplete.');
            // Return partial stats if possible?
        }
        // Even if not fully initialized, partitionConfigs might be loaded
        const totalConfiguredVectors = Array.from(this.partitionConfigs.values()).reduce((sum, config) => sum + (config.vectorCount || 0), 0);
        const loadedPartitionStats = {};
        let totalVectorsLoaded = 0;
        let totalMemoryLoaded = 0;
        // Iterate safely over potentially changing cache during async operations
        const loadedIds = Array.from(this.loadedPartitions.keys());
        for (const partitionId of loadedIds) {
            const partition = this.loadedPartitions.peek(partitionId);
            if (partition) {
                try {
                    const stats = partition.getStats(); // Get stats from underlying DB
                    loadedPartitionStats[partitionId] = stats;
                    totalVectorsLoaded += stats.vectorCount;
                    totalMemoryLoaded += stats.memoryUsage ?? 0;
                }
                catch (e) {
                    (0, log_1.log)('warn', `[PartitionedVectorDB] Could not retrieve stats for loaded partition ${partitionId}:`, e);
                }
            }
        }
        // Add HNSW stats
        const hnswStats = {};
        const hnswIds = Array.from(this.hnswIndices.keys());
        for (const id of hnswIds) {
            hnswStats[id] = this.getHNSWStats(id);
        }
        return {
            status: this.isInitialized ? (this.isClosing ? 'closing' : 'initialized') : 'initializing',
            partitions: {
                totalConfigured: this.partitionConfigs.size,
                loadedCount: this.loadedPartitions.size,
                maxLoaded: this.maxActivePartitions,
                activeId: this.activePartitionId,
                loadedIds: loadedIds,
                configs: Array.from(this.partitionConfigs.values()), // Include configs in stats
            },
            vectors: {
                totalConfigured: totalConfiguredVectors, // Sum from configs
                totalInMemory: totalVectorsLoaded, // Sum from loaded DB stats
            },
            memory: {
                estimatedUsageBytes: totalMemoryLoaded,
                lruCacheSize: this.loadedPartitions.size, // Current LRU size
            },
            indices: {
                hnswLoadedCount: this.hnswIndices.size,
                hnswLoadedIds: hnswIds,
                hnswStats: hnswStats, // Include detailed HNSW stats
            },
            settings: {
                partitionCapacity: this.partitionCapacity,
                autoCreatePartitions: this.autoCreatePartitions,
                useCompression: this.useCompression,
                suggestedVectorSize: this.vectorSize,
                autoLoadHNSW: this.autoLoadHNSW,
                maxActivePartitions: this.maxActivePartitions,
            },
            loadedPartitionDetails: loadedPartitionStats, // Keep detailed stats per loaded partition
        };
    }
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
    async getMetadataWithFieldAcrossPartitions(criteria, values, option) {
        await this._ensureInitialized();
        const results = [];
        // Search across all loaded partitions
        for (const partitionId of this.loadedPartitions.keys()) {
            const partition = this.loadedPartitions.peek(partitionId);
            if (partition) {
                const partitionResults = partition.getMetadataWithField(criteria, values, option);
                // Add partition ID to each result
                for (const item of partitionResults) {
                    results.push({
                        partitionId,
                        vectorId: item.id,
                        metadata: item.metadata,
                    });
                }
                // Mark partition as recently used
                this.loadedPartitions.get(partitionId);
            }
        }
        if (option?.limit) {
            return results.slice(0, option.limit);
        }
        return results;
    }
    /** Get partition configurations */
    getPartitionConfigs() {
        // No need for ensureInitialized check here, return what's available
        return Array.from(this.partitionConfigs.values());
    }
    /**
     * Extract communities of related vectors based on distance threshold across specified partitions.
     * A community is a group of vectors where each vector is related to at least one other vector in the group.
     *
     * @param threshold - The maximum distance between vectors to consider them related
     * @param options - Options including distance metric, partition IDs, and metadata inclusion
     * @returns Array of communities, where each community is an array of related vector information
     */
    async extractCommunities(threshold, options = {}) {
        await this._ensureInitialized();
        // Determine partitions to process
        const partitionIds = options.partitionIds
            ? options.partitionIds.filter((id) => this.loadedPartitions.has(id)) // Filter to loaded partitions
            : Array.from(this.loadedPartitions.keys()); // Default: process all loaded partitions
        if (partitionIds.length === 0) {
            (0, log_1.log)('warn', '[PartitionedVectorDB] No valid partitions to extract communities from');
            return [];
        }
        (0, log_1.log)('info', `[PartitionedVectorDB] Extracting vector communities with threshold ${threshold} from ${partitionIds.length} partitions...`);
        // Step 1: Extract communities from individual partitions
        const partitionCommunities = new Map();
        for (const partitionId of partitionIds) {
            const partition = await this._loadPartition(partitionId);
            if (!partition) {
                (0, log_1.log)('warn', `[PartitionedVectorDB] Skipping partition ${partitionId} - could not load`);
                continue;
            }
            try {
                const communities = partition.extractCommunities(threshold, options.metric || this.defaultClusterOptions.distanceMetric);
                partitionCommunities.set(partitionId, communities);
                (0, log_1.log)('info', `[PartitionedVectorDB] Extracted ${communities.length} communities from partition ${partitionId}`);
            }
            catch (error) {
                (0, log_1.log)('error', `[PartitionedVectorDB] Error extracting communities from partition ${partitionId}:`, error);
                this.emit('partition:error', {
                    id: partitionId,
                    error,
                    operation: 'extractCommunities',
                });
            }
        }
        // Step 2: Process cross-partition relationships (using extractRelationships)
        // This could be expensive for large datasets, but provides more accurate communities
        const crossPartitionGraph = new Map();
        // Function to create a globally unique ID
        const getGlobalId = (partitionId, localId) => `${partitionId}:${localId}`;
        // Initialize graph with all vectors from communities
        for (const [partitionId, communities] of partitionCommunities.entries()) {
            for (const community of communities) {
                for (const node of community) {
                    const globalId = getGlobalId(partitionId, node.id);
                    if (!crossPartitionGraph.has(globalId)) {
                        crossPartitionGraph.set(globalId, new Set());
                    }
                }
            }
        }
        // Add connections within each partition's communities
        for (const [partitionId, communities] of partitionCommunities.entries()) {
            for (const community of communities) {
                // For each community, add edges between all nodes
                for (let i = 0; i < community.length; i++) {
                    const nodeId1 = getGlobalId(partitionId, community[i].id);
                    for (let j = i + 1; j < community.length; j++) {
                        const nodeId2 = getGlobalId(partitionId, community[j].id);
                        // Add bidirectional connections
                        crossPartitionGraph.get(nodeId1)?.add(nodeId2);
                        crossPartitionGraph.get(nodeId2)?.add(nodeId1);
                    }
                }
            }
        }
        // Step 3: Build global communities using depth-first search
        const globalCommunities = [];
        const visited = new Set();
        for (const [globalId] of crossPartitionGraph.entries()) {
            if (!visited.has(globalId)) {
                const community = [];
                // DFS to find connected components
                const dfs = (nodeGlobalId) => {
                    if (visited.has(nodeGlobalId))
                        return;
                    visited.add(nodeGlobalId);
                    // Parse the global ID
                    const [partId, localId] = nodeGlobalId.split(':');
                    // Find metadata if requested
                    let metadata = null;
                    if (options.includeMetadata !== false) {
                        const partition = this.loadedPartitions.peek(partId);
                        if (partition) {
                            metadata = partition.getMetadata(localId);
                        }
                    }
                    // Add to community
                    community.push({
                        id: localId,
                        partitionId: partId,
                        metadata: metadata || undefined,
                    });
                    // Visit neighbors
                    const neighbors = crossPartitionGraph.get(nodeGlobalId) || new Set();
                    for (const neighbor of neighbors) {
                        dfs(neighbor);
                    }
                };
                dfs(globalId);
                // Add community if it contains at least 2 members
                if (community.length > 1) {
                    globalCommunities.push(community);
                }
            }
        }
        (0, log_1.log)('info', `[PartitionedVectorDB] Final result: ${globalCommunities.length} communities identified across partitions`);
        return globalCommunities;
    }
    /**
     * Extract relationships between vectors based on distance threshold across specified partitions.
     *
     * @param threshold - The maximum distance between vectors to consider them related
     * @param options - Options including distance metric, partition filtering, and metadata inclusion
     * @returns Array of relationships with vectorIds, partitionIds, optional metadata, and distances
     */
    async extractRelationships(threshold, options = {}) {
        await this._ensureInitialized();
        // Determine partitions to process
        const partitionIds = options.partitionIds ? options.partitionIds.filter((id) => this.loadedPartitions.has(id)) : Array.from(this.loadedPartitions.keys());
        if (partitionIds.length === 0) {
            (0, log_1.log)('warn', '[PartitionedVectorDB] No valid partitions to extract relationships from');
            return [];
        }
        (0, log_1.log)('info', `[PartitionedVectorDB] Extracting vector relationships with threshold ${threshold} from ${partitionIds.length} partitions...`);
        const relationships = [];
        // Process each partition individually
        for (const partitionId of partitionIds) {
            const partition = await this._loadPartition(partitionId);
            if (!partition) {
                (0, log_1.log)('warn', `[PartitionedVectorDB] Skipping partition ${partitionId} - could not load`);
                continue;
            }
            try {
                // Extract relationships within this partition
                const partitionRelationships = partition.extractRelationships(threshold, options.metric || this.defaultClusterOptions.distanceMetric);
                // Transform the results to include partition IDs and optionally metadata
                for (const rel of partitionRelationships) {
                    const relationship = {
                        vector1: {
                            id: rel.vector1,
                            partitionId,
                            metadata: undefined,
                        },
                        vector2: {
                            id: rel.vector2,
                            partitionId,
                            metadata: undefined,
                        },
                        distance: rel.distance,
                    };
                    // Add metadata if requested
                    if (options.includeMetadata !== false) {
                        const metadata1 = partition.getMetadata(rel.vector1);
                        const metadata2 = partition.getMetadata(rel.vector2);
                        if (metadata1) {
                            relationship.vector1.metadata = metadata1;
                        }
                        if (metadata2) {
                            relationship.vector2.metadata = metadata2;
                        }
                    }
                    relationships.push(relationship);
                }
                (0, log_1.log)('info', `[PartitionedVectorDB] Extracted ${partitionRelationships.length} relationships from partition ${partitionId}`);
            }
            catch (error) {
                (0, log_1.log)('error', `[PartitionedVectorDB] Error extracting relationships from partition ${partitionId}:`, error);
                this.emit('partition:error', {
                    id: partitionId,
                    error,
                    operation: 'extractRelationships',
                });
            }
        }
        (0, log_1.log)('info', `[PartitionedVectorDB] Total: ${relationships.length} relationships extracted across partitions`);
        return relationships;
    }
}
exports.PartitionedVectorDB = PartitionedVectorDB;
// --- END OF FILE partitioned_vector_db.ts ---
//# sourceMappingURL=partitioned_vector_db.js.map