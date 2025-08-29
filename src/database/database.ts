import { EventEmitter } from 'events';
import { existsSync, mkdirSync } from 'fs';
import { LRUCache } from 'lru-cache';
import path from 'path'; // Import path for directory handling
import configDefaults from '../config'; // Import the default config object
import { UnifiedSearch } from '../search/unified_search';
import {
  BuildIndexHNSWOptions,
  DatabaseEvents,
  DatabaseOptions,
  DatabaseStats,
  DistanceMetric,
  PartitionedDBEventData,
  PartitionedDBStats,
  PartitionedVectorDBInterface,
  PerformanceMetrics,
  SearchResult,
  TypedEventEmitter,
  UnifiedSearchOptions,
  UnifiedSearchPartitionedStats,
  Vector,
  VectorData,
} from '../types';
import { createTimer, Timer } from '../utils/profiling';
import { VectorDBMonitor } from '../utils/vector_monitoring';
import { PartitionedVectorDB } from '../vector/partitioned_vector_db';

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
export class Database extends (EventEmitter as new () => TypedEventEmitter<DatabaseEvents>) {
  // Core Components
  private vectorDB!: PartitionedVectorDBInterface; // Definite assignment assertion
  private unifiedSearch!: UnifiedSearch; // Definite assignment assertion

  // Caching & Monitoring
  private readonly searchCache: LRUCache<string, SearchResult[]>;
  private readonly monitor: VectorDBMonitor | null;
  private readonly timer: Timer = createTimer(); // General purpose timer

  // State & Configuration
  private readonly options: Required<DatabaseOptions>; // Use Required for internal consistency
  private isClosed: boolean = false;
  private isReady: boolean = false;
  public initializationPromise: Promise<void>; // Tracks async initialization

  // Background Tasks
  private autoSaveTimer: NodeJS.Timeout | null = null;

  // Concurrency Control
  private readonly activeSearchPromises: Set<Promise<any>> = new Set();

  // Performance Metrics (Simplified - detailed metrics come from components)
  private metrics: PerformanceMetrics = {
    queries: 0,
    totalSearchTime: 0,
    avgSearchTime: 0,
    cacheHits: 0,
    cacheMisses: 0,
    queryTimes: [], // Consider capping size
    avgAddTime: 0, // Currently not tracked here, relies on events if needed
    totalAddTime: 0, // Currently not tracked here
  };

  // Constants
  private static readonly DEFAULT_DB_PATH = 'database';
  private static readonly PARTITIONS_DIR_NAME = 'partitions';
  private static readonly CACHE_EVENTS = ['vector:add', 'vector:delete', 'vectors:bulkAdd', 'partition:created', 'partition:unloaded'];
  private static readonly METRICS_EVENTS = ['vector:add', 'vector:delete', 'vectors:bulkAdd', 'partition:loaded', 'partition:unloaded'];

  /**
   * Creates a new Database instance. Initialization is asynchronous.
   * Listen for the 'ready' event or await the ready() promise before use.
   *
   * @param options Configuration options for the database
   */
  constructor(options: DatabaseOptions) {
    super();

    this.options = this.mergeOptionsWithDefaults(options);
    this.searchCache = this.createSearchCache();
    this.monitor = this.createMonitor();

    this.initializationPromise = this.initialize();
  }

  /**
   * Merges user options with default configuration
   */
  private mergeOptionsWithDefaults(options: DatabaseOptions): Required<DatabaseOptions> {
    return {
      vectorSize: options.vectorSize ?? configDefaults.defaults.vectorSize,
      clustering: { ...configDefaults.clustering, ...options.clustering },
      partitioning: {
        ...configDefaults.partitioning,
        ...options.partitioning,
      },
      indexing: { ...configDefaults.indexing, ...options.indexing },
      cacheSize: options.cacheSize ?? configDefaults.defaults.cacheSize,
      maxConcurrentSearches: options.maxConcurrentSearches ?? configDefaults.defaults.maxConcurrentSearches,
      persistence: { ...configDefaults.persistence, ...options.persistence },
      backup: { ...configDefaults.backup, ...options.backup },
      monitoring: { ...configDefaults.monitoring, ...options.monitoring },
    };
  }

  /**
   * Creates and configures the search cache
   */
  private createSearchCache(): LRUCache<string, SearchResult[]> {
    return new LRUCache<string, SearchResult[]>({
      max: this.options.cacheSize,
    });
  }

  /**
   * Creates monitor instance if monitoring is enabled
   */
  private createMonitor(): VectorDBMonitor | null {
    if (!this.options.monitoring.enable) {
      return null;
    }

    return new VectorDBMonitor({
      interval: this.options.monitoring.intervalMs,
      logToConsole: this.options.monitoring.logToConsole,
      enableDatabaseMetrics: this.options.monitoring.enableDatabaseMetrics,
      enableSystemMetrics: this.options.monitoring.enableSystemMetrics,
    });
  }

  /**
   * Main asynchronous initialization sequence
   */
  private async initialize(): Promise<void> {
    try {
      console.log('[Database] Starting initialization...');
      this.emit('initializing', undefined);

      await this.initializeStorage();
      await this.initializeVectorDB();
      await this.initializeUnifiedSearch();
      await this.handleInitialIndexing();
      this.startBackgroundTasks();

      this.markAsReady();
    } catch (error: any) {
      await this.handleInitializationError(error);
      throw error;
    }
  }

  /**
   * Ensures base directory exists for storage
   */
  private async initializeStorage(): Promise<void> {
    const baseDir = path.join(
      process.cwd(),
      this.options.persistence.dbPath || Database.DEFAULT_DB_PATH
    );

    this.ensureDirectoryExists(baseDir, Database.PARTITIONS_DIR_NAME);
  }

  /**
   * Initializes the PartitionedVectorDB instance
   */
  private async initializeVectorDB(): Promise<void> {
    console.log('[Database] Initializing PartitionedVectorDB...');

    const baseDir = path.join(
      process.cwd(),
      this.options.persistence.dbPath || Database.DEFAULT_DB_PATH
    );

    this.vectorDB = new PartitionedVectorDB({
      partitionsDir: path.join(baseDir, Database.PARTITIONS_DIR_NAME),
      partitionCapacity: this.options.partitioning.partitionCapacity,
      maxActivePartitions: this.options.partitioning.maxActivePartitions,
      autoCreatePartitions: this.options.partitioning.autoCreatePartitions,
      autoLoadPartitions: this.options.partitioning.autoLoadPartitions,
      autoLoadHNSW: this.options.indexing.autoLoad,
      vectorSize: this.options.vectorSize,
      useCompression: this.options.clustering.useCompression,
      clusterOptions: this.options.clustering,
    });

    this.setupEventListeners();

    if (this.vectorDB.IsReady()) {
      console.log('[Database] PartitionedVectorDB is ready.');
    } else {
      await this.vectorDB.initializationPromise;
    }
  }

  /**
   * Initializes the UnifiedSearch instance
   */
  private async initializeUnifiedSearch(): Promise<void> {
    console.log('[Database] Initializing UnifiedSearch...');

    this.unifiedSearch = new UnifiedSearch(this.vectorDB, {
      debug: this.options.monitoring.logToConsole,
    });

    this.setupUnifiedSearchListeners();
  }

  /**
   * Handles initial indexing based on configuration
   */
  private async handleInitialIndexing(): Promise<void> {
    if (!this.options.indexing.buildOnStart) {
      return;
    }

    try {
      if (this.options.indexing.autoLoad) {
        console.log('[Database] Loading existing HNSW indices...');
        await this.vectorDB.loadHNSWIndices();
      }

      console.log('[Database] Building initial indices...');
      const progressCallback = (progress: number) => {
        console.log(`[Database] Index build progress: ${progress}%`);
        this.emit('index:progress', {
          message: 'Building initial indices...',
          progress,
        });
      };

      await this.buildIndexes(undefined, {
        force: true,
        dimensionAware: true,
        progressCallback,
      });
    } catch (error: any) {
      console.warn(`[Database] Initial indexing issue: ${error.message}`);
      this.emit('warn', {
        message: `Initial index handling issue: ${error.message}`,
        context: 'handleInitialIndexing',
        error,
      });
    }
  }

  /**
   * Starts background tasks like monitoring and auto-save
   */
  private startBackgroundTasks(): void {
    this.monitor?.start();
    this._setupAutoSave();
  }

  /**
   * Marks database as ready and emits ready event
   */
  private async markAsReady(): Promise<void> {
    this.isReady = true;
    console.log('[Database] Initialization successful. Ready for operations.');

    try {
      const initialStats = await this.getStats();
      console.log(`[Database] Initial State: ${initialStats.database?.partitions?.totalConfigured} partitions configured, ${initialStats.database?.partitions?.loadedCount} loaded`);
    } catch (error: any) {
      console.warn('[Database] Could not retrieve initial stats:', error.message);
    }

    this.emit('ready', undefined);
  }

  /**
   * Handles initialization errors
   */
  private async handleInitializationError(error: any): Promise<void> {
    console.error('[Database] FATAL: Database initialization failed:', error);
    this.isClosed = true;
    this.monitor?.stop();

    this.emit('error', {
      message: `[Database] Database initialization failed: ${error.message}`,
      error,
      context: 'initialize',
    });
  }

  /**
   * Ensures a directory exists, creating it if necessary
   */
  private ensureDirectoryExists(dirPath: string, purpose: string): void {
    try {
      if (!existsSync(dirPath)) {
        mkdirSync(dirPath, { recursive: true });
        console.log(`[Database] Created ${purpose} directory: ${dirPath}`);
      }
    } catch (error: any) {
      throw new Error(`[Database] Failed to create ${purpose} directory at ${dirPath}: ${error.message}`);
    }
  }

  /** Sets up internal event listeners for PartitionedVectorDB events. */
  private setupEventListeners(): void {
    if (!this.vectorDB || typeof (this.vectorDB as any).on !== 'function') return;

    const dbEmitter = this.vectorDB as any as TypedEventEmitter<PartitionedDBEventData>;

    // Generic handler to forward events, clear cache, and update monitor
    const handleDbEvent = <E extends keyof PartitionedDBEventData>(
      eventName: E,
      data: PartitionedDBEventData[E]
    ) => {
      if (this.isClosed) return;

      this.handleCacheInvalidation(eventName);
      this.forwardEvent(eventName, data);
      this.updateMetrics(eventName, data);
      this.recordEventInMonitor(eventName, data);
    };

    // Register listeners for all relevant events
    this.registerDatabaseEventListeners(dbEmitter, handleDbEvent);
    this.registerIndexingEventListeners(dbEmitter);
  }

  /**
   * Handles cache invalidation based on event type
   */
  private handleCacheInvalidation(eventName: string): void {
    if (Database.CACHE_EVENTS.includes(eventName as any)) {
      this.searchCache.clear();
    }
  }

  /**
   * Forwards events to external listeners
   */
  private forwardEvent(eventName: string, data: any): void {
    this.emit(eventName as keyof DatabaseEvents, data as any);
  }

  /**
   * Updates performance metrics based on event type
   */
  private updateMetrics(eventName: string, data: any): void {
    if (Database.METRICS_EVENTS.includes(eventName as any)) {
      this._updateMonitorDbMetrics();
    }
  }

  /**
   * Records events in the monitor if available
   */
  private recordEventInMonitor(eventName: string, data: any): void {
    if (!this.monitor) return;

    if (['vector:add', 'vector:delete', 'partition:created'].includes(eventName)) {
      this.monitor.recordEvent(eventName, data);
    }
    if (eventName === 'partition:error') {
      this.monitor.recordError('partition_error', data);
    }
  }

  /**
   * Registers database operation event listeners
   */
  private registerDatabaseEventListeners(
    dbEmitter: TypedEventEmitter<PartitionedDBEventData>,
    handler: <E extends keyof PartitionedDBEventData>(
      eventName: E,
      data: PartitionedDBEventData[E]
    ) => void
  ): void {
    const events: (keyof PartitionedDBEventData)[] = [
      'vector:add',
      'vector:delete',
      'vectors:bulkAdd',
      'partition:created',
      'partition:loaded',
      'partition:unloaded',
      'partition:activated',
      'partition:error',
      'config:saved',
      'db:saved',
      'db:loaded',
    ];

    events.forEach(event => {
      dbEmitter.on(event, (data) => handler(event, data));
    });
  }

  /**
   * Registers indexing-related event listeners
   */
  private registerIndexingEventListeners(dbEmitter: TypedEventEmitter<PartitionedDBEventData>): void {
    // Forward indexing events
    dbEmitter.on('partition:indexProgress', (data) => this.emit('index:progress', data));
    dbEmitter.on('partition:indexed', (data) => this.emit('partition:indexed', data));
    dbEmitter.on('partition:indexLoaded', (data) => this.emit('partition:indexLoaded', data));
    dbEmitter.on('partition:indexSaved', (data) => this.emit('partition:indexSaved', data));

    // Handle close event from underlying DB
    dbEmitter.on('db:close', () => {
      console.log('Underlying PartitionedVectorDB reported close. Closing high-level DB.');
      this.close().catch((err) => console.error('Error during close triggered by underlying DB:', err));
    });
  }

  /** Sets up listeners for UnifiedSearch events. */
  private setupUnifiedSearchListeners(): void {
    if (!this.unifiedSearch) return;

    // Forward search start events
    this.unifiedSearch.on('search:start', (data) => {
      this.emit('search:start', data);
    });

    // Handle search completion events
    this.unifiedSearch.on('search:complete', (data) => {
      this.updateSearchMetrics(data);
      this.emit('search:complete', data);
    });
  }

  /**
   * Updates search performance metrics
   */
  private updateSearchMetrics(data: any): void {
    if (this.monitor) {
      this.monitor.recordSearch({
        duration: data.totalTime,
        method: data.dbMethodUsed,
        results: data.resultCount,
        cacheUsed: data.cacheUsed,
      });
    }

    // Update aggregate metrics
    this.metrics.totalSearchTime += data.totalTime;
    this.metrics.queries++;
    this.metrics.avgSearchTime = this.metrics.totalSearchTime / this.metrics.queries;

    // Track query times for analysis (keep last 100)
    this.metrics.queryTimes.push(data.totalTime);
    if (this.metrics.queryTimes.length > 100) {
      this.metrics.queryTimes.shift();
    }
  }

  /** Pushes current DB stats to the monitor asynchronously. */
  private async _updateMonitorDbMetrics(): Promise<void> {
    if (!this.monitor || !this.options.monitoring.enableDatabaseMetrics) return;

    // Prevent updates if DB isn't ready or is closing
    if (!this.vectorDB || !this.isReady || this.isClosed) return;

    try {
      // Use peek to avoid potentially high cost of getStats if called frequently
      const stats = await this.vectorDB.getStats();
      this.monitor.updateDatabaseMetrics({
        vectorCount: stats.vectors.totalInMemory,
        memoryUsageBytes: stats.memory.estimatedUsageBytes,
        partitionCount: stats.partitions.loadedCount,
        // Add more metrics from PartitionedDBStats if needed
      });
    } catch (error) {
      console.warn('Failed to update monitor DB metrics:', error);
    }
  }

  /** Sets up the auto-save interval timer. */
  private _setupAutoSave(): void {
    if (this.autoSaveTimer) clearInterval(this.autoSaveTimer);
    console.log(`Setting up auto-save via vectorDB.save() every ${this.options.persistence.saveIntervalMs}ms`);
    this.autoSaveTimer = setInterval(async () => {
      if (this.isClosed || !this.isReady) return;
      console.log('Auto-save triggered...');
      try {
        // Delegate saving entirely to PartitionedVectorDB
        await this.vectorDB.save();
        console.log('Auto-save completed successfully.');
      } catch (error: any) {
        console.error('Auto-save failed:', error);
        this.emit('error', {
          message: `Auto-save failed: ${error.message}`,
          error: error,
          context: 'AutoSave',
        });
      }
    }, this.options.persistence.saveIntervalMs);
    this.autoSaveTimer.unref(); // Allow process to exit if this is the only timer
  }

  /** Generates a cache key for search results. */
  private _getCacheKey(query: Vector, k: number, options: UnifiedSearchOptions): string {
    const vectorHash = this._hashVector(query);
    // Include all options that influence the search result
    const optionsKey = JSON.stringify({
      k,
      filter: options.filter ? 'present' : 'absent', // Simplify filter representation
      distanceMetric: options.distanceMetric ?? 'default', // Use default marker
      efSearch: options.efSearch,
      useHNSW: options.useHNSW,
      rerank: options.rerank,
      rerankingMethod: options.rerankingMethod,
      partitionIds: options.partitionIds?.sort(), // Sort for consistency
      searchMethod: options.searchMethod,
    });
    return `${vectorHash}::${optionsKey}`;
  }

  /** Simple, fast vector hash (not cryptographically secure). */
  private _hashVector(vector: Vector): string {
    let hash = 0;
    // Sample fewer points for potentially faster hashing
    const samples = Math.min(vector.length, 16);
    const step = Math.max(1, Math.floor(vector.length / samples));
    for (let i = 0; i < vector.length; i += step) {
      // Combine value and index for slightly better distribution
      const val = Math.round((vector[i] || 0) * 1000); // Use 1000x scaling
      hash = (hash << 5) - hash + val + i;
      hash |= 0; // Convert to 32bit integer
    }
    return hash.toString(36); // Use base 36 for shorter string
  }

  // --- Public API Methods ---

  /** Checks if the database is initialized and ready for operations. */
  IsReady(): boolean {
    console.log('[Database] Checking if database is ready...');
    console.log('[Database] isReady:', this.isReady);
    console.log('[Database] isClosed:', this.isClosed);
    return this.isReady && !this.isClosed;
  }

  /** Throws an error if the database is not ready or closed. */
  private async _assertReady(operation: string = 'Operation'): Promise<void> {
    console.log('[Database] Asserting readiness for operation:', operation);
    if (this.isClosed) throw new Error(`Database is closed. Cannot perform ${operation}.`);
    if (!this.isReady) {
      console.log(`[Database] Waiting for database readiness before performing ${operation}...`);
      await this.initializationPromise;
      // Re-check after waiting
      if (this.isClosed) throw new Error(`Database was closed during initialization wait. Cannot perform ${operation}.`);
      if (!this.isReady) throw new Error(`Database initialization failed. Cannot perform ${operation}.`);
    }
  }

  /**
   * Adds a vector to the appropriate partition.
   * @returns An object containing the partitionId and the vectorId.
   */
  async addVector(id: number | string | undefined, vector: Vector, metadata?: Record<string, any>): Promise<{ partitionId: string; vectorId: number | string }> {
    console.log(`Adding vector with ID ${id ?? 'auto'}...`);

    await this._assertReady('addVector');
    this.timer.start('addVector');
    try {
      const result = await this.vectorDB.addVector(id, vector, metadata);
      const duration = this.timer.stop('addVector');
      this.metrics.totalAddTime += duration.total;

      console.log(`Vector added successfully with ID ${result.vectorId}.`);
      return result;
    } catch (error: any) {
      this.timer.stop('addVector');
      console.error(`Error in addVector (ID: ${id ?? 'auto'}):`, error);
      this.emit('error', {
        message: `Add vector failed: ${error.message}`,
        error: error,
        context: 'addVector',
      });
      throw error; // Re-throw original error
    }
  }

  /**
   * Bulk adds vectors, handling partitioning automatically.
   */
  async bulkAdd(vectors: VectorData[]): Promise<{ count: number; partitionIds: string[] }> {
    await this._assertReady('bulkAdd');
    this.timer.start('bulkAdd');
    try {
      const result = await this.vectorDB.bulkAdd(vectors);
      this.timer.stop('bulkAdd');
      return result;
    } catch (error: unknown) {
      this.timer.stop('bulkAdd');
      console.error(`Error during bulk add:`, error);
      this.emit('error', {
        message: `Bulk add failed: ${(error as Error).message}`,
        error: error,
        context: 'bulkAdd',
      });
      throw error;
    }
  }

  /** Deletes a vector from the partition it resides in. */
  async deleteVector(id: number | string): Promise<boolean> {
    await this._assertReady('deleteVector');
    this.timer.start('deleteVector');
    try {
      const deleted = await this.vectorDB.deleteVector(id);
      this.timer.stop('deleteVector');
      return deleted;
    } catch (error: any) {
      this.timer.stop('deleteVector');
      console.error(`Error deleting vector ${id}:`, error);
      this.emit('error', {
        message: `Delete vector failed for ${id}: ${error.message}`,
        error: error,
        context: 'deleteVector',
      });
      throw error;
    }
  }

  /** Checks if a vector exists in any loaded partition. */
  async hasVector(id: number | string): Promise<boolean> {
    await this._assertReady('hasVector');
    const result = await this.vectorDB.getVector(id);
    return result !== null;
  }

  /** Retrieves a vector by searching loaded partitions. */
  async getVector(id: number | string): Promise<{ partitionId: string; vector: Float32Array } | null> {
    await this._assertReady('getVector');
    // Directly delegate, no extra timing needed unless specific performance analysis required
    return this.vectorDB.getVector(id);
  }

  /** Adds or updates metadata for a vector. Requires finding the vector first. */
  async addMetadata(id: number | string, metadata: Record<string, any>): Promise<boolean> {
    await this._assertReady('addMetadata');
    this.timer.start('addMetadata');
    try {
      // Find the partition containing the vector
      const vectorInfo = await this.vectorDB.getVector(id);
      if (!vectorInfo) {
        console.warn(`addMetadata: Vector ${id} not found.`);
        this.timer.stop('addMetadata');
        return false;
      }
      // Get the specific partition DB instance
      const partition = await this.vectorDB.getPartition(vectorInfo.partitionId);
      if (!partition) {
        console.warn(`addMetadata: Partition ${vectorInfo.partitionId} for vector ${id} could not be loaded.`);
        this.timer.stop('addMetadata');
        return false;
      }
      // Add metadata using the partition's method
      partition.addMetadata(id, metadata); // Assuming this is synchronous or handled internally
      this.searchCache.clear(); // Clear cache as metadata might affect filtering
      this.timer.stop('addMetadata');
      return true;
    } catch (error: any) {
      this.timer.stop('addMetadata');
      console.error(`Error adding/updating metadata for vector ${id}:`, error);
      this.emit('error', {
        message: `Add/Update metadata failed for ${id}: ${error.message}`,
        error: error,
        context: 'addMetadata',
      });
      throw error;
    }
  }

  /** Updates metadata using a callback or merging. Requires finding the vector first. */
  async updateMetadata(id: number | string, metadataUpdate: Record<string, any> | ((existing: Record<string, any> | null) => Record<string, any>)): Promise<boolean> {
    await this._assertReady('updateMetadata');
    this.timer.start('updateMetadata');
    try {
      const vectorInfo = await this.vectorDB.getVector(id);
      if (!vectorInfo) {
        console.warn(`updateMetadata: Vector ${id} not found.`);
        this.timer.stop('updateMetadata');
        return false;
      }
      const partition = await this.vectorDB.getPartition(vectorInfo.partitionId);
      if (!partition) {
        console.warn(`updateMetadata: Partition ${vectorInfo.partitionId} for vector ${id} could not be loaded.`);
        this.timer.stop('updateMetadata');
        return false;
      }
      partition.updateMetadata(id, metadataUpdate);
      this.searchCache.clear();
      this.timer.stop('updateMetadata');
      return true;
    } catch (error: any) {
      this.timer.stop('updateMetadata');
      console.error(`Error updating metadata for vector ${id}:`, error);
      this.emit('error', {
        message: `Update metadata failed for ${id}: ${error.message}`,
        error: error,
        context: 'updateMetadata',
      });
      throw error;
    }
  }

  /** Retrieves metadata by searching loaded partitions. */
  async getMetadata(id: number | string): Promise<{ partitionId: string; metadata: Record<string, any> } | null> {
    await this._assertReady('getMetadata');
    return this.vectorDB.getMetadata(id);
  }

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
  async getMetadataWithField(
    criteria: string | string[] | Record<string, any>,
    values?: any | any[],
    option?: {
      limit: number;
    }
  ): Promise<Array<{ partitionId: string; vectorId: number | string; metadata: Record<string, any> }>> {
    await this._assertReady('getMetadataWithField');
    this.timer.start('getMetadataWithField');
    try {
      const results = await this.vectorDB.getMetadataWithFieldAcrossPartitions(criteria, values, option);
      this.timer.stop('getMetadataWithField');
      return results;
    } catch (error: any) {
      this.timer.stop('getMetadataWithField');
      console.error(`Error in getMetadataWithField:`, error);
      this.emit('error', {
        message: `Get metadata with field failed: ${error.message}`,
        error: error,
        context: 'getMetadataWithField',
      });
      throw error;
    }
  }

  /**
   * Performs a nearest neighbor search using the UnifiedSearch engine.
   * Handles caching and concurrency limits.
   */
  async findNearest(query: Vector, k?: number, options: UnifiedSearchOptions = {}): Promise<SearchResult[]> {
    console.log(`[Database] Searching for nearest vectors to query...`);

    await this._assertReady('findNearest');
    const operationTimer = createTimer();
    operationTimer.start('findNearest_total');

    const effectiveK = k ?? options.k ?? 10;
    const searchOptions = { ...options, k: effectiveK };

    // --- Cache Check ---
    let cacheKey: string | null = null;
    if (!searchOptions.skipCache) {
      cacheKey = this._getCacheKey(query, effectiveK, searchOptions);
      const cachedResults = this.searchCache.get(cacheKey);
      if (cachedResults) {
        this.metrics.cacheHits++;
        this.monitor?.recordCacheHit();
        operationTimer.stop('findNearest_total');
        // UnifiedSearch event won't fire for cache hit, emit simple event here if needed
        this.emit('search:cacheHit', { options: searchOptions, k: effectiveK });
        return [...cachedResults]; // Return copy
      }
      this.metrics.cacheMisses++;
      this.monitor?.recordCacheMiss();
    }

    // --- Concurrency Management ---
    if (this.activeSearchPromises.size >= this.options.maxConcurrentSearches) {
      this.timer.start('findNearest_wait');
      console.log(`Search concurrency limit (${this.options.maxConcurrentSearches}) reached. Waiting...`);
      try {
        await Promise.race(this.activeSearchPromises); // Wait for any ongoing search to finish
      } catch {
        /* Ignore errors from waiting */
      }
      this.timer.stop('findNearest_wait');
      // Recheck limit after waiting (another search might have started)
      if (this.activeSearchPromises.size >= this.options.maxConcurrentSearches) {
        // If still full after waiting, throw or queue? Let's throw for now.
        operationTimer.stop('findNearest_total');
        throw new Error(`Search concurrency limit (${this.options.maxConcurrentSearches}) exceeded after wait.`);
      }
    }

    // --- Execute Search via UnifiedSearch ---
    const searchPromise = this.unifiedSearch.search(query, searchOptions);
    this.activeSearchPromises.add(searchPromise);

    try {
      const results = await searchPromise;
      // UnifiedSearch emits search:complete which updates metrics

      // Cache results if cache was checked and not skipped
      if (cacheKey && results.length > 0) {
        this.searchCache.set(cacheKey, [...results]); // Store copy
      }
      operationTimer.stop('findNearest_total');
      return results;
    } catch (error: any) {
      operationTimer.stop('findNearest_total');
      // Error event emitted by UnifiedSearch listener
      console.error('Error during findNearest execution:', error);
      throw error; // Re-throw
    } finally {
      this.activeSearchPromises.delete(searchPromise); // Clean up promise tracking
    }
  }

  /** Alias for findNearest */
  async search(query: Vector, options: UnifiedSearchOptions = {}): Promise<SearchResult[]> {
    return this.findNearest(query, options.k, options);
  }

  /**
   * Saves the current state of the database (delegated to PartitionedVectorDB).
   * This includes partition configurations, loaded partition data, and loaded HNSW indices.
   */
  async save(): Promise<void> {
    console.log('[Database] Manual save requested...');

    await this._assertReady('save');
    console.log('[Database] Manual save requested. Delegating to PartitionedVectorDB...');
    this.timer.start('save_database');
    try {
      await this.vectorDB.save(); // Delegate the comprehensive save
      this.timer.stop('save_database');
      console.log('Database save completed successfully.');
      // db:saved event is emitted by PartitionedVectorDB listener
    } catch (error: any) {
      this.timer.stop('save_database');
      console.error('Manual save failed:', error);
      this.emit('error', {
        message: `Save failed: ${error.message}`,
        error: error,
        context: 'save',
      });
      throw error;
    }
  }

  /**
   * Builds HNSW indexes. Delegates to PartitionedVectorDB.
   */
  async buildIndexes(partitionId?: string, options?: BuildIndexHNSWOptions): Promise<void> {
    console.log(`[Database] Building HNSW index for partition ${partitionId ?? 'all loaded'}...`);
    if (options?.force !== true) await this._assertReady('buildIndexes');
    console.log(`[Database] Requesting index build for ${partitionId ?? 'all loaded partitions'}...`);
    this.timer.start('buildIndexes');
    try {
      const buildOptions = {
        ...this.options.indexing.hnsw, // Global defaults from Database options
        ...options, // Specific options for this call
      };
      await this.vectorDB.buildIndexHNSW(partitionId, buildOptions);
      const duration = this.timer.stop('buildIndexes');
      console.log(`Index build process finished in ${duration.total.toFixed(2)}ms for ${partitionId ?? 'relevant partitions'}.`);
      // index:complete event emitted by listener
    } catch (error: any) {
      const duration = this.timer.stop('buildIndexes');
      console.error(`Index build failed after ${duration.total.toFixed(2)}ms for ${partitionId ?? 'partitions'}:`, error);
      this.emit('error', {
        message: `Index build failed: ${error.message}`,
        error: error,
        context: 'buildIndexes',
      });
      // index:error emitted by listener
      throw error;
    }
  }

  /** Closes the database, saves state, stops background tasks, and releases resources. */
  async close(): Promise<void> {
    if (this.isClosed) {
      console.log('Database already closed.');
      return;
    }
    console.log('Closing database...');
    this.isClosed = true; // Mark as closing immediately
    this.isReady = false;

    // Stop background tasks
    if (this.autoSaveTimer) clearInterval(this.autoSaveTimer);
    this.autoSaveTimer = null;

    // Stop monitoring
    this.monitor?.stop();

    // Wait for active searches? (Optional, might delay closing)
    // console.log(`Waiting for ${this.activeSearchPromises.size} active searches to complete...`);
    // await Promise.allSettled(this.activeSearchPromises);
    this.activeSearchPromises.clear();

    // Close UnifiedSearch
    try {
      this.unifiedSearch?.close(); // Assuming UnifiedSearch has a close method
    } catch (e) {
      console.error('Error closing UnifiedSearch:', e);
    }

    // Close PartitionedVectorDB (this should handle saving its state)
    if (this.vectorDB && typeof this.vectorDB.close === 'function') {
      try {
        console.log('Closing PartitionedVectorDB (will trigger final save)...');
        await this.vectorDB.close();
      } catch (err: any) {
        console.error('Error closing PartitionedVectorDB:', err.message);
        // Continue closing process
      }
    }

    // Clear search cache
    this.searchCache.clear();

    console.log('Database closed successfully.');
    this.emit('close', undefined);
    this.removeAllListeners(); // Clean up all listeners on this instance
  }

  // --- Getters and Utility Methods ---

  /** Gets combined statistics about the database state, components, and system. */
  async getStats(): Promise<DatabaseStats> {
    // No readiness check here, return best effort stats even if initializing/closing
    let dbStats: PartitionedDBStats | null = null;
    let searchStats: UnifiedSearchPartitionedStats | null = null;

    if (this.vectorDB && typeof this.vectorDB.getStats === 'function') {
      try {
        dbStats = await this.vectorDB.getStats();
      } catch (e) {
        console.warn('Failed to get PartitionedDB stats:', e);
      }
    }
    if (this.unifiedSearch && typeof this.unifiedSearch.getStats === 'function') {
      try {
        searchStats = await this.unifiedSearch.getStats();
      } catch (e) {
        console.warn('Failed to get UnifiedSearch stats:', e);
      }
    }

    const cacheHitRate = this.metrics.cacheHits + this.metrics.cacheMisses > 0 ? this.metrics.cacheHits / (this.metrics.cacheHits + this.metrics.cacheMisses) : 0;

    return {
      state: {
        isClosed: this.isClosed,
        isReady: this.isReady,
        status: this.isClosed ? 'closed' : this.isReady ? 'ready' : 'initializing',
      },
      database: dbStats, // Nullable if failed
      search: searchStats?.search ?? null, // Access nested search stats, nullable
      searchCache: {
        size: this.searchCache.size,
        capacity: this.options.cacheSize,
        hits: this.metrics.cacheHits,
        misses: this.metrics.cacheMisses,
        hitRate: parseFloat((cacheHitRate * 100).toFixed(2)),
      },
      performance: {
        queries: this.metrics.queries,
        avgSearchTimeMs: this.metrics.queries > 0 ? parseFloat((this.metrics.totalSearchTime / this.metrics.queries).toFixed(2)) : 0,
        cacheHitRate: parseFloat((cacheHitRate * 100).toFixed(2)),
        concurrentSearches: this.activeSearchPromises.size,
        // Include timer stats if needed, e.g., from this.timer.getSummary()
      },
      system: this.monitor?.getSystemMetrics(),
      memoryUsage: process.memoryUsage(),
      options: {
        vectorSize: this.options.vectorSize || 0,
        partitionsDir: this.options.partitioning.partitionsDir,
        partitionCapacity: this.options.partitioning.partitionCapacity,
        maxActivePartitions: this.options.partitioning.maxActivePartitions,
        cacheSize: this.options.cacheSize,
        maxConcurrentSearches: this.options.maxConcurrentSearches || 0,
        autoSaveIntervalMs: this.options.persistence.saveIntervalMs || 0,
        monitoringEnabled: this.options.monitoring.enable || false,
      },
    };
  }

  /** Gets the underlying PartitionedVectorDB instance. Use with caution. */
  getVectorDB(): PartitionedVectorDBInterface {
    if (!this.isReady && !this.isClosed) console.warn('Accessing VectorDB instance before Database is fully ready.');
    if (this.isClosed) throw new Error('Cannot access VectorDB: Database is closed.');
    return this.vectorDB;
  }

  /** Gets the UnifiedSearch instance. */
  getUnifiedSearch(): UnifiedSearch {
    if (!this.isReady && !this.isClosed) console.warn('Accessing UnifiedSearch instance before Database is fully ready.');
    if (this.isClosed) throw new Error('Cannot access UnifiedSearch: Database is closed.');
    return this.unifiedSearch;
  }

  /** Gets the total count of vectors across all configured partitions. */
  async getTotalVectorCount(): Promise<number> {
    await this._assertReady('getTotalVectorCount');
    const stats = await this.vectorDB.getStats();
    return stats?.vectors?.totalConfigured ?? 0;
  }

  /** Gets the count of vectors currently loaded in memory. */
  async getInMemoryVectorCount(): Promise<number> {
    await this._assertReady('getInMemoryVectorCount');
    const stats = await this.vectorDB.getStats();
    return stats?.vectors?.totalInMemory ?? 0;
  }

  /** Gets the number of partitions currently loaded in memory. */
  async getLoadedPartitionCount(): Promise<number> {
    await this._assertReady('getLoadedPartitionCount');
    const stats = await this.vectorDB.getStats();
    return stats?.partitions?.loadedCount ?? 0;
  }

  /** Gets the IDs of the partitions currently loaded in memory. */
  async getLoadedPartitionIds(): Promise<string[]> {
    await this._assertReady('getLoadedPartitionIds');
    const stats = await this.vectorDB.getStats();
    return stats?.partitions?.loadedIds ?? [];
  }

  /**
   * Extract relationships between vectors based on distance threshold across all loaded partitions.
   *
   * @param threshold - The maximum distance between vectors to consider them related
   * @param options - Options including distance metric, partition filtering, and metadata inclusion
   * @returns An array of relationships with vectorIds, partitionIds, optional metadata, and distances
   */
  async extractRelationships(
    threshold: number,
    options: {
      metric?: DistanceMetric;
      partitionIds?: string[];
      includeMetadata?: boolean;
    } = {}
  ): Promise<
    Array<{
      vector1: { id: number | string; partitionId: string; metadata?: Record<string, any> };
      vector2: { id: number | string; partitionId: string; metadata?: Record<string, any> };
      distance: number;
    }>
  > {
    await this._assertReady('extractRelationships');
    this.timer.start('extractRelationships');

    try {
      console.log(`[Database] Extracting relationships with threshold ${threshold}...`);

      const relationships = await this.vectorDB.extractRelationships(threshold, options);

      const duration = this.timer.stop('extractRelationships');
      console.log(`[Database] Extracted ${relationships.length} relationships in ${duration.total.toFixed(2)}ms`);

      return relationships;
    } catch (error: any) {
      this.timer.stop('extractRelationships');
      console.error(`[Database] Error extracting relationships:`, error);
      this.emit('error', {
        message: `Extract relationships failed: ${error.message}`,
        error,
        context: 'extractRelationships',
      });
      throw error;
    }
  }

  /**
   * Extract communities of related vectors based on distance threshold across all loaded partitions.
   * A community is a group of vectors where each vector is related to at least one other vector in the group.
   *
   * @param threshold - The maximum distance between vectors to consider them related
   * @param options - Options including distance metric and partition filtering
   * @returns An array of communities, where each community is an array of related vector information
   */
  async extractCommunities(
    threshold: number,
    options: {
      metric?: DistanceMetric;
      partitionIds?: string[];
      includeMetadata?: boolean;
    } = {}
  ): Promise<
    Array<
      Array<{
        id: number | string;
        partitionId: string;
        metadata?: Record<string, any>;
      }>
    >
  > {
    await this._assertReady('extractCommunities');
    this.timer.start('extractCommunities');

    try {
      // Determine which partitions to process
      let partitionIds = options.partitionIds;
      if (!partitionIds || partitionIds.length === 0) {
        const stats = await this.vectorDB.getStats();
        partitionIds = stats.partitions.loadedIds;
      }

      console.log(`[Database] Extracting vector communities across ${partitionIds.length} partitions with threshold ${threshold}...`);

      // Delegate to the vectorDB implementation
      const communities = await this.vectorDB.extractCommunities(threshold, {
        metric: options.metric,
        partitionIds,
        includeMetadata: options.includeMetadata ?? true,
      });

      const duration = this.timer.stop('extractCommunities');
      console.log(`[Database] Extracted ${communities.length} communities with ${communities.reduce((sum, c) => sum + c.length, 0)} total vectors in ${duration.total.toFixed(2)}ms`);

      return communities;
    } catch (error: any) {
      this.timer.stop('extractCommunities');
      console.error(`[Database] Error extracting communities:`, error);
      this.emit('error', {
        message: `Extract communities failed: ${error.message}`,
        error,
        context: 'extractCommunities',
      });
      throw error;
    }
  }
}
