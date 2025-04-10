import { Database } from 'database/database';
import { PartitionedVectorDB } from 'vector/partitioned_vector_db';
import express, { Request, Response, NextFunction, Express } from 'express';
import { Timer } from 'utils/profiling';

/**
 * Vector data with ID for bulk operations
 */
export interface VectorData {
  id: number | string;
  vector: Vector;
  metadata?: Record<string, any>;
}

/**
 * Reranking options interface
 */
export interface RerankingOptions {
  method?: RerankingMethod;
  k?: number;
  metadata?: Map<string | number, any>;
  vectors?: Map<string | number, Vector>;
  weights?: Record<string, number>;
  [key: string]: any;
}

/**
 * Common search options interface
 */
export interface SearchOptions extends UnifiedSearchOptions {
  // Backward compatibility
  limit?: number; // Alias for k
  offset?: number; // For pagination
  stopEarly?: boolean; // Alias for earlyStoppingThreshold
}

/**
 * Hybrid search options interface
 */
export interface HybridSearchOptions extends UnifiedSearchOptions {
  // Additional hybrid-specific options
  buildIndexes?: boolean;
  methods?: string[];
  useParallelExecution?: boolean;
  cachingEnabled?: boolean;
  cacheSize?: number;
  useClustered?: boolean; // Dùng findNearest (clustered search) nếu true
  partitionIds?: string[]; // Chỉ tìm trên các partition này
  buildIndexOptions?: BuildIndexHNSWOptions; // Tùy chọn khi build index
}

/**
 * Import/Export options interface
 */
export interface ImportExportOptions {
  format?: 'json' | 'binary' | 'csv';
  includeMetadata?: boolean;
  compression?: boolean;
  csvSeparator?: string;
  precision?: number;
}

/**
 * Backup options interface
 */
export interface BackupOptions {
  compress?: boolean;
  includeIndexes?: boolean;
  destination?: string;
  excludeMetadata?: boolean;
}

/**
 * Worker message interface for multithreading
 */
export interface WorkerMessage {
  type: string;
  data: any;
}

/**
 * Worker result interface for multithreading
 */
export interface WorkerResult {
  error?: string;
  data?: any;
}

/**
 * Streaming result interface for large operations
 */
export interface StreamingResult<T> {
  complete: boolean;
  progress: number;
  results: T[];
  error?: string;
}

/**
 * Batch operation interface
 */
export interface BatchOperation {
  id: string;
  type: 'add' | 'delete' | 'update';
  data?: any;
}

/**
 * Search result interface
 */
export interface SearchResult {
  id: IDVector;
  dist: number;
  [key: string]: any; // For additional properties like metadata
}

/**
 * Database statistics
 */
export interface DBStats {
  vectorCount: number;
  vectorSize: number;
  defaultVectorSize: number;
  metadataCount: number;
  memoryUsage: number;
  dimensions: {
    counts: Record<number, number>;
    unique: number;
  };
  clusters: {
    count: number;
    avgSize: number;
    dimensions: Record<number, number>;
    distribution: Array<{
      id: number;
      size: number;
      centroidNorm: number;
      dimension: number;
      members: {
        id: IDVector;
      }[];
    }>;
  };
}
export type IDVector = number | string;

/**
 * Vector data for saving to file
 */
export interface VectorDataForSave {
  id: IDVector;
  vector: number[];
  metadata?: Record<string, any>;
}

// Interface voor de data structuren van de events
export interface VectorDBEventData {
  'vector:add': { id: IDVector; dimensions: number };
  'vectors:bulkAdd': { count: number; ids: IDVector[] };
  'vector:delete': { id: number | string };
  'metadata:add': { id: number | string; metadata: Record<string, any> };
  'metadata:update': { id: number | string; metadata: Record<string, any> };
  'db:save': { path: string; count: number };
  'db:load': { path: string; count: number };
  'db:close': {}; // Geen data nodig voor close event
  'cluster:create': { clusterId: number; vectorId: number | string };
  'cluster:delete': { clusterId: number };
  'db:error': { operation: string; error: Error | unknown };
  'kmeans:complete': { k: number; iterations: number };
  'kmeans:error': { error: Error | unknown };
  'kmeans:start': { k: number; iterations: number };
  'vector:update': { id: number | string; dimensions: number };
}
export type BulkAddResult = {};

// Optioneel: Maak een strikter getypte EventEmitter klasse
// Dit zorgt ervoor dat je alleen gedefinieerde events kunt emitten/listenen
// en dat de argumenten overeenkomen.

export interface TypedEventEmitter<Events extends Record<string, any>> {
  on<E extends keyof Events>(event: E, listener: (payload: Events[E]) => void): this;
  once<E extends keyof Events>(event: E, listener: (payload: Events[E]) => void): this;
  off<E extends keyof Events>(event: E, listener: (payload: Events[E]) => void): this;
  emit<E extends keyof Events>(event: E, payload: Events[E]): boolean;
  listenerCount<E extends keyof Events>(event: E): number;
  listeners<E extends keyof Events>(event: E): ((payload: Events[E]) => void)[];
  removeAllListeners<E extends keyof Events>(event?: E): this;
}

// Interface for index build/load/save progress/status
export interface IndexProgressEvent {
  type: IndexType;
  progress: number; // 0.0 to 1.0
  dimension?: number; // For dimension-aware indexes like PQ
  dimensionAware?: boolean; // For HNSW/LSH
}

export interface IndexBuiltEvent {
  type: IndexType;
  timeMs: number;
  stats: any; // Stats specific to the index type
}

export interface IndexActionEvent {
  timeMs: number;
  types: IndexType[];
}

export interface IndexErrorEvent {
  type?: IndexType; // Optional: General error might not have a type
  error: any; // The actual error object or message
  timeMs?: number; // Optional: Duration if error occurred during timed operation
}

// Combine IndexManager specific events with VectorDB events if needed,
// or define a separate event map for IndexManager.
export interface IndexManagerEventData {
  'indexes:building': { types: IndexType[] };
  'indexes:built': IndexActionEvent; // Overall build completion
  'indexes:error': IndexErrorEvent; // Error during overall build
  'index:built': IndexBuiltEvent; // Specific index completion
  'index:error': IndexErrorEvent; // Error for a specific index
  'index:progress': IndexProgressEvent;
  'indexes:saving': void;
  'indexes:saved': { timeMs: number };
  'indexes:savingError': { error: any };
  'indexes:loading': void;
  'indexes:loaded': { timeMs: number; loadedTypes: IndexType[] };
  'indexes:loadingError': { error: any; timeMs?: number };
  reset: void; // Event when indexes are reset
}

// Index options interface (cleaned up)
export interface IndexOptions {
  indexPath?: string; // Directory to store index files
  buildOnStart?: boolean; // Build indexes when IndexManager is created/loaded
  autoSave?: boolean; // Save indexes automatically after building
  autoBuildThreshold?: number; // Rebuild indexes if DB size changes by this amount
  // Explicitly enable/disable specific index types
  indexes?: {
    hnsw?: boolean;
    lsh?: boolean;
    pq?: boolean;
    flat?: boolean; // Flat is usually implicit, but can be listed
  };
  // Options specific to each index type
  hnswOptions?: {
    M?: number;
    efConstruction?: number;
    efSearch?: number;
    // dimensionAware is handled internally based on DB capabilities
  };
  lshOptions?: {
    numberOfHashes?: number;
    numberOfBuckets?: number;
    // allowMismatchedDimensions handled internally
  };
  pqOptions?: {
    subvectorSize?: number;
    numClusters?: number;
    // dynamicDimensions handled internally
  };
}

// Statistics structure (remains similar)
export interface IndexStats {
  indexTypes: IndexType[];
  isBuilding: boolean;
  lastBuildTimeMs?: number;
  lastSaveTimeMs?: number;
  lastLoadTimeMs?: number;
  dbVectorsAtLastBuild: number; // Track DB size when indexes were last built
  // Specific index stats (consider more detailed types per index)
  indexes: Partial<Record<IndexType, any>>; // Use Partial as not all indexes might exist or have stats
}

export interface TimerData {
  start: [number, number];
  splits: { label: string | null; elapsed: number }[];
  lastDuration?: number; // Store the duration of the last stop
}

export interface TimerResult {
  total: number;
  splits: { label: string | null; elapsed: number }[];
}

type OptionCheck<T> =
  | {
      enable: true;
      options?: T;
    }
  | {
      enable: false;
    };

// Interface for performance metrics (remains the same)
export interface PerformanceMetrics {
  queries: number;
  totalSearchTime: number; // milliseconds
  avgSearchTime: number; // milliseconds
  totalAddTime: number; // milliseconds
  avgAddTime: number; // milliseconds
  cacheHits: number;
  cacheMisses: number;
  queryTimes: number[]; // Store recent query times
}

/**
 * Options for the monitoring system
 */
export interface MonitoringOptions {
  interval?: number;
  historySize?: number;
  logToConsole?: boolean;
  enableSystemMetrics?: boolean;
  enableSearchMetrics?: boolean;
  enableDatabaseMetrics?: boolean; // Add option for DB metrics
  enableCacheMetrics?: boolean; // Add option for cache metrics
}

/**
 * Search event data structure
 */
export interface SearchEvent {
  timestamp: number;
  duration: number;
  method: string;
}

/**
 * Database metrics data structure
 */
export interface DatabaseMetrics {
  vectorCount: number;
  memoryUsage: number;
}

/**
 * System metrics data structure
 */
export interface SystemMetrics {
  cpu: number;
  memory: number;
  loadAvg: number;
}

/**
 * Search metrics data structure
 */
export interface SearchMetrics {
  queryCount: number;
  averageResponseTime: number;
  p95ResponseTime: number;
  queriesPerMinute: number;
  methodUsage: Record<string, number>;
  responseTimes: number[];
}

/**
 * HNSW node interface
 */
export interface HNSWNode {
  id: number | string;
  connections: Map<number, Set<number | string>>;
  dimension?: number; // Store vector dimension
}

/**
 * HNSW options interface
 */
export interface HNSWOptions {
  M?: number; // Maximum number of connections per node
  efConstruction?: number; // Size of the dynamic candidate list during construction
  efSearch?: number; // Size of the dynamic candidate list during search
  distanceFunc?: (a: Vector, b: Vector) => number;
  maxLevel?: number; // Maximum level in the graph
  levelProbability?: number; // Probability of assigning a higher level
  entryPointId?: number | string; // Custom entry point ID
  dimensionAware?: boolean; // Whether to handle vectors of different dimensions
  nodes?: HNSWNode[]; // Predefined nodes for loading
}

/**
 * HNSW build index options
 */
export interface BuildIndexHNSWOptions {
  progressCallback?: (progress: number) => void;
  dimensionAware?: boolean;
  force?: boolean;
}

/**
 * HNSW load index options
 */
export interface LoadIndexHNSWOptions {
  dimensionAware?: boolean;
}

// --- Interfaces for Monitoring Data ---

export interface MonitoringOptions {
  interval?: number; // ms
  historySize?: number; // Number of data points to keep
  logToConsole?: boolean;
  enableSystemMetrics?: boolean;
  enableSearchMetrics?: boolean;
  enableDatabaseMetrics?: boolean; // Add option for DB metrics
}

// Structure for a single CPU core's times
export interface CpuTimes {
  user: number;
  nice: number;
  sys: number;
  idle: number;
  irq: number;
}

// Structure for system metrics history
export interface SystemMetricsHistory {
  cpu: number[]; // Overall CPU usage percentage (0-1)
  memory: number[]; // Memory usage percentage (0-1)
  loadAvg1m: number[]; // 1-minute load average
}
// Bovenin het bestand of in types.ts
export interface CurrentSystemMetrics {
  cpuUsage: number | null;
  memoryUsage: number | null;
  loadAvg1m: number | null;
}

// Structure for search metrics history/state
export interface SearchMetricsState {
  queryCount: number; // Total queries since start/reset
  averageResponseTime: number; // Rolling average response time (ms)
  p95ResponseTime: number; // Calculated P95 response time (ms)
  queriesPerMinute: number; // Calculated QPM
  methodUsage: Record<string, number>; // Count per search method/index type
  // Store recent response times for calculations
  recentResponseTimes: number[]; // Limited size buffer for P95/Avg calculation
  responseTimes?: number[]; // Store all response times for analysis
}

// Structure for database metrics history/state
export interface DatabaseMetricsState {
  vectorCount: number;
  // Add other relevant DB metrics if needed (e.g., index size, cluster count)
  memoryUsageBytes: number; // Estimated DB memory usage
  partitionCount: number; // Estimated DB memory usage
}

// Snapshot of all metrics at a point in time
export interface MetricsSnapshot {
  timestamp: string; // ISO timestamp
  uptimeSeconds: number;
  collectionTimeMs: number; // Time taken to collect this snapshot
  metrics: {
    system: {
      cpuUsage: number | null;
      memoryUsage: number | null;
      loadAvg1m: number | null;
    };
    search: {
      queryCount: number;
      averageResponseTime: number;
      p95ResponseTime: number;
      queriesPerMinute: number;
      methodUsage: Record<string, number>;
    };
    database: {
      vectorCount: number;
      memoryUsageBytes: number;
    };
    cache: {
      hits: number;
      misses: number;
      hitRate: number | null;
    };
  };
}

// Event structure for generic events
export interface MonitorEvent {
  type: string;
  timestamp: number; // Milliseconds epoch
  data: any;
}

// Event structure for search events
export interface SearchEventData {
  timestamp: number; // Milliseconds epoch
  duration: number; // Milliseconds
  method: string; // e.g., 'hnsw', 'flat', 'unified'
  // Add k, filter presence, etc. if needed for deeper analysis
}

// Event map for the monitor's TypedEventEmitter
export interface MonitorEvents {
  metrics: MetricsSnapshot;
  event: MonitorEvent;
  error: { message: string; error?: Error; context?: string };
  'cache:hit': void;
  'cache:miss': void;
}

// State voor cache metrics
export interface CacheMetricsState {
  hits: number;
  misses: number;
}

// Snapshot data voor cache
export interface CacheMetricsSnapshotData {
  hits: number;
  misses: number;
  hitRate: number | null;
}
export interface ClusteredVectorDBOptions {
  useCompression?: boolean;
  clusterSize?: number; // Target cluster size
  // Search parameters
  searchProbes?: number; // Number of clusters to check during search (higher = more accurate, slower)
  // Cluster creation parameters
  newClusterThresholdFactor?: number; // e.g., 1.5 -> create new if best cluster > 1.5 * target size
  newClusterDistanceThreshold?: number; // e.g., 0.5 -> create new if distance > threshold
  maxClusters?: number; // Hard limit on the number of clusters
  distanceMetric?: DistanceMetric; // Default metric for clustering and search
  kmeansMaxIterations?: number; // Max iterations for k-means clustering
  runKMeansOnLoad?: boolean; // Run K-Means after loading if cluster state is missing/invalid
}

export interface PartitionConfig {
  id: string;
  name: string;
  dbDirName: string; // Store the directory name relative to partitionsDir
  active: boolean;
  vectorCount: number; // Renamed from size for clarity
  description?: string;
  properties?: Record<string, any>;
  clusterSize?: number; // Specific cluster setting for this partition
  // Add other relevant metadata if needed
}

export interface PartitionedVectorDBOptions {
  partitionsDir?: string;
  partitionCapacity?: number; // Max vectors per partition (approximate)
  autoLoadPartitions?: boolean; // Load active/recent on start
  autoCreatePartitions?: boolean; // Create new partition when active is full
  maxActivePartitions?: number; // Max partitions loaded in memory (LRU)
  vectorSize?: number | null; // Default vector size suggestion
  useCompression?: boolean; // Compression for underlying DBs
  autoLoadHNSW?: boolean; // Compression for underlying DBs
  clusterOptions?: Omit<ClusteredVectorDBOptions, 'clusterSize'>; // Default options for new clusters
  runKMeansOnLoad?: boolean; // Option to run K-Means on partition load if needed
}

// Define events and their payload types
export interface PartitionedDBEventData {
  'db:initialized': {
    partitionCount: number;
    loadedCount: number;
    activeId: string | null;
  };
  'partitions:loaded': { count: number; active: string | null };
  'partition:loaded': {
    id: string;
    name: string;
    vectorCount: number;
    hnswLoaded: boolean;
  };
  'partition:indexLoaded': { id: string; indexType: string; path: string };
  'partition:unloaded': { id: string };
  'partition:created': { id: string; name: string; active: boolean };
  'partition:activated': { id: string };
  'partition:error': {
    id?: string;
    error: Error | unknown;
    operation: string;
    path?: string;
  };
  'vectors:bulkAdd': {
    count: number;
    partitionIds: string[];
  };
  'vector:add': {
    partitionId: string;
    vectorId: number | string;
    metadata?: Record<string, any>;
  };
  'vector:delete': { partitionId: string; vectorId: number | string };
  'db:close': void;
  'db:loaded': {
    partitionCount: number;
    loadedCount: number;
    activeId: string | null;
  };
  'db:saved': {
    partitionsSaved: number;
    indicesSaved: number;
  };
  'config:saved': void;
  'partition:indexSaved': {
    id: string;
    indexType: string;
    path: string;
  };
  'partition:indexProgress': {};
  'partition:progress': { id: string; progress: number };
  'partition:save': { id: string; timeMs: number };
  'partition:load': { id: string; timeMs: number };
  'partition:saveError': { id: string; error: Error };
  'partition:loadError': { id: string; error: Error };
  'partition:reset': { id: string };
  'partition:indexed': { id: string; indexType: string };
  'vector:metadataUpdate': {
    partitionId: string;
    vectorId: number | string;
  };
}

/**
 * Storage manager interface that handles database orchestration,
 * versioning, backup, and recovery for vector databases.
 */
export interface StorageManager {
  /**
   * Initialize the storage manager
   */
  initialize(): Promise<void>;

  /**
   * Get database status and statistics
   */
  getStatus(): Promise<{
    version: string;
    lastBackup?: Date;
    stats: DBStats;
    partitions?: number;
  }>;

  /**
   * Create a backup of the current database state
   */
  createBackup(tag?: string): Promise<string>;

  /**
   * Restore from a specific backup
   */
  restoreFromBackup(backupId: string): Promise<boolean>;

  /**
   * List available backups
   */
  listBackups(): Promise<
    Array<{
      id: string;
      timestamp: Date;
      tag?: string;
      size: number;
    }>
  >;

  /**
   * Add vector(s) to the database
   */
  addVector(vector: Vector, metadata?: Record<string, any>): Promise<number | string>;
  bulkAdd(vectors: VectorData[]): Promise<{ count: number }>;

  /**
   * Search for vectors
   */
  search(query: Vector, k?: number, options?: any): Promise<SearchResult[]>;

  /**
   * Close the storage manager and underlying databases
   */
  close(): Promise<void>;
}

/**
 * Interface for the PartitionedVectorDB class
 */
export interface PartitionedVectorDBInterface {
  // Core methods for vector operations
  addVector(id: number | string | undefined, vector: Vector, metadata?: Record<string, any>): Promise<{ partitionId: string; vectorId: number | string }>;

  bulkAdd(vectors: VectorData[]): Promise<{ count: number; partitionIds: string[] }>;

  getVector(id: number | string): Promise<{ partitionId: string; vector: Float32Array } | null>;

  deleteVector(id: number | string): Promise<boolean>;

  findNearest(query: Vector, k?: number, options?: any): Promise<SearchResult[]>;

  // Partition management
  createPartition(id: string, name: string, options?: any): Promise<string>;
  setActivePartition(id: string): Promise<void>;
  getPartition(id: string): Promise<any>;
  getActivePartition(): Promise<any>;
  getPartitionConfigs(): any[];

  // Database management
  getStats(): Promise<PartitionedDBStats>;
  savePartitionConfigs(): Promise<void>;
  close(): Promise<void>;
  buildIndexHNSW(partitionId?: string, options?: BuildIndexHNSWOptions): Promise<void>;
  findNearestHNSW(
    query: Vector,
    k: number,
    options: SearchOptions & {
      partitionIds?: string[];
      exactDimensions?: boolean;
    }
  ): Promise<SearchResult[]>;
  getMetadata(id: number | string): Promise<{ partitionId: string; metadata: Record<string, any> } | null>;
  saveHNSWIndices(partitionId?: string): Promise<void>;
  loadHNSWIndices(partitionId?: string): Promise<void>;
  save(): Promise<void>;
  IsReady(): boolean;
  initializationPromise: Promise<void>;
  getMetadataWithFieldAcrossPartitions(
    criteria: string | string[] | Record<string, any>,
    values?: any | any[],
    option?: {
      limit: number;
    }
  ): Promise<Array<{ partitionId: string; vectorId: number | string; metadata: Record<string, any> }>>;
}

/**
 * Interface for UnifiedSearch stats
 */
export interface UnifiedSearchStats {
  search: {
    calls: number;
    totalTime: number;
    avgTime: number;
    methodCounts: Record<string, number>;
    lastSearchTime: number;
    errors: number;
    lastError?: Error;
    lastSearchTimestamp?: Date;
    methods: {
      knn: { available: boolean; stats?: KNNStats };
      hnsw: { available: boolean; stats?: HNSWStats };
      hybrid: { available: boolean; stats?: HybridSearchStats };
    };
    reranker: { available: boolean };
  };
  database: {
    vectorCount: number;
    dimensions: {
      counts: Record<number, number>;
      unique: number;
    };
  };
}

/**
 * Interface for KNN search statistics
 */
export interface KNNStats {
  calls: number;
  totalTime: number;
  avgTime: number;
  lastSearchTime: number;
  cacheHits: number;
  cacheMisses: number;
  workerCount: number;
  workerBusy: number;
  cached: {
    normalizedVectorsCount: number;
    vectorNormsCount: number;
    resultsCount: number;
  };
  options: KNNOptions;
}
// Extended KNN options with performance settings
export interface KNNOptions {
  metric: string;
  useMultithreading: boolean;
  useHeap: boolean;
  batchSize: number;
  earlyStoppingThreshold: number;
  maxThreads: number;
  spatialPartitioning?: boolean;
  vectorizedCalculation?: boolean;
  cacheResults?: boolean;
  blockSize?: number;
  partitionCount?: number;
}

export interface WorkerInfo {
  worker: Worker;
  busy: boolean;
}

/**
 * Options for LSH configuration
 */
export interface LSHOptions {
  dimensions?: number;
  numberOfHashes?: number;
  numberOfBuckets?: number;
  allowMismatchedDimensions?: boolean;
}

/**
 * Options for LSH build index
 */
export interface BuildIndexOptions {
  progressCallback?: (progress: number) => void;
  dimensionGroups?: boolean;
}

/**
 * Options for LSH load index
 */
export interface LoadIndexOptions {
  allowMismatchedDimensions?: boolean;
}
/**
 * Product Quantization options interface
 */
export interface PQOptions {
  vectorSize?: number;
  subvectorSize?: number;
  numSubvectors?: number;
  numClusters?: number;
  dynamicDimensions?: boolean;
  minSubquantizers?: number; // New option to specify minimum subquantizers
}

/**
 * Training options interface
 */
export interface TrainingOptions {
  progressCallback?: (progress: number) => void;
}

/**
 * Load model options
 */
export interface LoadModelOptions {
  dynamicDimensions?: boolean;
}

/**
 * Storage options for persistence
 */
export interface StorageOptions {
  db: PartitionedVectorDB;
  basePath?: string;
  autoSave?: boolean;
  saveInterval?: number;
  compressionEnabled?: boolean;
  filePrefix?: string;
}
/**
 * BatchSearch type definitions
 */

export interface BatchQuery {
  query: Vector;
  k: number;
  options?: UnifiedSearchOptions;
}

export interface BatchSearchOptions {
  filter?: (id: number | string, meta: any) => boolean;
  maxBatchSize?: number;
  maxWorkers?: number;
  useWorkers?: boolean;
  disableWorkers?: boolean;
  prioritizeOrder?: boolean;
  groupSimilarQueries?: boolean;
  workerPath?: string;
  [key: string]: any;
}

export interface BatchSearchResult {
  results: SearchResult[][];
  stats?: {
    totalTime: number;
    queriesProcessed: number;
    workersUsed: number;
  };
}
/**
 * Interface for HybridSearch stats when using PartitionedVectorDB
 */
export interface HybridSearchStats {
  options: HybridSearchOptions; // Giữ lại options cấu hình
  dbStats: Record<string, any>; // Lấy toàn bộ stats từ Partitioned DB
  // Có thể thêm các thống kê riêng của HybridSearch nếu cần
}

/**
 * Options for BatchSearch using PartitionedVectorDB
 */
export interface BatchSearchOptions {
  maxBatchSize?: number;
  prioritizeOrder?: boolean;
  groupSimilarQueries?: boolean; // Tùy chọn này có thể vẫn hữu ích ở mức độ nào đó
  defaultSearchTimeout?: number; // Timeout cho mỗi truy vấn trong batch
}

// --- Core Types ---

export type Vector = Float32Array | number[];

export interface VectorData {
  id: number | string;
  vector: Vector;
  metadata?: Record<string, any>;
}

export interface SearchResult {
  id: number | string;
  dist: number;
  metadata?: Record<string, any>; // Thêm metadata trực tiếp nếu thường xuyên include
}

export type DistanceMetric = 'euclidean' | 'cosine';

// --- Database & Persistence Types ---

export interface PersistenceOptions {
  dbPath?: string; // Đường dẫn chính cho dữ liệu DB
  autoSave?: boolean;
  saveIntervalMs?: number; // Đổi tên rõ ràng hơn
  useCompression?: boolean;
}

export interface BackupOptions {
  destinationPath?: string; // Đường dẫn thư mục backup
  compress?: boolean;
  includeIndexes?: boolean;
  excludeMetadata?: boolean;
  tag?: string; // Optional tag for backup identification
}

export interface ImportExportOptions {
  format?: 'json' | 'binary' | 'csv';
  includeMetadata?: boolean;
  compression?: boolean;
  csvSeparator?: string;
  precision?: number;
}

// --- Clustering Configuration (for ClusteredVectorDB) ---

export interface ClusteringConfiguration {
  clusterSize?: number; // Target cluster size
  searchProbes?: number; // Number of clusters to check during search
  newClusterThresholdFactor?: number;
  newClusterDistanceThreshold?: number;
  maxClusters?: number; // Hard limit
  distanceMetric?: DistanceMetric; // Metric for clustering
  useCompression?: boolean; // Compression specific to cluster data storage
  kmeansMaxIterations?: number; // Max iterations for k-means clustering
}

// --- Partitioning Configuration (for PartitionedVectorDB) ---

export interface PartitionConfig {
  id: string;
  name: string;
  dbDirName: string; // Directory name relative to partitionsDir
  active: boolean;
  vectorCount: number;
  description?: string;
  properties?: Record<string, any>;
  // Allow overriding global clustering config per partition
  clustering?: Partial<ClusteringConfiguration>;
}

export interface PartitioningConfiguration {
  partitionsDir?: string; // Directory for all partitions
  partitionCapacity?: number; // Max vectors per partition (approximate)
  autoLoadPartitions?: boolean; // Load active/recent on start
  autoCreatePartitions?: boolean; // Create new partition when active is full
  maxActivePartitions?: number; // Max partitions loaded in memory (LRU)
  defaultVectorSize?: number | null; // Hint for vector size if needed early
  // Default clustering options applied to *new* partitions if not overridden
  defaultClusterOptions?: ClusteringConfiguration;
}

// --- Indexing General Types ---

export type IndexType = 'hnsw' | 'lsh' | 'pq' | 'flat'; // Flat is often implicit

export interface IndexBuildOptions {
  progressCallback?: (progress: { type: IndexType; percentage: number }) => void;
}

// --- HNSW Index Types ---

export interface HNSWNode {
  id: number | string;
  connections: Map<number, Set<number | string>>;
  dimension?: number;
}

export interface HNSWIndexConfiguration {
  M?: number; // Max connections per node
  efConstruction?: number; // Construction candidate list size
  efSearch?: number; // Search candidate list size
  maxLevel?: number;
  levelProbability?: number;
  distanceMetric?: DistanceMetric; // Metric used *within* HNSW
  // dimensionAware is usually determined internally based on DB
  nodes: HNSWNode[];
}

export interface HNSWBuildOptions extends IndexBuildOptions {
  // HNSW specific build options can go here if any
}

export interface HNSWLoadOptions {
  // HNSW specific load options can go here if any
}

export interface HNSWStats {
  totalNodes: number;
  maxM: number;
  efConstruction: number;
  efSearch: number;
  levels: number;
  nodesPerLevel: number[];
  avgConnectionsPerLevel: number[];
  entryPoint: number | string | null;
  dimensionAware: boolean;
  dimensionGroups?: number; // If dimension aware
  dimensions?: {
    // If dimension aware
    counts: Record<number, number>;
    entryPoints: Record<string, number | string>;
  };
  deletedNodesCount: number;
}

// --- LSH Index Types --- (Tương tự HNSW)

export interface LSHIndexConfiguration {
  numberOfHashes?: number;
  numberOfBuckets?: number;
  // allowMismatchedDimensions handled internally
}
export interface LSHBuildOptions extends IndexBuildOptions {}
export interface LSHLoadOptions {}
// Add LSHStats if needed

// --- PQ Index Types --- (Tương tự HNSW)

export interface PQIndexConfiguration {
  subvectorSize?: number; // Dimension of each subvector
  numClusters?: number; // Number of clusters (centroids) per subquantizer (k in k-means)
  // dynamicDimensions handled internally
}
export interface PQBuildOptions extends IndexBuildOptions {
  // PQ training might have specific options
}
export interface PQLoadOptions {}
// Add PQStats if needed

// --- Indexing Configuration (for IndexManager or similar) ---

export interface IndexingConfiguration {
  indexPath?: string; // Directory for index files (relative to dbPath?)
  buildOnStart?: boolean; // Build indexes when DB starts
  autoLoad?: boolean; // Save indexes automatically after building
  autoSave?: boolean; // Save indexes automatically after building
  autoRebuildThreshold?: number; // Rebuild if DB size changes significantly
  runKMeansOnLoad: boolean; // Run K-Means after loading if needed
  // Configuration for specific index types
  // Use optional properties: if the property exists, the index is enabled
  hnsw?: HNSWIndexConfiguration;
  lsh?: LSHIndexConfiguration;
  pq?: PQIndexConfiguration;
  flat?: {}; // Flat usually has no config, existence implies usage/availability
}

// --- Search Types ---

export interface BaseSearchOptions {
  k?: number;
  filter?: (id: number | string, metadata?: Record<string, any>) => boolean;
  includeMetadata?: boolean;
  includeVectors?: boolean;
  distanceMetric?: DistanceMetric; // Overrides default if specified
}

// Options specific to *how* the search is performed, especially within PartitionedDB
export interface SearchExecutionOptions {
  partitionIds?: string[]; // Limit search to specific partitions
  // Clustered search specific
  probes?: number; // Alias for searchProbes in clustering, specific to this search
  // HNSW search specific
  efSearch?: number; // Overrides HNSW efSearch config for this query
}

// Options passed to unified search methods
export interface UnifiedSearchOptions extends BaseSearchOptions, SearchExecutionOptions {
  useHNSW?: boolean; // Prefer HNSW if available?
  rerank?: boolean; // Apply reranking?
  rerankingMethod?: RerankingMethod;
  // Timeout?
  searchTimeoutMs?: number;
  rerankLambda?: number; // Lambda for reranking (if applicable)
  skipCache?: boolean; // Skip cache for this search
  searchMethod?: string; // e.g., 'hnsw', 'lsh', 'hybrid'
}

// --- Batch Search Types ---

export interface BatchSearchQuery {
  query: Vector;
  k: number;
  // Options for this specific query within the batch
  options?: BaseSearchOptions & SearchExecutionOptions & { useHNSW?: boolean };
}

export interface BatchSearchConfiguration {
  maxBatchSize?: number;
  prioritizeOrder?: boolean; // Maintain result order corresponding to input queries
  groupSimilarQueries?: boolean; // Optimize by grouping identical queries
  defaultSearchTimeoutMs?: number; // Timeout per query in the batch
}

// --- Reranking Types ---

export type RerankingMethod = 'diversity' | 'standard' | 'weighted';

export interface RerankingOptions {
  method?: RerankingMethod;
  k?: number; // Target number of results after reranking
  // Data needed for specific methods
  metadataMap?: Map<string | number, any>; // For weighted
  // vectorsMap?: Map<string | number, Vector>; // Potentially needed for diversity
  weights?: Record<string, number>; // For weighted
}

// --- Monitoring Types ---

export interface MonitoringConfiguration {
  enable?: boolean; // Simple toggle
  intervalMs?: number;
  historySize?: number;
  logToConsole?: boolean;
  enableSystemMetrics?: boolean;
  enableSearchMetrics?: boolean;
  enableDatabaseMetrics?: boolean;
  enableCacheMetrics?: boolean; // If caching is implemented
}

// ... (giữ lại các interface chi tiết cho Metrics: SystemMetricsHistory, SearchMetricsState, etc.)

// --- Event Data Types ---
// (Giữ lại các event data interface đã có, nhóm chúng lại)
export interface VectorDBEventData {
  /* ... */
}
export interface IndexManagerEventData {
  /* ... */
}
export interface PartitionedDBEventData {
  /* ... */
}
export interface MonitorEvents {
  /* ... */
}
// ... other event data interfaces

// --- Overall System Configuration ---

export interface SystemConfiguration {
  version: string;
  // Core persistence settings
  persistence: PersistenceOptions;

  // Default behavior settings
  defaults: {
    vectorSize: number; // Suggestion, can be null
    k: number;
    distanceMetric: DistanceMetric;
    cacheSize: number;
    maxConcurrentSearches: number;
    dimensionMismatchPenalty: number; // Penalty factor for comparing vectors with different dimensions
  };

  // Configuration for specific modules/layers
  clustering: ClusteringConfiguration; // For underlying ClusteredVectorDB instances
  partitioning: PartitioningConfiguration;
  indexing: IndexingConfiguration;
  batchSearch: BatchSearchConfiguration; // Optional if BatchSearch engine is used
  monitoring: MonitoringConfiguration; // Optional

  // Server settings (if applicable)
  server: {
    port: number;
    host: string;
    enableRateLimit?: boolean;
    maxRequestsPerMinute?: number;
    rateLimit: {
      enable?: boolean /** Whether to enable rate limiting */;
      maxRequestsPerMinute?: number /** Maximum requests per minute for rate limiting */;
      windowMs?: number /** Time window for rate limiting in milliseconds */;
    };
  };
  backup: DatabaseBackUp; // Optional backup settings
}
export type PartitionedDBStats = {
  status: string;
  partitions: {
    totalConfigured: number;
    loadedCount: number;
    maxLoaded: number;
    activeId: string | null;
    loadedIds: string[];
    configs: PartitionConfig[];
  };
  vectors: { totalConfigured: number; totalInMemory: number };
  memory: { estimatedUsageBytes: number; lruCacheSize: number };
  indices: {
    hnswLoadedCount: number;
    hnswLoadedIds: string[];
    hnswStats: Record<string, HNSWStats | null>; // Include detailed HNSW stats
  };
  settings: {
    partitionCapacity: number;
    autoCreatePartitions: boolean;
    useCompression: boolean;
    suggestedVectorSize: number | null;
    autoLoadHNSW: boolean;
    maxActivePartitions: number;
  };
  loadedPartitionDetails: Record<string, DBStats>;
};

/**
 * Statistics structure for UnifiedSearch when operating with PartitionedVectorDB.
 */
export interface UnifiedSearchPartitionedStats {
  /** Statistics related to the search calls made through UnifiedSearch */
  search: {
    calls: number;
    totalTime: number; // Tổng thời gian (ms) của các lệnh gọi search()
    avgTime: number; // Thời gian trung bình (ms) mỗi lệnh gọi search()
    methodCounts: Record<string, number>; // Đếm số lần sử dụng mỗi phương thức DB (vd: {'partitioned-hnsw': 10, 'partitioned-clustered': 5})
    lastSearchTime: number; // Thời gian (ms) của lệnh gọi search() cuối cùng
    errors: number; // Số lỗi xảy ra trong search()
    lastError?: Error; // Lỗi cuối cùng xảy ra
    lastSearchTimestamp?: Date; // Thời điểm lệnh gọi search() cuối cùng
  };
  /** Statistics obtained directly from the underlying PartitionedVectorDB instance */
  database: PartitionedDBStats; // Nhúng toàn bộ stats từ Partitioned DB
  /** Information about the reranking capability */
  reranker: {
    available: boolean;
    // Có thể thêm stats của reranker nếu có
  };
  // Có thể thêm các mục thống kê khác nếu cần, ví dụ:
  // cache?: CacheMetricsSnapshotData; // Nếu UnifiedSearch có cache riêng
}

/**
 * Represents the configuration for a database backup.
 *
 * @property backupIntervalMs - Optional. The interval, in milliseconds, at which the database should be backed up.
 */
export type DatabaseBackUp = {
  backupIntervalMs?: number;
};

/**
 * Configuration options for the database.
 */
export interface DatabaseOptions {
  /**
   * Configuration for persistence options.
   */
  persistence: PersistenceOptions;

  /**
   * Suggested vector size (dimensionality). Providing this value can improve performance,
   * but it can also be inferred automatically.
   */
  vectorSize?: number | null;

  /**
   * Configuration for indexing behavior and settings.
   */
  indexing: IndexingConfiguration;

  /**
   * Configuration for clustering behavior, used for underlying ClusteredVectorDB instances.
   */
  clustering: ClusteringConfiguration;

  /**
   * Configuration for partitioning behavior and settings.
   */
  partitioning: PartitioningConfiguration;

  /**
   * Size of the Least Recently Used (LRU) cache for search results.
   */
  cacheSize?: number;

  /**
   * Maximum number of concurrent search operations allowed.
   */
  maxConcurrentSearches?: number;

  /**
   * Interval (in milliseconds) for automatically saving partition configurations
   * and potentially indices. Set to 0 to disable.
   */
  backup?: DatabaseBackUp;

  /**
   * Configuration for the performance and system monitoring module.
   */
  monitoring?: MonitoringConfiguration;
}

// --- 2. DatabasePartitionedEvents ---

/**
 * Defines the events emitted by the DatabasePartitioned class.
 * Includes forwarded events from underlying components and specific high-level events.
 */
export type DatabaseEvents = {
  /** Emitted when the database finishes asynchronous initialization and is ready for use. */
  ready: void;
  /** Emitted when the database instance is closing or has closed. */
  close: void;
  /** Emitted when a general error occurs within the DatabasePartitioned instance or its components. */
  error: { message: string; error?: Error | unknown; context?: string };

  // --- Forwarded Partition Events (subset from PartitionedDBEventData) ---
  /** Emitted when a new partition configuration is created. */
  'partition:created': PartitionedDBEventData['partition:created'];
  /** Emitted when a partition is successfully loaded into memory. */
  'partition:loaded': PartitionedDBEventData['partition:loaded'];
  /** Emitted when a partition is unloaded from memory (e.g., by LRU cache). */
  'partition:unloaded': PartitionedDBEventData['partition:unloaded'];
  /** Emitted when the active partition changes. */
  'partition:activated': PartitionedDBEventData['partition:activated'];
  /** Emitted when an error occurs related to a specific partition operation. */
  'partition:error': PartitionedDBEventData['partition:error'];
  'partition:indexed': PartitionedDBEventData['partition:indexed'];
  'partition:indexLoaded': PartitionedDBEventData['partition:indexLoaded'];
  'partition:indexSaved': PartitionedDBEventData['partition:indexSaved'];

  // --- Forwarded Indexing Events ---
  /** Emitted to report progress during HNSW index building for a partition. */
  'index:progress': PartitionedDBEventData['partition:indexProgress']; // Payload needs definition in PartitionedDBEventData
  /** Emitted when HNSW index building for a partition completes. */
  'index:complete': { partitionId?: string }; // Simplified event for completion
  /** Emitted when an error occurs during index building. */
  'index:error': { partitionId?: string; error: Error | unknown }; // Simplified error event
  'search:start': {};

  // --- Forwarded Search Events (from UnifiedSearch) ---
  /** Emitted when a search operation (via findNearest/search) completes successfully. */
  'search:complete': {
    // Define payload based on UnifiedSearch 'search:complete'
    methodUsed: string;
    searchOnlyTime: number; // Time spent in DB search
    rerankTime: number; // Time spent reranking
    totalTime: number; // Total time for the search call
    resultCount: number;
    kRequested: number;
    optionsUsed: UnifiedSearchOptions;
  };
  /** Emitted when a search operation fails. */
  'search:error': {
    // Define payload based on UnifiedSearch 'search:error'
    error: Error | unknown;
    method: string;
    options: UnifiedSearchOptions;
    totalTime: number;
  };

  // --- DatabasePartitioned Specific Events ---
  /** Emitted when partition configurations and/or indices have been successfully saved. */
  'save:complete': { type: 'config' | 'indices' | 'config_indices' }; // Indicates what was saved
  /** Emitted when the simplified backup (saving configs/indices) completes. */
  'backup:complete': { type: 'config_index' };
  'search:cacheHit': {
    options: Record<string, any>;
    k: number;
  };
  initializing: void;
  warn: {
    message: string;
    context: string;
    error: Error | unknown;
  };
};

export type ISystem = {
  platform: string;
  cpuCores: number;
  totalMemoryMB: number;
  freeMemoryMB: number;
  nodeVersion: string;
};
// --- 3. DatabasePartitionedStats ---

/**
 * Structure containing comprehensive statistics for the DatabasePartitioned instance.
 */
export interface DatabaseStats {
  /** Statistics obtained directly from the underlying PartitionedVectorDB instance. */
  database: PartitionedDBStats | null;

  /** Statistics related to search operations performed via this instance. */
  search: UnifiedSearchPartitionedStats['search'] | null; // Use the search part of UnifiedSearch stats

  /** Statistics for the search result cache managed by DatabasePartitioned. */
  searchCache: {
    size: number; // Current number of items in cache
    capacity: number; // Maximum cache size
    hits: number;
    misses: number;
    hitRate: number | null; // Cache hit rate (0.0 to 1.0)
  };

  /** Aggregated performance metrics collected by DatabasePartitioned. */
  performance: {
    queries: number; // Total search queries processed
    avgSearchTimeMs: number; // Average total time per search call (incl. cache check, concurrency)
    cacheHitRate: number; // Overall cache hit rate percentage (0-100)
    concurrentSearches: number; // Number of search operations currently active
  };

  /** Basic information about the host system. */
  system?: ISystem;

  /** Memory usage of the current Node.js process. */
  memoryUsage: NodeJS.MemoryUsage;

  /** Overall state of the database instance. */
  state: {
    isReady: boolean;
    isClosed: boolean;
    status: string;
  };

  /** Key configuration options currently in effect. */
  options: PartitioningConfiguration & {
    cacheSize: number;
    vectorSize: number;
    maxConcurrentSearches: number;
    autoSaveIntervalMs: number; // Interval for auto-saving partition configs/indices
    monitoringEnabled: boolean; // Whether monitoring is active
    // Add other relevant resolved options
  };
}

/**
 * Request interfaces for type checking
 */
export interface AddVectorRequest {
  id?: number | string;
  vector: Vector;
  metadata?: Record<string, any>;
}

export interface BulkAddRequest {
  vectors: AddVectorRequest[];
  options?: {
    buildIndex: number;
  };
}

export interface SearchRequest {
  query: Vector;
  k?: number;
  method?: string;
  filters?: Record<string, any>;
  options?: Record<string, any>;
  includeMetadata?: boolean;
  includeVectors?: boolean;
  useParallel?: boolean;
}

export interface BatchSearchRequest {
  queries: {
    query: Vector;
    k?: number;
    filters?: Record<string, any>;
  }[];
  options?: Record<string, any>;
}

export interface UpdateMetadataRequest {
  id: number | string;
  metadata: Record<string, any>;
  operation?: 'replace' | 'merge';
}

export interface TrainIndexRequest {
  indexType: string;
  options?: Record<string, any>;
}

export interface FilterConfig {
  field: string;
  operator: '$eq' | '$ne' | '$gt' | '$gte' | '$lt' | '$lte' | '$in' | '$nin' | '$exists' | '$regex';
  value: any;
}

export interface SaveLoadDatabaseRequest {
  path: string;
  options?: Record<string, any>;
}

export type IServerOptions = {
  /** Port to run the server on */
  port?: number;
  /** Host address to bind the server to */
  host?: string;

  /** Express middleware to add before the API routes */
  middleware?: express.RequestHandler[];

  rateLimit?: {
    /** Whether to enable rate limiting */
    enable?: boolean;
    /** Maximum requests per minute for rate limiting */
    maxRequestsPerMinute?: number;
    /** Time window for rate limiting in milliseconds */
    windowMs?: number;
  };
  /** Whether to enable debug logging */
  debug?: boolean;
  database?: DatabaseOptions;
  /** Custom error handler */
  errorHandler?: (err: Error, req: Request, res: Response, next: NextFunction) => void;
};

/**
 * The API context object containing shared resources
 */
export interface ApiContext {
  timer: Timer;
  createFilterFunction: (filters: Record<string, any> | FilterConfig[]) => (id: number | string, metadata?: Record<string, any> | null) => boolean;
  database: Database;
}

/**
 * Return type for the createServer function
 */
export interface IServerInstance {
  app: Express;
  gracefulShutdown: () => Promise<void>;
  database: Database;
  context: ApiContext;
}

/**
 * Options for KNN when using PartitionedVectorDB
 */
export interface KNNOptionsPartitioned {
  metric?: DistanceMetric; // Still need metric

  cacheResults?: boolean; // Still useful
}

/**
 * Statistics for KNN (simplified version for PartitionedDB)
 */
export interface KNNStatsPartitioned {
  calls: number;
  totalTime: number;
  avgTime: number;
  lastSearchTime: number;
  cacheHits: number;
  cacheMisses: number;
  cachedResultsCount: number;
  options: Required<KNNOptionsPartitioned>;
}

// Define custom event types for HybridEngineSearch (if needed)
// or may not be needed if only forwarding from DB without emitting custom events
export interface HybridSearchEvents {
  'search:complete': {
    querySize: number;
    k: number;
    dbMethodUsed: string;
    resultCount: number;
    totalTime: number;
    // Can add other information if needed
  };
  'search:error': {
    error: unknown;
    dbMethodUsed: string;
    totalTime: number;
  };
  // Forward indexing events from DB
  'indexing:start': { method: string };
  'indexing:progress': {
    method: string;
    partitionId?: string;
    percentage: number;
  };
  'indexing:complete': {
    method: string;
    partitionId?: string;
    timeMs?: number;
  };
  'indexing:error': { method: string; partitionId?: string; error: unknown };
}

// Type aliases for DB event payloads (keep as before)
export type IndexProgressPayload = PartitionedDBEventData['partition:indexProgress'] extends infer T ? (T extends { id: string; progress: number } ? T : any) : any;
export type IndexedPayload = PartitionedDBEventData['partition:indexed'] extends infer T ? (T extends { id: string; indexType: string } ? T : any) : any;
export type PartitionErrorPayload = PartitionedDBEventData['partition:error'] extends infer T ? (T extends { id?: string; error: unknown; operation: string } ? T : any) : any;
