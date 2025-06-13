/// <reference types="node" />
import { Database } from './database/database';
import { PartitionedVectorDB } from './vector/partitioned_vector_db';
import express, { Request, Response, NextFunction, Express } from 'express';
import { Timer } from './utils/profiling';
import { ClusteredVectorDB } from './vector';
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
    limit?: number;
    offset?: number;
    stopEarly?: boolean;
}
/**
 * Hybrid search options interface
 */
export interface HybridSearchOptions extends UnifiedSearchOptions {
    buildIndexes?: boolean;
    methods?: string[];
    useParallelExecution?: boolean;
    cachingEnabled?: boolean;
    cacheSize?: number;
    useClustered?: boolean;
    partitionIds?: string[];
    buildIndexOptions?: BuildIndexHNSWOptions;
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
    [key: string]: any;
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
export interface VectorDBEventData {
    'vector:add': {
        id: IDVector;
        dimensions: number;
    };
    'vectors:bulkAdd': {
        count: number;
        ids: IDVector[];
    };
    'vector:delete': {
        id: number | string;
    };
    'metadata:add': {
        id: number | string;
        metadata: Record<string, any>;
    };
    'metadata:update': {
        id: number | string;
        metadata: Record<string, any>;
    };
    'db:save': {
        path: string;
        count: number;
    };
    'db:load': {
        path: string;
        count: number;
    };
    'db:close': {};
    'cluster:create': {
        clusterId: number;
        vectorId: number | string;
    };
    'cluster:delete': {
        clusterId: number;
    };
    'db:error': {
        operation: string;
        error: Error | unknown;
    };
    'kmeans:complete': {
        k: number;
        iterations: number;
    };
    'kmeans:error': {
        error: Error | unknown;
    };
    'kmeans:start': {
        k: number;
        iterations: number;
    };
    'vector:update': {
        id: number | string;
        dimensions: number;
    };
}
export type BulkAddResult = {};
export interface TypedEventEmitter<Events extends Record<string, any>> {
    on<E extends keyof Events>(event: E, listener: (payload: Events[E]) => void): this;
    once<E extends keyof Events>(event: E, listener: (payload: Events[E]) => void): this;
    off<E extends keyof Events>(event: E, listener: (payload: Events[E]) => void): this;
    emit<E extends keyof Events>(event: E, payload: Events[E]): boolean;
    listenerCount<E extends keyof Events>(event: E): number;
    listeners<E extends keyof Events>(event: E): ((payload: Events[E]) => void)[];
    removeAllListeners<E extends keyof Events>(event?: E): this;
}
export interface IndexProgressEvent {
    type: IndexType;
    progress: number;
    dimension?: number;
    dimensionAware?: boolean;
}
export interface IndexBuiltEvent {
    type: IndexType;
    timeMs: number;
    stats: any;
}
export interface IndexActionEvent {
    timeMs: number;
    types: IndexType[];
}
export interface IndexErrorEvent {
    type?: IndexType;
    error: any;
    timeMs?: number;
}
export interface IndexManagerEventData {
    'indexes:building': {
        types: IndexType[];
    };
    'indexes:built': IndexActionEvent;
    'indexes:error': IndexErrorEvent;
    'index:built': IndexBuiltEvent;
    'index:error': IndexErrorEvent;
    'index:progress': IndexProgressEvent;
    'indexes:saving': void;
    'indexes:saved': {
        timeMs: number;
    };
    'indexes:savingError': {
        error: any;
    };
    'indexes:loading': void;
    'indexes:loaded': {
        timeMs: number;
        loadedTypes: IndexType[];
    };
    'indexes:loadingError': {
        error: any;
        timeMs?: number;
    };
    reset: void;
}
export interface IndexOptions {
    indexPath?: string;
    buildOnStart?: boolean;
    autoSave?: boolean;
    autoBuildThreshold?: number;
    indexes?: {
        hnsw?: boolean;
        lsh?: boolean;
        pq?: boolean;
        flat?: boolean;
    };
    hnswOptions?: {
        M?: number;
        efConstruction?: number;
        efSearch?: number;
    };
    lshOptions?: {
        numberOfHashes?: number;
        numberOfBuckets?: number;
    };
    pqOptions?: {
        subvectorSize?: number;
        numClusters?: number;
    };
}
export interface IndexStats {
    indexTypes: IndexType[];
    isBuilding: boolean;
    lastBuildTimeMs?: number;
    lastSaveTimeMs?: number;
    lastLoadTimeMs?: number;
    dbVectorsAtLastBuild: number;
    indexes: Partial<Record<IndexType, any>>;
}
export interface TimerData {
    start: [number, number];
    splits: {
        label: string | null;
        elapsed: number;
    }[];
    lastDuration?: number;
}
export interface TimerResult {
    total: number;
    splits: {
        label: string | null;
        elapsed: number;
    }[];
}
export interface PerformanceMetrics {
    queries: number;
    totalSearchTime: number;
    avgSearchTime: number;
    totalAddTime: number;
    avgAddTime: number;
    cacheHits: number;
    cacheMisses: number;
    queryTimes: number[];
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
    enableDatabaseMetrics?: boolean;
    enableCacheMetrics?: boolean;
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
    dimension?: number;
}
/**
 * HNSW options interface
 */
export interface HNSWOptions {
    M?: number;
    efConstruction?: number;
    efSearch?: number;
    distanceFunc?: (a: Vector, b: Vector) => number;
    maxLevel?: number;
    levelProbability?: number;
    entryPointId?: number | string;
    dimensionAware?: boolean;
    nodes?: HNSWNode[];
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
export interface MonitoringOptions {
    interval?: number;
    historySize?: number;
    logToConsole?: boolean;
    enableSystemMetrics?: boolean;
    enableSearchMetrics?: boolean;
    enableDatabaseMetrics?: boolean;
}
export interface CpuTimes {
    user: number;
    nice: number;
    sys: number;
    idle: number;
    irq: number;
}
export interface SystemMetricsHistory {
    cpu: number[];
    memory: number[];
    loadAvg1m: number[];
}
export interface CurrentSystemMetrics {
    cpuUsage: number | null;
    memoryUsage: number | null;
    loadAvg1m: number | null;
}
export interface SearchMetricsState {
    queryCount: number;
    averageResponseTime: number;
    p95ResponseTime: number;
    queriesPerMinute: number;
    methodUsage: Record<string, number>;
    recentResponseTimes: number[];
    responseTimes?: number[];
}
export interface DatabaseMetricsState {
    vectorCount: number;
    memoryUsageBytes: number;
    partitionCount: number;
}
export interface MetricsSnapshot {
    timestamp: string;
    uptimeSeconds: number;
    collectionTimeMs: number;
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
export interface MonitorEvent {
    type: string;
    timestamp: number;
    data: any;
}
export interface SearchEventData {
    timestamp: number;
    duration: number;
    method: string;
}
export interface MonitorEvents {
    metrics: MetricsSnapshot;
    event: MonitorEvent;
    error: {
        message: string;
        error?: Error;
        context?: string;
    };
    'cache:hit': void;
    'cache:miss': void;
}
export interface CacheMetricsState {
    hits: number;
    misses: number;
}
export interface CacheMetricsSnapshotData {
    hits: number;
    misses: number;
    hitRate: number | null;
}
export interface ClusteredVectorDBOptions {
    useCompression?: boolean;
    clusterSize?: number;
    newClusterThresholdFactor?: number;
    newClusterDistanceThreshold?: number;
    maxClusters?: number;
    distanceMetric?: DistanceMetric;
    kmeansMaxIterations?: number;
    runKMeansOnLoad?: boolean;
}
export interface PartitionConfig {
    id: string;
    name: string;
    dbDirName: string;
    active: boolean;
    vectorCount: number;
    description?: string;
    properties?: Record<string, any>;
    clusterSize?: number;
}
export interface PartitionedVectorDBOptions {
    partitionsDir?: string;
    partitionCapacity?: number;
    autoLoadPartitions?: boolean;
    autoCreatePartitions?: boolean;
    maxActivePartitions?: number;
    vectorSize?: number | null;
    useCompression?: boolean;
    autoLoadHNSW?: boolean;
    clusterOptions?: Omit<ClusteredVectorDBOptions, 'clusterSize'>;
    runKMeansOnLoad?: boolean;
}
export interface PartitionedDBEventData {
    'db:initialized': {
        partitionCount: number;
        loadedCount: number;
        activeId: string | null;
    };
    'partitions:loaded': {
        count: number;
        active: string | null;
    };
    'partition:loaded': {
        id: string;
        name: string;
        vectorCount: number;
        hnswLoaded: boolean;
    };
    'partition:indexLoaded': {
        id: string;
        indexType: string;
        path: string;
    };
    'partition:unloaded': {
        id: string;
    };
    'partition:created': {
        id: string;
        name: string;
        active: boolean;
    };
    'partition:activated': {
        id: string;
    };
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
    'vector:delete': {
        partitionId: string;
        vectorId: number | string;
    };
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
    'partition:progress': {
        id: string;
        progress: number;
    };
    'partition:save': {
        id: string;
        timeMs: number;
    };
    'partition:load': {
        id: string;
        timeMs: number;
    };
    'partition:saveError': {
        id: string;
        error: Error;
    };
    'partition:loadError': {
        id: string;
        error: Error;
    };
    'partition:reset': {
        id: string;
    };
    'partition:indexed': {
        id: string;
        indexType: string;
    };
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
    listBackups(): Promise<Array<{
        id: string;
        timestamp: Date;
        tag?: string;
        size: number;
    }>>;
    /**
     * Add vector(s) to the database
     */
    addVector(vector: Vector, metadata?: Record<string, any>): Promise<number | string>;
    bulkAdd(vectors: VectorData[]): Promise<{
        count: number;
    }>;
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
    addVector(id: number | string | undefined, vector: Vector, metadata?: Record<string, any>): Promise<{
        partitionId: string;
        vectorId: number | string;
    }>;
    bulkAdd(vectors: VectorData[]): Promise<{
        count: number;
        partitionIds: string[];
    }>;
    getVector(id: number | string): Promise<{
        partitionId: string;
        vector: Float32Array;
    } | null>;
    deleteVector(id: number | string): Promise<boolean>;
    findNearest(query: Vector, k?: number, options?: any): Promise<SearchResult[]>;
    createPartition(id: string, name: string, options?: any): Promise<string>;
    setActivePartition(id: string): Promise<void>;
    getPartition(id: string): Promise<ClusteredVectorDB | null>;
    getActivePartition(): Promise<any>;
    getPartitionConfigs(): any[];
    getStats(): Promise<PartitionedDBStats>;
    savePartitionConfigs(): Promise<void>;
    close(): Promise<void>;
    buildIndexHNSW(partitionId?: string, options?: BuildIndexHNSWOptions): Promise<void>;
    findNearestHNSW(query: Vector, k: number, options: SearchOptions & {
        partitionIds?: string[];
        exactDimensions?: boolean;
    }): Promise<SearchResult[]>;
    getMetadata(id: number | string): Promise<{
        partitionId: string;
        metadata: Record<string, any>;
    } | null>;
    saveHNSWIndices(partitionId?: string): Promise<void>;
    loadHNSWIndices(partitionId?: string): Promise<void>;
    save(): Promise<void>;
    IsReady(): boolean;
    initializationPromise: Promise<void>;
    getMetadataWithFieldAcrossPartitions(criteria: string | string[] | Record<string, any>, values?: any | any[], option?: {
        limit: number;
    }): Promise<Array<{
        partitionId: string;
        vectorId: number | string;
        metadata: Record<string, any>;
    }>>;
    extractRelationships(threshold: number, options: {
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
    extractCommunities(threshold: number, options: {
        metric?: DistanceMetric;
        partitionIds?: string[];
        includeMetadata?: boolean;
    }): Promise<Array<Array<{
        id: number | string;
        partitionId: string;
        metadata?: Record<string, any>;
    }>>>;
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
            knn: {
                available: boolean;
                stats?: KNNStats;
            };
            hnsw: {
                available: boolean;
                stats?: HNSWStats;
            };
            hybrid: {
                available: boolean;
                stats?: HybridSearchStats;
            };
        };
        reranker: {
            available: boolean;
        };
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
    minSubquantizers?: number;
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
    options: HybridSearchOptions;
    dbStats: Record<string, any>;
}
/**
 * Options for BatchSearch using PartitionedVectorDB
 */
export interface BatchSearchOptions {
    maxBatchSize?: number;
    prioritizeOrder?: boolean;
    groupSimilarQueries?: boolean;
    defaultSearchTimeout?: number;
}
export type Vector = Float32Array | number[];
export interface VectorData {
    id: number | string;
    vector: Vector;
    metadata?: Record<string, any>;
}
export interface SearchResult {
    id: number | string;
    dist: number;
    metadata?: Record<string, any>;
}
export type DistanceMetric = 'euclidean' | 'cosine';
export interface PersistenceOptions {
    dbPath?: string;
    autoSave?: boolean;
    saveIntervalMs?: number;
    useCompression?: boolean;
}
export interface BackupOptions {
    destinationPath?: string;
    compress?: boolean;
    includeIndexes?: boolean;
    excludeMetadata?: boolean;
    tag?: string;
}
export interface ImportExportOptions {
    format?: 'json' | 'binary' | 'csv';
    includeMetadata?: boolean;
    compression?: boolean;
    csvSeparator?: string;
    precision?: number;
}
export interface ClusteringConfiguration {
    clusterSize?: number;
    newClusterThresholdFactor?: number;
    newClusterDistanceThreshold?: number;
    maxClusters?: number;
    distanceMetric?: DistanceMetric;
    useCompression?: boolean;
    kmeansMaxIterations?: number;
}
export interface PartitionConfig {
    id: string;
    name: string;
    dbDirName: string;
    active: boolean;
    vectorCount: number;
    description?: string;
    properties?: Record<string, any>;
    clustering?: Partial<ClusteringConfiguration>;
}
export interface PartitioningConfiguration {
    partitionsDir?: string;
    partitionCapacity?: number;
    autoLoadPartitions?: boolean;
    autoCreatePartitions?: boolean;
    maxActivePartitions?: number;
    defaultVectorSize?: number | null;
    defaultClusterOptions?: ClusteringConfiguration;
}
export type IndexType = 'hnsw' | 'lsh' | 'pq' | 'flat';
export interface IndexBuildOptions {
    progressCallback?: (progress: {
        type: IndexType;
        percentage: number;
    }) => void;
}
export interface HNSWNode {
    id: number | string;
    connections: Map<number, Set<number | string>>;
    dimension?: number;
}
export interface HNSWIndexConfiguration {
    M?: number;
    efConstruction?: number;
    efSearch?: number;
    maxLevel?: number;
    levelProbability?: number;
    distanceMetric?: DistanceMetric;
    nodes: HNSWNode[];
}
export interface HNSWBuildOptions extends IndexBuildOptions {
}
export interface HNSWLoadOptions {
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
    dimensionGroups?: number;
    dimensions?: {
        counts: Record<number, number>;
        entryPoints: Record<string, number | string>;
    };
    deletedNodesCount: number;
}
export interface LSHIndexConfiguration {
    numberOfHashes?: number;
    numberOfBuckets?: number;
}
export interface LSHBuildOptions extends IndexBuildOptions {
}
export interface LSHLoadOptions {
}
export interface PQIndexConfiguration {
    subvectorSize?: number;
    numClusters?: number;
}
export interface PQBuildOptions extends IndexBuildOptions {
}
export interface PQLoadOptions {
}
export interface IndexingConfiguration {
    indexPath?: string;
    buildOnStart?: boolean;
    autoLoad?: boolean;
    autoSave?: boolean;
    autoRebuildThreshold?: number;
    runKMeansOnLoad: boolean;
    hnsw?: HNSWIndexConfiguration;
    lsh?: LSHIndexConfiguration;
    pq?: PQIndexConfiguration;
    flat?: {};
}
export interface BaseSearchOptions {
    k?: number;
    filter?: (id: number | string, metadata?: Record<string, any>) => boolean;
    includeMetadata?: boolean;
    includeVectors?: boolean;
    distanceMetric?: DistanceMetric;
}
export interface SearchExecutionOptions {
    partitionIds?: string[];
    efSearch?: number;
}
export interface UnifiedSearchOptions extends BaseSearchOptions, SearchExecutionOptions {
    useHNSW?: boolean;
    rerank?: boolean;
    rerankingMethod?: RerankingMethod;
    searchTimeoutMs?: number;
    rerankLambda?: number;
    skipCache?: boolean;
    searchMethod?: string;
}
export interface BatchSearchQuery {
    query: Vector;
    k: number;
    options?: BaseSearchOptions & SearchExecutionOptions & {
        useHNSW?: boolean;
    };
}
export interface BatchSearchConfiguration {
    maxBatchSize?: number;
    prioritizeOrder?: boolean;
    groupSimilarQueries?: boolean;
    defaultSearchTimeoutMs?: number;
}
export type RerankingMethod = 'diversity' | 'standard' | 'weighted';
export interface RerankingOptions {
    method?: RerankingMethod;
    k?: number;
    metadataMap?: Map<string | number, any>;
    weights?: Record<string, number>;
}
export interface MonitoringConfiguration {
    enable?: boolean;
    intervalMs?: number;
    historySize?: number;
    logToConsole?: boolean;
    enableSystemMetrics?: boolean;
    enableSearchMetrics?: boolean;
    enableDatabaseMetrics?: boolean;
    enableCacheMetrics?: boolean;
}
export interface VectorDBEventData {
}
export interface IndexManagerEventData {
}
export interface PartitionedDBEventData {
}
export interface MonitorEvents {
}
export interface SystemConfiguration {
    version: string;
    persistence: PersistenceOptions;
    defaults: {
        vectorSize: number;
        k: number;
        distanceMetric: DistanceMetric;
        cacheSize: number;
        maxConcurrentSearches: number;
        dimensionMismatchPenalty: number;
    };
    clustering: ClusteringConfiguration;
    partitioning: PartitioningConfiguration;
    indexing: IndexingConfiguration;
    batchSearch: BatchSearchConfiguration;
    monitoring: MonitoringConfiguration;
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
    backup: DatabaseBackUp;
    windowsService: {
        name: string;
        description: string;
        script: string;
    };
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
    vectors: {
        totalConfigured: number;
        totalInMemory: number;
    };
    memory: {
        estimatedUsageBytes: number;
        lruCacheSize: number;
    };
    indices: {
        hnswLoadedCount: number;
        hnswLoadedIds: string[];
        hnswStats: Record<string, HNSWStats | null>;
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
        totalTime: number;
        avgTime: number;
        methodCounts: Record<string, number>;
        lastSearchTime: number;
        errors: number;
        lastError?: Error;
        lastSearchTimestamp?: Date;
    };
    /** Statistics obtained directly from the underlying PartitionedVectorDB instance */
    database: PartitionedDBStats;
    /** Information about the reranking capability */
    reranker: {
        available: boolean;
    };
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
    error: {
        message: string;
        error?: Error | unknown;
        context?: string;
    };
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
    /** Emitted to report progress during HNSW index building for a partition. */
    'index:progress': PartitionedDBEventData['partition:indexProgress'];
    /** Emitted when HNSW index building for a partition completes. */
    'index:complete': {
        partitionId?: string;
    };
    /** Emitted when an error occurs during index building. */
    'index:error': {
        partitionId?: string;
        error: Error | unknown;
    };
    'search:start': {};
    /** Emitted when a search operation (via findNearest/search) completes successfully. */
    'search:complete': {
        methodUsed: string;
        searchOnlyTime: number;
        rerankTime: number;
        totalTime: number;
        resultCount: number;
        kRequested: number;
        optionsUsed: UnifiedSearchOptions;
    };
    /** Emitted when a search operation fails. */
    'search:error': {
        error: Error | unknown;
        method: string;
        options: UnifiedSearchOptions;
        totalTime: number;
    };
    /** Emitted when partition configurations and/or indices have been successfully saved. */
    'save:complete': {
        type: 'config' | 'indices' | 'config_indices';
    };
    /** Emitted when the simplified backup (saving configs/indices) completes. */
    'backup:complete': {
        type: 'config_index';
    };
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
/**
 * Structure containing comprehensive statistics for the DatabasePartitioned instance.
 */
export interface DatabaseStats {
    /** Statistics obtained directly from the underlying PartitionedVectorDB instance. */
    database: PartitionedDBStats | null;
    /** Statistics related to search operations performed via this instance. */
    search: UnifiedSearchPartitionedStats['search'] | null;
    /** Statistics for the search result cache managed by DatabasePartitioned. */
    searchCache: {
        size: number;
        capacity: number;
        hits: number;
        misses: number;
        hitRate: number | null;
    };
    /** Aggregated performance metrics collected by DatabasePartitioned. */
    performance: {
        queries: number;
        avgSearchTimeMs: number;
        cacheHitRate: number;
        concurrentSearches: number;
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
        autoSaveIntervalMs: number;
        monitoringEnabled: boolean;
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
    metric?: DistanceMetric;
    cacheResults?: boolean;
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
export interface HybridSearchEvents {
    'search:complete': {
        querySize: number;
        k: number;
        dbMethodUsed: string;
        resultCount: number;
        totalTime: number;
    };
    'search:error': {
        error: unknown;
        dbMethodUsed: string;
        totalTime: number;
    };
    'indexing:start': {
        method: string;
    };
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
    'indexing:error': {
        method: string;
        partitionId?: string;
        error: unknown;
    };
}
export type IndexProgressPayload = PartitionedDBEventData['partition:indexProgress'] extends infer T ? (T extends {
    id: string;
    progress: number;
} ? T : any) : any;
export type IndexedPayload = PartitionedDBEventData['partition:indexed'] extends infer T ? (T extends {
    id: string;
    indexType: string;
} ? T : any) : any;
export type PartitionErrorPayload = PartitionedDBEventData['partition:error'] extends infer T ? (T extends {
    id?: string;
    error: unknown;
    operation: string;
} ? T : any) : any;
