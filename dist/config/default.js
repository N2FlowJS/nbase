"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.defaultSystemConfiguration = void 0;
const os_1 = __importDefault(require("os")); // Import os nếu cần
/**
 * Default system configuration for the vector database ecosystem.
 * Provides reasonable starting values for various modules.
 * THIS IS THE SINGLE SOURCE OF TRUTH FOR DEFAULTS.
 */
exports.defaultSystemConfiguration = {
    version: '0.1.0',
    persistence: {
        dbPath: `${process.env.NBASE_DB_PATH}` || 'database',
        autoSave: true,
        saveIntervalMs: 1 * 60 * 1000, // 1 minutes
        useCompression: true,
    },
    defaults: {
        vectorSize: 1536,
        k: 10,
        distanceMetric: 'euclidean',
        cacheSize: 1000, // Thêm default cacheSize
        maxConcurrentSearches: Math.max(1, os_1.default.cpus().length - 1), // Thêm default maxConcurrentSearches
        dimensionMismatchPenalty: 0.01, // Default penalty factor for dimension mismatches.
    },
    clustering: {
        clusterSize: 100,
        newClusterThresholdFactor: 1.5,
        newClusterDistanceThreshold: 0.5,
        maxClusters: 1000,
        distanceMetric: 'euclidean',
        useCompression: true,
    },
    partitioning: {
        partitionsDir: './database/partitions',
        partitionCapacity: 100000,
        autoLoadPartitions: true,
        autoCreatePartitions: true,
        maxActivePartitions: 3,
    },
    indexing: {
        indexPath: './database/indexes',
        buildOnStart: true,
        autoSave: true,
        runKMeansOnLoad: true,
        autoRebuildThreshold: 500,
        hnsw: {
            // Không cần enable ở đây, logic enable nên ở code sử dụng
            M: 16,
            efConstruction: 200,
            efSearch: 50,
            distanceMetric: 'euclidean',
            nodes: [],
        },
        lsh: {
            // Không cần enable ở đây
            numberOfHashes: 128,
            numberOfBuckets: 1024,
        },
        pq: {
            subvectorSize: 8,
            numClusters: 256,
        },
    },
    batchSearch: {
        maxBatchSize: 64,
        prioritizeOrder: true,
        groupSimilarQueries: false,
        defaultSearchTimeoutMs: 15000,
    },
    monitoring: {
        enable: true,
        intervalMs: 60000,
        historySize: 60,
        logToConsole: false,
        enableSystemMetrics: true,
        enableSearchMetrics: true,
        enableDatabaseMetrics: true,
        enableCacheMetrics: true,
    },
    backup: {
        // Thêm mục backup nếu cần
        backupIntervalMs: 60 * 1000, // 60 minisecon default
        // Thêm các tùy chọn backup khác nếu có
    },
    server: {
        port: parseInt(process.env.NBASE_PORT || '1307', 10),
        host: process.env.NBASE_HOST || 'localhost',
        enableRateLimit: false,
        maxRequestsPerMinute: 1000,
        rateLimit: {
            windowMs: 1 * 60 * 1000, // 1 minute
            enable: true,
            maxRequestsPerMinute: 500,
        },
    },
};
//# sourceMappingURL=default.js.map