"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.HybridEngineSearch = void 0;
const events_1 = require("events");
const profiling_1 = require("../utils/profiling");
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
class HybridEngineSearch extends events_1.EventEmitter {
    constructor(db, options = {}) {
        super();
        this.isBuildingIndex = false;
        // --- Handler Methods for Event Forwarding ---
        // Use arrow function class properties to automatically bind `this`
        this.handleIndexProgress = (data) => {
            // Check the received data type if needed
            const progressValue = typeof data === 'object' && data !== null && typeof data.progress === 'number' ? data.progress : 0;
            const partitionId = typeof data === 'object' && data !== null ? data.id : undefined;
            this.emit('indexing:progress', {
                method: 'hnsw', // Or determine from data if possible
                partitionId: partitionId,
                percentage: progressValue * 100,
            });
        };
        this.handleIndexComplete = (data) => {
            // Check the received data type if needed
            const partitionId = typeof data === 'object' && data !== null ? data.id : undefined;
            const indexType = typeof data === 'object' && data !== null ? data.indexType : 'hnsw';
            const timeMs = typeof data === 'object' && data !== null ? data.timeMs : undefined;
            this.emit('indexing:complete', {
                method: indexType,
                partitionId: partitionId,
                timeMs: timeMs,
            });
        };
        this.handlePartitionError = (data) => {
            // Check the received data type if needed
            const operation = typeof data === 'object' && data !== null ? data.operation : '';
            const partitionId = typeof data === 'object' && data !== null ? data.id : undefined;
            const error = typeof data === 'object' && data !== null ? data.error : new Error('Unknown partition error');
            if (operation?.toLowerCase().includes('index')) {
                this.emit('indexing:error', {
                    method: 'unknown', // Or try to infer from operation/error
                    partitionId: partitionId,
                    error: error,
                });
            }
            // Handle other errors if needed
        };
        this.db = db;
        this.timer = (0, profiling_1.createTimer)();
        this.defaultK = options.defaultK || 10;
        this.defaultDistanceMetric = options.defaultDistanceMetric;
        this._setupEventForwarding();
    }
    // --- Setup and Teardown ---
    _setupEventForwarding() {
        if (typeof this.db.on === 'function') {
            const dbEmitter = this.db;
            // Use the handler methods defined
            dbEmitter.on('partition:indexProgress', this.handleIndexProgress);
            dbEmitter.on('partition:indexed', this.handleIndexComplete);
            dbEmitter.on('partition:error', this.handlePartitionError);
        }
    }
    close() {
        console.log('Closing HybridEngineSearch and removing listeners...');
        if (typeof this.db.off === 'function') {
            const dbEmitter = this.db;
            // Remove listeners using the exact handler methods registered
            dbEmitter.off('partition:indexProgress', this.handleIndexProgress);
            dbEmitter.off('partition:indexed', this.handleIndexComplete);
            dbEmitter.off('partition:error', this.handlePartitionError);
        }
        this.removeAllListeners(); // Remove listeners of HybridEngineSearch itself
        console.log('HybridEngineSearch closed.');
    }
    // --- Public Methods ---
    /**
     * Instructs the PartitionedVectorDB to build its HNSW indexes.
     * Adapts the progress callback signature.
     * @param options Options for the build process, accepting a structured progress callback.
     */
    async buildIndexes(options) {
        // <-- Use the Hybrid-specific options type here
        if (this.isBuildingIndex) {
            console.warn('Index building is already in progress.');
            return;
        }
        // Ensure the interface method uses the correct type from the DB's perspective
        if (typeof this.db.buildIndexHNSW !== 'function') {
            console.error('Database instance does not support HNSW index building (buildIndexHNSW method missing).');
            return;
        }
        this.isBuildingIndex = true;
        this.emit('indexing:start', { method: 'hnsw' });
        try {
            console.log('Calling PartitionedVectorDB.buildIndexHNSW...');
            // 1. Extract the user's structured callback (if provided)
            const userProgressCallback = options?.progressCallback;
            // 2. Create an intermediate callback adapter that matches the DB's expected signature
            const dbProgressCallbackAdapter = userProgressCallback
                ? (progress) => {
                    userProgressCallback(progress);
                }
                : undefined; // Pass undefined if the user didn't provide a callback
            // 3. Prepare options for the DB call using the DB's expected type
            const dbOptions = {
                // Spread other relevant options from `options` if DbBuildIndexHNSWOptions accepts them
                // e.g., ...options,
                progressCallback: dbProgressCallbackAdapter,
            };
            // 4. Call the DB method with the adapted options
            await this.db.buildIndexHNSW(undefined, dbOptions); // <-- Pass dbOptions
            console.log('PartitionedVectorDB index building process initiated.');
        }
        catch (error) {
            console.error('Error initiating index build:', error);
            this.emit('indexing:error', { method: 'hnsw', error: error });
        }
        finally {
            this.isBuildingIndex = false; // Consider better state management
        }
    }
    async findNearest(query, options = {}) {
        const k = options.k ?? this.defaultK;
        const distanceMetric = options.distanceMetric ?? this.defaultDistanceMetric;
        const useHNSW = options.useHNSW !== false;
        this.timer.start('hybrid_search_partitioned');
        let results = [];
        let dbMethodUsed = 'unknown';
        try {
            const dbSearchOptions = {
                k: k,
                filter: options.filter,
                includeMetadata: options.includeMetadata,
                distanceMetric: distanceMetric,
                partitionIds: options.partitionIds,
                efSearch: options.efSearch,
            };
            if (useHNSW && typeof this.db.findNearestHNSW === 'function') {
                dbMethodUsed = 'PartitionedDB.findNearestHNSW';
                this.timer.start(dbMethodUsed);
                results = await this.db.findNearestHNSW(query, k, dbSearchOptions);
                this.timer.stop(dbMethodUsed);
            }
            else if (typeof this.db.findNearest === 'function') {
                const { efSearch, ...clusteredOptions } = dbSearchOptions;
                dbMethodUsed = 'PartitionedDB.findNearest';
                this.timer.start(dbMethodUsed);
                results = await this.db.findNearest(query, k, clusteredOptions);
                this.timer.stop(dbMethodUsed);
            }
            else {
                throw new Error('No suitable search method available in the database instance.');
            }
            const timeInfo = this.timer.stop('hybrid_search_partitioned');
            const totalTime = timeInfo?.total || this.timer.getElapsed('hybrid_search_partitioned');
            this.emit('search:complete', {
                querySize: Array.isArray(query) ? query.length : query.byteLength / 4,
                k: k,
                dbMethodUsed: dbMethodUsed,
                resultCount: results.length,
                totalTime: totalTime,
            });
            return results;
        }
        catch (error) {
            const timeInfo = this.timer.stop('hybrid_search_partitioned');
            const totalTime = timeInfo?.total || this.timer.getElapsed('hybrid_search_partitioned');
            console.error(`Hybrid search error using ${dbMethodUsed}:`, error);
            this.emit('search:error', {
                error,
                dbMethodUsed,
                totalTime: totalTime,
            });
            throw error;
        }
    }
    async getStats() {
        if (this.db && typeof this.db.getStats === 'function') {
            try {
                return await this.db.getStats();
            }
            catch (error) {
                console.error('Failed to get stats from database:', error);
                return { error: 'Failed to retrieve DB stats' };
            }
        }
        else {
            console.warn('Database instance does not support getStats method.');
            return { warning: 'getStats not available on DB instance' };
        }
    }
}
exports.HybridEngineSearch = HybridEngineSearch;
//# sourceMappingURL=hybrid_search.js.map