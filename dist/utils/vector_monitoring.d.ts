import { CacheMetricsState, DatabaseMetricsState, ISystem, MetricsSnapshot, MonitorEvents, MonitoringOptions, SearchMetricsState, SystemMetricsHistory, TypedEventEmitter } from '../types';
declare const VectorDBMonitor_base: new () => TypedEventEmitter<MonitorEvents>;
/**
 * A monitoring utility for tracking system, search, database, and cache metrics
 * in a vector database environment. This class extends `EventEmitter` to emit
 * events related to metrics collection, cache hits/misses, and errors.
 *
 * The `VectorDBMonitor` collects and maintains metrics such as CPU usage, memory
 * usage, query performance, database state, and cache efficiency. It provides
 * methods to start and stop monitoring, record events, and retrieve the current
 * metrics state and history.
 *
 * ### Features:
 * - Periodic collection of system metrics (CPU, memory, load average).
 * - Tracking and calculation of search metrics (queries per minute, average response time, P95).
 * - Monitoring database state (vector count, memory usage).
 * - Cache hit/miss tracking and hit rate calculation.
 * - Event-based architecture for emitting metrics snapshots, cache events, and errors.
 *
 * ### Usage:
 * 1. Instantiate the monitor with optional configuration.
 * 2. Start the monitoring process using `start()`.
 * 3. Record search and cache events using `recordSearch`, `recordCacheHit`, and `recordCacheMiss`.
 * 4. Retrieve metrics snapshots via the `metrics` event or `getMetrics()` method.
 * 5. Stop monitoring using `stop()` when no longer needed.
 *
 * @example
 * ```typescript
 * const monitor = new VectorDBMonitor({
 *   interval: 30000, // Collect metrics every 30 seconds
 *   logToConsole: true, // Log metrics snapshots to the console
 * });
 *
 * monitor.on("metrics", (snapshot) => {
 *   console.log("Metrics Snapshot:", snapshot);
 * });
 *
 * monitor.start();
 *
 * // Record a search operation
 * monitor.recordSearch({
 *   duration: 120, // Response time in ms
 *   method: "vectorSearch",
 *   results: 10,
 *   cacheUsed: 0
 * });
 *
 * // Stop monitoring
 * monitor.stop();
 * ```
 *
 * @extends TypedEventEmitter<MonitorEvents>
 */
export declare class VectorDBMonitor extends VectorDBMonitor_base {
    private readonly options;
    private metricsHistory;
    private searchState;
    private databaseState;
    private cacheState;
    private timer;
    private monitorInterval;
    private startTime;
    private queriesInLastInterval;
    private lastCpuInfo;
    constructor(options?: MonitoringOptions);
    /** Starts the monitoring interval. */
    start(): void;
    /** Stops the monitoring interval. */
    stop(): void;
    /** Collects all enabled metrics and emits a snapshot. */
    collectMetrics(): Promise<MetricsSnapshot>;
    /** Collects current system metrics (CPU, Memory, Load) and updates history. */
    private _collectSystemMetrics;
    /**
     * Calculates derived search metrics like QPM and P95 based on current state.
     * This version calculates QPM based on queries recorded within the monitor interval.
     */
    private _calculateSearchMetrics;
    /** Calculates cache metrics based on current state. */
    private _calculateCacheMetrics;
    /** Logs a metrics snapshot to the console. */
    private _logSnapshot;
    /** Adds a metric value to the history array, trimming if necessary. */
    private _addMetricHistory;
    /** Records a completed search operation. */
    recordSearch(data: {
        duration: number;
        method: string;
        results: number;
        cacheUsed: number;
    }): void;
    /** Records a cache hit event. */
    recordCacheHit(): void;
    /** Records a cache miss event. */
    recordCacheMiss(): void;
    /** Records a generic event. */
    recordEvent(eventType: string, data: any): void;
    /** Records an error event. */
    recordError(context: string, error: Error | unknown, extraData?: any): void;
    /** Updates the database-related metrics. Called externally. */
    updateDatabaseMetrics(data: Partial<DatabaseMetricsState>): void;
    /** Gets the current metrics state and history. */
    getMetrics(): {
        startTime: string;
        options: Required<MonitoringOptions>;
        history: {
            system: SystemMetricsHistory;
        };
        currentState: {
            search: Omit<SearchMetricsState, 'recentResponseTimes'>;
            database: DatabaseMetricsState;
            cache: CacheMetricsState;
        };
    };
    getSystemMetrics(): ISystem;
}
export {};
