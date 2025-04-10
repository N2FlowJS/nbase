// --- START OF FILE vector_monitoring_updated.ts ---

import os from 'os';
import { EventEmitter } from 'events';
import { createTimer, Timer } from './profiling';
import { CacheMetricsSnapshotData, CacheMetricsState, CurrentSystemMetrics, DatabaseMetricsState, ISystem, MetricsSnapshot, MonitorEvent, MonitorEvents, MonitoringOptions, SearchMetricsState, SystemMetricsHistory, TypedEventEmitter } from '../types';

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
export class VectorDBMonitor extends (EventEmitter as new () => TypedEventEmitter<MonitorEvents>) {
  private readonly options: Required<MonitoringOptions>; // Use Required for internal consistency
  private metricsHistory: {
    system: SystemMetricsHistory;
    // Search history (like QPM) can be derived, no need to store full history here
  };
  private searchState: SearchMetricsState;
  private databaseState: DatabaseMetricsState;
  private cacheState: CacheMetricsState; // State for cache hits/misses

  private timer: Timer;
  private monitorInterval: NodeJS.Timeout | null = null;
  private startTime: number;
  private queriesInLastInterval: number = 0; // Counter for QPM calculation

  // For CPU calculation
  private lastCpuInfo: { time: number; cpus: os.CpuInfo[] } | null = null;

  constructor(options: MonitoringOptions = {}) {
    super();

    // Resolve options with defaults
    this.options = {
      interval: options.interval ?? 60000, // Default: 1 minute
      historySize: options.historySize ?? 60, // Default: Keep last 60 points (e.g., 1 hour of 1-min intervals)
      logToConsole: options.logToConsole ?? false,
      enableSystemMetrics: options.enableSystemMetrics !== false, // Default: true
      enableSearchMetrics: options.enableSearchMetrics !== false, // Default: true
      enableDatabaseMetrics: options.enableDatabaseMetrics !== false, // Default: true
      enableCacheMetrics: options.enableCacheMetrics !== false, // Default: true
    };

    // Initialize metric storage
    this.metricsHistory = {
      system: { cpu: [], memory: [], loadAvg1m: [] },
    };
    this.searchState = {
      queryCount: 0,
      averageResponseTime: 0,
      p95ResponseTime: 0,
      queriesPerMinute: 0, // Calculated periodically
      methodUsage: {},
      recentResponseTimes: [], // Limited buffer for P95/Avg calculation
    };
    this.databaseState = {
      vectorCount: 0,
      memoryUsageBytes: 0,
      partitionCount: 0,
    };
    this.cacheState = {
      hits: 0,
      misses: 0,
    };

    this.timer = createTimer();
    this.startTime = Date.now();
  }

  /** Starts the monitoring interval. */
  start(): void {
    if (this.monitorInterval) {
      console.warn('Monitoring is already running.');
      return;
    }

    console.log(`Starting monitoring (interval: ${this.options.interval}ms)`);
    // Collect initial metrics immediately
    this.collectMetrics().catch((err) => console.error('Initial metrics collection failed:', err));

    this.monitorInterval = setInterval(() => {
      this.collectMetrics().catch((err) => console.error('Periodic metrics collection failed:', err));
    }, this.options.interval);
    this.monitorInterval.unref(); // Allow process to exit if this is the only timer
  }

  /** Stops the monitoring interval. */
  stop(): void {
    if (this.monitorInterval) {
      clearInterval(this.monitorInterval);
      this.monitorInterval = null;
      console.log('Monitoring stopped.');
    }
  }

  /** Collects all enabled metrics and emits a snapshot. */
  async collectMetrics(): Promise<MetricsSnapshot> {
    this.timer.start('collect');
    const collectionTimestamp = Date.now();

    // --- System Metrics ---
    let systemMetrics: CurrentSystemMetrics = {
      cpuUsage: null,
      memoryUsage: null,
      loadAvg1m: null,
    };
    if (this.options.enableSystemMetrics) {
      systemMetrics = this._collectSystemMetrics(); // This now correctly calls _addMetricHistory inside
    }

    // --- Search Metrics ---
    let calculatedSearchMetrics = this._calculateSearchMetrics(); // Calculate QPM, P95 etc. based on current state

    // --- Database Metrics ---
    // databaseState is updated externally via updateDatabaseMetrics

    // --- Cache Metrics ---
    let cacheMetrics: CacheMetricsSnapshotData = {
      hits: 0,
      misses: 0,
      hitRate: null,
    };
    if (this.options.enableCacheMetrics) {
      cacheMetrics = this._calculateCacheMetrics();
    }

    // --- Finalize Snapshot ---
    const collectionTimeMs = this.timer.stop('collect').total;
    const uptimeSeconds = (collectionTimestamp - this.startTime) / 1000;

    const snapshot: MetricsSnapshot = {
      timestamp: new Date(collectionTimestamp).toISOString(),
      uptimeSeconds: parseFloat(uptimeSeconds.toFixed(2)),
      collectionTimeMs: parseFloat(collectionTimeMs.toFixed(2)),
      metrics: {
        system: systemMetrics,
        search: {
          // Use the calculated metrics, not the raw state
          queryCount: this.searchState.queryCount, // Total count is still relevant
          averageResponseTime: calculatedSearchMetrics.averageResponseTime,
          p95ResponseTime: calculatedSearchMetrics.p95ResponseTime,
          queriesPerMinute: calculatedSearchMetrics.queriesPerMinute,
          methodUsage: { ...this.searchState.methodUsage }, // Copy method usage
        },
        database: { ...this.databaseState }, // Use current DB state
        cache: cacheMetrics, // Add cache metrics
      },
    };

    // Reset interval-based counters
    this.queriesInLastInterval = 0;

    // Emit and Log
    this.emit('metrics', snapshot);
    if (this.options.logToConsole) {
      this._logSnapshot(snapshot);
    }

    return snapshot;
  }

  /** Collects current system metrics (CPU, Memory, Load) and updates history. */
  private _collectSystemMetrics(): CurrentSystemMetrics {
    let cpuUsage: number | null = null;
    let memoryUsage: number | null = null;
    let loadAvg1m: number | null = null;

    try {
      // Memory Usage
      const totalMem = os.totalmem();
      const freeMem = os.freemem();
      memoryUsage = totalMem > 0 ? (totalMem - freeMem) / totalMem : null;
      this._addMetricHistory('memory', memoryUsage);

      // Load Average
      const loadAvg = os.loadavg();
      loadAvg1m = loadAvg.length > 0 ? loadAvg[0] : null;
      this._addMetricHistory('loadAvg1m', loadAvg1m); // Correct key used here

      // CPU Usage (Interval Calculation)
      const currentCpuInfo = { time: Date.now(), cpus: os.cpus() };
      if (this.lastCpuInfo && currentCpuInfo.time > this.lastCpuInfo.time) {
        let totalDiff = 0;
        let idleDiff = 0;

        for (let i = 0; i < currentCpuInfo.cpus.length; i++) {
          const currentCore = currentCpuInfo.cpus[i];
          // Ensure the corresponding core exists in the previous snapshot
          if (this.lastCpuInfo.cpus && i < this.lastCpuInfo.cpus.length) {
            const lastCore = this.lastCpuInfo.cpus[i];
            const currentTotal = Object.values(currentCore.times).reduce((a, b) => a + b, 0);
            const lastTotal = Object.values(lastCore.times).reduce((a, b) => a + b, 0);
            totalDiff += currentTotal - lastTotal;
            idleDiff += currentCore.times.idle - lastCore.times.idle;
          } else {
            // Handle cases where the number of CPUs might change or last data is incomplete
            // For simplicity, we might skip calculation in this edge case or handle it based on requirements
            console.warn(`CPU core mismatch or missing last data at index ${i}. Skipping diff calculation for this core.`);
          }
        }

        if (totalDiff > 0) {
          // Prevent division by zero and ensure usage is between 0 and 1
          cpuUsage = Math.max(0, Math.min(1, (totalDiff - idleDiff) / totalDiff));
        }
      }
      this.lastCpuInfo = currentCpuInfo; // Update last info for next interval
      this._addMetricHistory('cpu', cpuUsage);
    } catch (error: any) {
      console.error('Error collecting system metrics:', error);
      this.emit('error', {
        message: 'System metrics collection failed',
        error,
        context: 'CollectSystemMetrics',
      });
      // Return nulls on error
      return { cpuUsage: null, memoryUsage: null, loadAvg1m: null };
    }

    // Return formatted values
    return {
      cpuUsage: cpuUsage !== null ? parseFloat(cpuUsage.toFixed(4)) : null,
      memoryUsage: memoryUsage !== null ? parseFloat(memoryUsage.toFixed(4)) : null,
      loadAvg1m: loadAvg1m !== null ? parseFloat(loadAvg1m.toFixed(2)) : null,
    };
  }

  /**
   * Calculates derived search metrics like QPM and P95 based on current state.
   * This version calculates QPM based on queries recorded within the monitor interval.
   */
  private _calculateSearchMetrics(): Omit<SearchMetricsState, 'queryCount' | 'methodUsage' | 'recentResponseTimes'> {
    let calculatedP95 = 0;
    let calculatedAvg = this.searchState.averageResponseTime; // Use the rolling average calculated in recordSearch

    // Calculate P95 from recentResponseTimes buffer
    if (this.searchState.recentResponseTimes.length > 0) {
      const sortedTimes = [...this.searchState.recentResponseTimes].sort((a, b) => a - b);
      const p95Index = Math.max(0, Math.ceil(sortedTimes.length * 0.95) - 1);
      calculatedP95 = parseFloat((sortedTimes[p95Index] ?? 0).toFixed(2));
    }

    // Calculate QPM based on queries counted in the last interval
    const intervalSeconds = this.options.interval / 1000;
    const calculatedQPM = intervalSeconds > 0 ? Math.round((this.queriesInLastInterval / intervalSeconds) * 60) : 0;

    return {
      averageResponseTime: calculatedAvg,
      p95ResponseTime: calculatedP95,
      queriesPerMinute: calculatedQPM,
    };
  }

  /** Calculates cache metrics based on current state. */
  private _calculateCacheMetrics(): CacheMetricsSnapshotData {
    const total = this.cacheState.hits + this.cacheState.misses;
    const hitRate = total > 0 ? parseFloat((this.cacheState.hits / total).toFixed(4)) : null; // Use null for 0/0 case? Or 0?
    return {
      hits: this.cacheState.hits,
      misses: this.cacheState.misses,
      hitRate: hitRate,
    };
  }

  /** Logs a metrics snapshot to the console. */
  private _logSnapshot(snapshot: MetricsSnapshot): void {
    const cpu = snapshot.metrics.system.cpuUsage !== null ? `${(snapshot.metrics.system.cpuUsage * 100).toFixed(1)}%` : 'N/A';
    const mem = snapshot.metrics.system.memoryUsage !== null ? `${(snapshot.metrics.system.memoryUsage * 100).toFixed(1)}%` : 'N/A';
    const load = snapshot.metrics.system.loadAvg1m !== null ? snapshot.metrics.system.loadAvg1m.toFixed(2) : 'N/A';
    const qpm = snapshot.metrics.search.queriesPerMinute;
    const avgT = snapshot.metrics.search.averageResponseTime.toFixed(2);
    const p95 = snapshot.metrics.search.p95ResponseTime.toFixed(2);
    const vecCount = snapshot.metrics.database.vectorCount;
    const cacheHitRate = snapshot.metrics.cache?.hitRate !== null ? `${(snapshot.metrics.cache.hitRate * 100).toFixed(1)}%` : 'N/A'; // Added cache

    console.log(
      `[${snapshot.timestamp}] Monitor: ` + `CPU=${cpu} | Mem=${mem} | Load1m=${load} | QPM=${qpm} | ` + `AvgTime=${avgT}ms | P95=${p95}ms | Vectors=${vecCount} | CacheHit=${cacheHitRate}` // Added cache
    );
  }

  /** Adds a metric value to the history array, trimming if necessary. */
  private _addMetricHistory(key: keyof SystemMetricsHistory, value: number | null): void {
    // Check if the key is valid for system metrics history
    if (!(key in this.metricsHistory.system)) {
      console.warn(`Attempted to add metric to non-existent history key: system.${key}`);
      return;
    }
    if (value === null || value === undefined || isNaN(value)) return; // Don't add invalid values

    const historyArray = this.metricsHistory.system[key];
    historyArray.push(value);
    // Trim history
    if (historyArray.length > this.options.historySize) {
      historyArray.shift();
    }
  }

  /** Records a completed search operation. */
  recordSearch(data: { duration: number; method: string; results: number; cacheUsed: number }): void {
    if (!this.options.enableSearchMetrics || !data) return;

    const { duration, method } = data;

    this.searchState.queryCount++;
    this.queriesInLastInterval++; // Increment counter for QPM calculation

    // Update method usage
    this.searchState.methodUsage[method] = (this.searchState.methodUsage[method] || 0) + 1;

    // Update rolling average and P95 buffer
    const times = this.searchState.recentResponseTimes;
    times.push(duration);
    // Keep buffer size limited (e.g., last 1000 queries for P95/Avg) - Make this configurable?
    const maxResponseTimeHistory = 1000;
    if (times.length > maxResponseTimeHistory) {
      times.shift();
    }
    // Recalculate rolling average (simple moving average over the buffer)
    const sum = times.reduce((a, b) => a + b, 0);
    this.searchState.averageResponseTime = parseFloat((sum / times.length).toFixed(2));

    // P95 is calculated during collectMetrics
  }

  /** Records a cache hit event. */
  recordCacheHit(): void {
    if (!this.options.enableCacheMetrics) return;
    this.cacheState.hits++;
    this.emit('cache:hit', undefined); // Emit event (void payload)
  }

  /** Records a cache miss event. */
  recordCacheMiss(): void {
    if (!this.options.enableCacheMetrics) return;
    this.cacheState.misses++;
    this.emit('cache:miss', undefined); // Emit event (void payload)
  }

  /** Records a generic event. */
  recordEvent(eventType: string, data: any): void {
    const event: MonitorEvent = {
      type: eventType,
      timestamp: Date.now(),
      data,
    };
    this.emit('event', event);
    if (this.options.logToConsole) {
      console.log(
        `[${new Date(event.timestamp).toISOString()}] Monitor Event: ${eventType}`,
        JSON.stringify(data) // Stringify data for cleaner logging potentially
      );
    }
  }

  /** Records an error event. */
  recordError(context: string, error: Error | unknown, extraData?: any): void {
    const errorMessage = error instanceof Error ? error.message : String(error);
    console.error(`[${new Date().toISOString()}] Monitor Error [${context}]: ${errorMessage}`, extraData ?? '');
    this.emit('error', {
      message: errorMessage,
      error: error instanceof Error ? error : undefined,
      context: context,
      ...extraData, // Include extra data if provided
    });
  }

  /** Updates the database-related metrics. Called externally. */
  updateDatabaseMetrics(data: Partial<DatabaseMetricsState>): void {
    // Use Partial for flexibility
    if (!this.options.enableDatabaseMetrics || !data) return;
    if (data.vectorCount !== undefined) {
      this.databaseState.vectorCount = data.vectorCount;
    }
    if (data.memoryUsageBytes !== undefined) {
      this.databaseState.memoryUsageBytes = data.memoryUsageBytes;
    }
    // Add other DB metrics here if needed
  }

  /** Gets the current metrics state and history. */
  getMetrics(): {
    startTime: string;
    options: Required<MonitoringOptions>;
    history: { system: SystemMetricsHistory };
    currentState: {
      search: Omit<SearchMetricsState, 'recentResponseTimes'>; // Exclude raw times
      database: DatabaseMetricsState;
      cache: CacheMetricsState;
    };
  } {
    const currentSearchMetrics = this._calculateSearchMetrics();
    // Return a deep copy or a structured object to prevent external modification
    return {
      startTime: new Date(this.startTime).toISOString(),
      options: { ...this.options }, // Shallow copy of options is usually fine
      history: {
        system: {
          // Deep copy history arrays
          cpu: [...this.metricsHistory.system.cpu],
          memory: [...this.metricsHistory.system.memory],
          loadAvg1m: [...this.metricsHistory.system.loadAvg1m],
        },
      },
      currentState: {
        search: {
          // Use calculated metrics + state that doesn't include raw times
          queryCount: this.searchState.queryCount,
          averageResponseTime: currentSearchMetrics.averageResponseTime,
          p95ResponseTime: currentSearchMetrics.p95ResponseTime,
          queriesPerMinute: currentSearchMetrics.queriesPerMinute,
          methodUsage: { ...this.searchState.methodUsage },
        },
        database: { ...this.databaseState }, // Copy current state
        cache: { ...this.cacheState }, // Copy current cache state
      },
    };
  }
  getSystemMetrics(): ISystem {
    return {
      platform: os.platform(),
      cpuCores: os.cpus().length,
      totalMemoryMB: Math.round(os.totalmem() / (1024 * 1024)),
      freeMemoryMB: Math.round(os.freemem() / (1024 * 1024)),
      nodeVersion: process.version,
    };
  }
}

// --- END OF FILE vector_monitoring_updated.ts ---
