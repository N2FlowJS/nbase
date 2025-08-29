// src/utils/validation.ts
import { Vector, DistanceMetric } from '../types';

/**
 * Utility class for input validation and sanitization
 */
export class ValidationUtils {
  /**
   * Validates that a vector has the correct dimensions
   */
  static validateVectorDimensions(
    vector: Vector,
    expectedDimensions: number,
    vectorId?: string | number
  ): void {
    if (!Array.isArray(vector) && !(vector instanceof Float32Array)) {
      throw new Error(`Vector must be an array or Float32Array, got ${typeof vector}`);
    }

    if (vector.length !== expectedDimensions) {
      const id = vectorId ? ` (ID: ${vectorId})` : '';
      throw new Error(
        `Vector dimension mismatch${id}. Expected: ${expectedDimensions}, Got: ${vector.length}`
      );
    }
  }

  /**
   * Validates search parameters
   */
  static validateSearchParameters(k: number, maxAllowed: number = 1000): void {
    if (!Number.isInteger(k) || k <= 0) {
      throw new Error(`k must be a positive integer, got ${k}`);
    }

    if (k > maxAllowed) {
      throw new Error(`k cannot exceed ${maxAllowed}, got ${k}`);
    }
  }

  /**
   * Validates distance metric
   */
  static validateDistanceMetric(metric: string): asserts metric is DistanceMetric {
    const validMetrics: DistanceMetric[] = ['cosine', 'euclidean'];

    if (!validMetrics.includes(metric as DistanceMetric)) {
      throw new Error(`Invalid distance metric: ${metric}. Valid options: ${validMetrics.join(', ')}`);
    }
  }

  /**
   * Sanitizes and validates metadata object
   */
  static sanitizeMetadata(metadata: Record<string, any>): Record<string, any> {
    const sanitized: Record<string, any> = {};

    for (const [key, value] of Object.entries(metadata)) {
      // Skip undefined values
      if (value === undefined) continue;

      // Convert functions to strings for storage
      if (typeof value === 'function') {
        sanitized[key] = value.toString();
      } else {
        sanitized[key] = value;
      }
    }

    return sanitized;
  }
}

// src/utils/performance.ts
/**
 * Performance monitoring utilities
 */
export class PerformanceUtils {
  private static readonly MAX_SAMPLES = 1000;

  /**
   * Calculates moving average of an array of numbers
   */
  static calculateMovingAverage(values: number[], windowSize: number = 10): number {
    if (values.length === 0) return 0;
    if (values.length < windowSize) {
      return values.reduce((sum, val) => sum + val, 0) / values.length;
    }

    const window = values.slice(-windowSize);
    return window.reduce((sum, val) => sum + val, 0) / windowSize;
  }

  /**
   * Calculates percentile from an array of numbers
   */
  static calculatePercentile(values: number[], percentile: number): number {
    if (values.length === 0) return 0;

    const sorted = [...values].sort((a, b) => a - b);
    const index = Math.ceil((percentile / 100) * sorted.length) - 1;
    return sorted[Math.max(0, index)];
  }

  /**
   * Tracks performance metrics with automatic cleanup
   */
  static trackMetric(
    metrics: { values: number[]; total: number; count: number },
    value: number
  ): void {
    metrics.values.push(value);
    metrics.total += value;
    metrics.count++;

    // Keep only recent samples
    if (metrics.values.length > this.MAX_SAMPLES) {
      const removed = metrics.values.shift()!;
      metrics.total -= removed;
      metrics.count--;
    }
  }
}

// src/utils/constants.ts
/**
 * Application-wide constants
 */
export const CONSTANTS = {
  // Database
  DEFAULT_VECTOR_SIZE: 1536,
  DEFAULT_CACHE_SIZE: 1000,
  MAX_CONCURRENT_SEARCHES: Math.max(1, require('os').cpus().length - 1),

  // Search
  DEFAULT_SEARCH_K: 10,
  MAX_SEARCH_K: 1000,
  DEFAULT_EF_SEARCH: 50,
  DEFAULT_EF_CONSTRUCTION: 200,

  // HNSW
  DEFAULT_HNSW_M: 16,
  MAX_HNSW_LEVEL: 16,
  HNSW_LEVEL_PROBABILITY: 0.5,

  // Partitioning
  DEFAULT_PARTITION_CAPACITY: 100000,
  DEFAULT_MAX_ACTIVE_PARTITIONS: 3,

  // Clustering
  DEFAULT_CLUSTER_SIZE: 100,
  MAX_CLUSTERS: 1000,

  // Performance
  DEFAULT_SAVE_INTERVAL_MS: 60 * 1000, // 1 minute
  DEFAULT_MONITOR_INTERVAL_MS: 60 * 1000, // 1 minute
  MAX_QUERY_TIME_SAMPLES: 100,

  // File paths
  DEFAULT_DB_PATH: 'database',
  PARTITIONS_DIR_NAME: 'partitions',
  INDEX_DIR_NAME: 'hnsw',

  // Events
  CACHE_EVENTS: ['vector:add', 'vector:delete', 'vectors:bulkAdd', 'partition:created', 'partition:unloaded'],
  METRICS_EVENTS: ['vector:add', 'vector:delete', 'vectors:bulkAdd', 'partition:loaded', 'partition:unloaded'],
} as const;
