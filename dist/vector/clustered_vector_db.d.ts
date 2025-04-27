import { VectorDB } from './vector_db';
import { ClusteredVectorDBOptions, DBStats, DistanceMetric, SearchResult, Vector } from '../types';
/**
 * A clustered vector database that extends the base VectorDB with efficient approximate nearest neighbor search.
 *
 * ClusteredVectorDB organizes vectors into clusters to improve search performance on large datasets.
 * Instead of performing an exhaustive linear search across all vectors, it first identifies the most
 * promising clusters and then only searches vectors within those clusters.
 *
 * Features:
 * - Dynamic cluster management with automatic creation of new clusters when needed
 * - Configurable clustering parameters to tune performance vs. accuracy tradeoffs
 * - Persistence of cluster state alongside vector data
 * - Support for different distance metrics
 * - Optional K-Means clustering for potentially better cluster quality
 *
 * @example
 * ```ts
 * const db = new ClusteredVectorDB(128, './vector-db', {
 *   clusterSize: 100,
 *   distanceMetric: 'cosine'
 * });
 *
 * // Add vectors with automatic cluster assignment
 * db.addVector('doc1', [0.1, 0.2, ...], { title: 'Document 1' });
 *
 * // Search efficiently using cluster-based approximation
 * const results = db.findNearest([0.3, 0.4, ...], 5);
 * ```
 *
 * @extends VectorDB
 */
export declare class ClusteredVectorDB extends VectorDB {
    readonly targetClusterSize: number;
    protected readonly newClusterThresholdFactor: number;
    protected readonly newClusterDistanceThreshold: number;
    protected readonly maxClusters: number;
    protected readonly distanceMetric: DistanceMetric;
    protected readonly kmeansMaxIterations: number;
    protected readonly runKMeansOnLoad: boolean;
    private kmeans;
    private clusters;
    private clusterCentroids;
    private clusterDimensions;
    private clusterIdCounter;
    constructor(suggestedVectorSize?: number | null, dbPath?: string | null, options?: ClusteredVectorDBOptions);
    protected _getClusterStateFilePath(): string;
    save(): Promise<void>;
    load(): Promise<void>;
    getDistanceMetric(): DistanceMetric;
    addVector(id: number | string | undefined, vector: Vector, metadata?: Record<string, any>): number | string;
    deleteVector(id: number | string): boolean;
    updateVector(id: number | string, vector: Vector): boolean;
    findNearest(query: Vector, k?: number, options?: {
        filter?: (id: number | string, metadata?: Record<string, any>) => boolean;
        metric?: DistanceMetric;
    }): SearchResult[];
    private _assignVectorToCluster;
    private _createNewCluster;
    private _removeVectorFromCluster;
    private _updateCentroidIncrementally;
    private _recalculateCentroid;
    private _rebuildAllClusters;
    /**
     * Runs the K-Means clustering algorithm to potentially improve cluster quality.
     * This is computationally more expensive than incremental updates.
     *
     * @param k - The target number of clusters. Defaults to the current number of clusters or a minimum of 1.
     * @param maxIterations - Maximum number of iterations for the algorithm. Defaults to instance configuration.
     * @returns A promise that resolves when K-Means completes.
     */
    runKMeans(k?: number, maxIterations?: number): Promise<void>;
    private _updateClustersFromKMeans;
    getStats(): DBStats;
    close(): Promise<void>;
    getClusterInfo(): Array<{
        id: number;
        centroid: Float32Array;
        size: number;
        dimension: number;
    }>;
    /**
   * Extract relationships between vectors based on distance or custom criteria.
   *
   * @param threshold - The maximum distance between vectors to consider them related.
   * @param metric - Distance metric to use (e.g., 'cosine', 'euclidean').
   * @returns An array of relationships, where each relationship links two vector IDs, their distance, and optional metadata.
   */
    extractRelationships(threshold: number, metric?: DistanceMetric): Array<{
        vector1: number | string;
        vector2: number | string;
        distance: number;
        metadata1?: Record<string, any>;
        metadata2?: Record<string, any>;
    }>;
    /**
       * Extract communities of related vectors based on distance threshold.
       * Uses cluster information to optimize the community detection process.
       *
       * @param threshold - The maximum distance between vectors to consider them related
       * @param metric - Distance metric to use (e.g., 'cosine', 'euclidean')
       * @returns Array of communities, where each community is an array of related vector information
       */
    extractCommunities(threshold: number, metric?: DistanceMetric): Array<Array<{
        id: number | string;
        metadata?: Record<string, any>;
    }>>;
}
