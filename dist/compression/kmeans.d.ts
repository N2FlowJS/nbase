import { Vector } from '../types';
/**
 * KMeans class for clustering a set of vectors into k clusters using the k-means algorithm.
 *
 * The k-means algorithm partitions the input data into k clusters by iteratively refining
 * cluster centroids to minimize the variance within each cluster. This implementation
 * includes the k-means++ initialization method for better centroid selection and supports
 * asynchronous processing to avoid blocking the main thread during long computations.
 *
 * @example
 * ```typescript
 * const kmeans = new KMeans(3, 100, 0.01);
 * const vectors = [
 *     new Float32Array([1.0, 2.0]),
 *     new Float32Array([1.5, 1.8]),
 *     new Float32Array([5.0, 8.0]),
 *     new Float32Array([8.0, 8.0]),
 *     new Float32Array([1.0, 0.6]),
 *     new Float32Array([9.0, 11.0])
 * ];
 * const centroids = await kmeans.cluster(vectors);
 * console.log(centroids);
 * ```
 *
 * @class
 * @template Vector - A type representing a numerical vector, such as `Float32Array` or `number[]`.
 *
 * @property {number} k - The number of clusters to form.
 * @property {number} maxIterations - The maximum number of iterations for the algorithm.
 * @property {number} tolerance - The threshold for centroid movement to determine convergence.
 *
 * @constructor
 * @param {number} [k=8] - The number of clusters to form.
 * @param {number} [maxIterations=100] - The maximum number of iterations for the algorithm.
 * @param {number} [tolerance=0.001] - The threshold for centroid movement to determine convergence.
 */
export declare class KMeans {
    private k;
    private maxIterations;
    private tolerance;
    constructor(k?: number, maxIterations?: number, tolerance?: number);
    /**
     * Cluster a set of vectors using k-means
     * @param vectors - Set of vectors to cluster
     * @returns Array of cluster centroids
     */
    cluster(vectors: Vector[]): Promise<Float32Array[]>;
    /**
     * Initialize centroids using k-means++ method
     * @private
     */
    private _initializeCentroids;
    /**
     * Assign vectors to nearest centroids
     * @private
     */
    private _assignToClusters;
    /**
     * Update centroids based on assignments
     * @private
     */
    private _updateCentroids;
    /**
     * Calculate squared Euclidean distance between vectors
     * @private
     */
    private _squaredDistance;
}
