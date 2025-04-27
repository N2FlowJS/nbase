"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.KMeans = void 0;
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
class KMeans {
    constructor(k = 8, maxIterations = 100, tolerance = 0.001) {
        this.k = k;
        this.maxIterations = maxIterations;
        this.tolerance = tolerance;
    }
    /**
     * Cluster a set of vectors using k-means
     * @param vectors - Set of vectors to cluster
     * @returns Array of cluster centroids
     */
    async cluster(vectors) {
        if (vectors.length === 0) {
            throw new Error('Cannot cluster empty vector set');
        }
        if (vectors.length <= this.k) {
            // If we have fewer vectors than clusters, return vectors as centroids
            return vectors.map((v) => (v instanceof Float32Array ? v : new Float32Array(v)));
        }
        // Initialize centroids with k-means++ method
        const centroids = this._initializeCentroids(vectors);
        // Iterative refinement
        let iterations = 0;
        let changed = true;
        while (changed && iterations < this.maxIterations) {
            // Assign vectors to nearest centroids
            const assignments = this._assignToClusters(vectors, centroids);
            // Update centroids based on assignments
            changed = this._updateCentroids(vectors, assignments, centroids);
            iterations++;
            // Allow for async processing to not block main thread
            if (iterations % 10 === 0) {
                await new Promise((resolve) => setTimeout(resolve, 0));
            }
        }
        return centroids;
    }
    /**
     * Initialize centroids using k-means++ method
     * @private
     */
    _initializeCentroids(vectors) {
        const centroids = [];
        const n = vectors.length;
        // Choose first centroid randomly
        const firstIdx = Math.floor(Math.random() * n);
        centroids.push(vectors[firstIdx] instanceof Float32Array ? vectors[firstIdx].slice() : new Float32Array(vectors[firstIdx]));
        // KMeans++ initialization
        let distances = new Float32Array(n).fill(0);
        for (let i = 1; i < this.k; i++) {
            let totalDistance = 0;
            for (let j = 0; j < n; j++) {
                let minDist = Infinity;
                for (const centroid of centroids) {
                    const dist = this._squaredDistance(vectors[j], centroid);
                    minDist = Math.min(minDist, dist);
                }
                distances[j] = minDist;
                totalDistance += minDist;
            }
            // Select next centroid with probability proportional to squared distance
            let rand = Math.random() * totalDistance;
            let nextCentroidIndex = -1;
            for (let j = 0; j < n; j++) {
                rand -= distances[j];
                if (rand <= 0) {
                    nextCentroidIndex = j;
                    break;
                }
            }
            if (nextCentroidIndex !== -1) {
                centroids.push(vectors[nextCentroidIndex] instanceof Float32Array ? vectors[nextCentroidIndex].slice() : new Float32Array(vectors[nextCentroidIndex]));
            }
            else {
                // Fallback: choose a random vector
                let randomIndex = Math.floor(Math.random() * n);
                centroids.push(vectors[randomIndex] instanceof Float32Array ? vectors[randomIndex].slice() : new Float32Array(vectors[randomIndex]));
            }
        }
        return centroids;
    }
    /**
     * Assign vectors to nearest centroids
     * @private
     */
    _assignToClusters(vectors, centroids) {
        const n = vectors.length;
        const assignments = new Array(n);
        for (let i = 0; i < n; i++) {
            let minDist = Infinity;
            let nearestCentroid = 0;
            for (let c = 0; c < centroids.length; c++) {
                const dist = this._squaredDistance(vectors[i], centroids[c]);
                if (dist < minDist) {
                    minDist = dist;
                    nearestCentroid = c;
                }
            }
            assignments[i] = nearestCentroid;
        }
        return assignments;
    }
    /**
     * Update centroids based on assignments
     * @private
     */
    _updateCentroids(vectors, assignments, centroids) {
        const n = vectors.length;
        const dimensions = vectors[0].length;
        const k = centroids.length;
        // Count vectors in each cluster
        const counts = new Array(k).fill(0);
        // Initialize new centroids
        const newCentroids = [];
        for (let c = 0; c < k; c++) {
            newCentroids.push(new Float32Array(dimensions));
        }
        // Sum vectors in each cluster
        for (let i = 0; i < n; i++) {
            const clusterIdx = assignments[i];
            const vector = vectors[i];
            counts[clusterIdx]++;
            for (let d = 0; d < dimensions; d++) {
                newCentroids[clusterIdx][d] += vector[d];
            }
        }
        // Calculate means and check for significant changes
        let changed = false;
        for (let c = 0; c < k; c++) {
            // Handle empty clusters
            if (counts[c] === 0) {
                // Find the cluster with most points and take a point from there
                let maxCount = 0;
                let largestCluster = 0;
                for (let j = 0; j < k; j++) {
                    if (counts[j] > maxCount) {
                        maxCount = counts[j];
                        largestCluster = j;
                    }
                }
                // Find points in largest cluster
                const pointsInLargest = [];
                for (let i = 0; i < n; i++) {
                    if (assignments[i] === largestCluster) {
                        pointsInLargest.push(i);
                    }
                }
                // Take a random point from largest cluster
                if (pointsInLargest.length > 0) {
                    const randomIdx = Math.floor(Math.random() * pointsInLargest.length);
                    const vectorIdx = pointsInLargest[randomIdx];
                    // Copy this vector as new centroid for empty cluster
                    for (let d = 0; d < dimensions; d++) {
                        newCentroids[c][d] = vectors[vectorIdx][d];
                    }
                    changed = true;
                }
                continue;
            }
            // Calculate mean and check for change
            for (let d = 0; d < dimensions; d++) {
                newCentroids[c][d] /= counts[c];
                // Check if centroid moved significantly
                const diff = Math.abs(newCentroids[c][d] - centroids[c][d]);
                if (diff > this.tolerance) {
                    changed = true;
                }
            }
            // Update centroid
            centroids[c] = newCentroids[c];
        }
        return changed;
    }
    /**
     * Calculate squared Euclidean distance between vectors
     * @private
     */
    _squaredDistance(a, b) {
        let sum = 0;
        const len = Math.min(a.length, b.length);
        for (let i = 0; i < len; i++) {
            const diff = a[i] - b[i];
            sum += diff * diff;
        }
        return sum;
    }
}
exports.KMeans = KMeans;
//# sourceMappingURL=kmeans.js.map