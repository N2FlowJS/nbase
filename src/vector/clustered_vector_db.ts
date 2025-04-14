// --- START OF FILE clustered_vector_db.ts ---

import { VectorDB } from './vector_db';
import config from '../config'; // Assuming config exists and has defaults
import { ClusteredVectorDBOptions, DBStats, DistanceMetric, IDVector, SearchResult, Vector } from '../types';
import { existsSync, promises as fsPromises } from 'fs';
import path from 'path';
import zlib from 'zlib'; // Import zlib for potential compression
import { promisify } from 'util'; // Import promisify
import {KMeans} from '../compression/kmeans'; // Import KMeans

const gzip = promisify(zlib.gzip);
const gunzip = promisify(zlib.gunzip);

// Helper function to pick k random distinct elements from an array
function getRandomElements<T>(arr: T[], k: number): T[] {
  if (k >= arr.length) {
    return [...arr]; // Return a copy of the whole array if k is too large
  }
  const shuffled = [...arr].sort(() => 0.5 - Math.random());
  return shuffled.slice(0, k);
}

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
export class ClusteredVectorDB extends VectorDB {
  // Configuration
  public readonly targetClusterSize: number;
  protected readonly newClusterThresholdFactor: number;
  protected readonly newClusterDistanceThreshold: number;
  protected readonly maxClusters: number;
  protected readonly distanceMetric: DistanceMetric;
  protected readonly kmeansMaxIterations: number; // New config option
  protected readonly runKMeansOnLoad: boolean; // Option to run K-Means automatically on load
  private kmeans: KMeans;

  // Clustering structures
  private clusters: Map<
    number,
    Array<{
      id: IDVector /* vector not needed here if memoryStorage is source */;
    }>
  >; // Store only IDs in cluster list
  private clusterCentroids: Map<number, Float32Array>;
  private clusterDimensions: Map<number, number>; // Track dimensions per cluster centroid
  private clusterIdCounter: number;

  constructor(suggestedVectorSize: number | null = null, dbPath: string | null = null, options: ClusteredVectorDBOptions = {}) {
    super(suggestedVectorSize, dbPath, {
      useCompression: options.useCompression,
    });

    // Set configuration with defaults from config or reasonable values
    this.targetClusterSize = options.clusterSize ?? config.clustering.clusterSize ?? 100;
    this.newClusterThresholdFactor = options.newClusterThresholdFactor ?? 1.5;
    this.newClusterDistanceThreshold = options.newClusterDistanceThreshold ?? 0.5;
    this.maxClusters = options.maxClusters ?? config.clustering.maxClusters ?? 1000; // Set a reasonable max
    this.distanceMetric = options.distanceMetric ?? 'euclidean'; // Default metric
    this.kmeansMaxIterations = options.kmeansMaxIterations ?? config.clustering.kmeansMaxIterations ?? 100; // K-Means iterations
    this.runKMeansOnLoad = options.runKMeansOnLoad ?? false; // Default to false
    this.clusterCentroids = new Map(); // Initialize clusterCentroids before use
    this.kmeans = new KMeans(this.clusterCentroids.size, this.kmeansMaxIterations);

    // Initialize clustering structures
    this.clusters = new Map();
    this.clusterDimensions = new Map();
    this.clusterIdCounter = 0; // Separate counter for cluster keys

    // No automatic rebuild after loading - we'll handle this in the load method
  }

  // --- File Path for Cluster State ---
  protected _getClusterStateFilePath(): string {
    if (!this.dbPath) throw new Error('DB path not set for cluster state');
    // Store cluster state separately from base vector/meta data
    console.log(`[ClusteredVectorDB] Cluster state file path: ${this.dbPath}`);

    return path.join(this.dbPath, 'cluster.json') + (this.useCompression ? '.gz' : '');
  }

  // --- Overridden Save Method ---
  override async save(): Promise<void> {
    if (!this.dbPath) {
      console.warn('[ClusteredVectorDB] No dbPath specified, skipping save.');
      return;
    }
    if (this.isClosed) {
      console.warn('[ClusteredVectorDB] Attempted to save a closed database.');
      return;
    }

    console.log(`[ClusteredVectorDB] Saving state to ${this.dbPath}`);

    // Use a single save promise to prevent race conditions if called multiple times
    if (this.savePromise) {
      console.log(`[ClusteredVectorDB] Save already in progress, waiting...`);
      return this.savePromise;
    }

    this.savePromise = (async () => {
      try {
        // 1. Save base data (vectors, metadata) using parent method
        await super.save(); // This handles its own file paths and logic
        console.log('[ClusteredVectorDB] Base VectorDB data saved.');

        // 2. Prepare cluster state for serialization
        const clusterState = {
          version: 1, // Versioning for cluster state format
          clusterIdCounter: this.clusterIdCounter,
          // Convert Maps to structures suitable for JSON
          clusters: Array.from(this.clusters.entries()), // [[key1, members1], [key2, members2]]
          // Convert Float32Arrays in centroids to regular arrays for JSON
          clusterCentroids: Array.from(this.clusterCentroids.entries()).map(([key, centroid]) => [key, Array.from(centroid)]),
          clusterDimensions: Array.from(this.clusterDimensions.entries()), // [[key1, dim1], ...]
        };

        // 3. Save cluster state to its own file
        const clusterFilePath = this._getClusterStateFilePath();
        console.log(`[ClusteredVectorDB] Saving cluster state to: ${clusterFilePath}`);
        let clusterContent: string | Buffer = JSON.stringify(clusterState);
        if (this.useCompression) {
          clusterContent = await gzip(clusterContent);
        }
        await fsPromises.writeFile(clusterFilePath, clusterContent);
        console.log('[ClusteredVectorDB] Cluster state saved successfully.');

        // Emit the save event (perhaps redundant if parent emits, decide based on needs)
        // this.emit('db:save', { path: this.dbPath, count: this.memoryStorage.size });
      } catch (error) {
        console.error(`[ClusteredVectorDB] Error saving database state to ${this.dbPath}:`, error);
        throw error; // Re-throw to indicate failure
      } finally {
        this.savePromise = null; // Release lock
      }
    })();

    return this.savePromise;
  }

  // --- Overridden Load Method ---
  override async load(): Promise<void> {
    if (!this.dbPath) {
      throw new Error('[ClusteredVectorDB] Database path not specified for loading.');
    }
    if (this.isClosed) {
      throw new Error('[ClusteredVectorDB] Cannot load into a closed database.');
    }

    console.log(`[ClusteredVectorDB] Loading state from ${this.dbPath}`);

    // 1. Load base data (vectors, metadata) using parent method
    await super.load(); // This handles its own file paths and logic
    console.log('[ClusteredVectorDB] Base VectorDB data loaded.');

    // 2. Load cluster state if the file exists
    const clusterFilePath = this._getClusterStateFilePath();
    let clusterStateLoaded = false;

    if (existsSync(clusterFilePath)) {
      console.log(`[ClusteredVectorDB] Loading cluster state from: ${clusterFilePath}`);
      try {
        let clusterContentBuffer = await fsPromises.readFile(clusterFilePath);
        if (this.useCompression) {
          clusterContentBuffer = await gunzip(clusterContentBuffer);
        }
        const clusterState = JSON.parse(clusterContentBuffer.toString('utf8'));

        if (clusterState.version !== 1) {
          throw new Error(`Unsupported cluster state format version: ${clusterState.version}`);
        }

        // 3. Restore cluster state from loaded data
        this.clusterIdCounter = clusterState.clusterIdCounter ?? 0;
        this.clusters = new Map(clusterState.clusters);
        // Convert centroid arrays back to Float32Arrays
        this.clusterCentroids = new Map(clusterState.clusterCentroids.map(([key, centroidArray]: [number, number[]]) => [key, new Float32Array(centroidArray)]));
        this.clusterDimensions = new Map(clusterState.clusterDimensions);

        console.log(`[ClusteredVectorDB] Cluster state loaded successfully (${this.clusterCentroids.size} clusters).`);
        clusterStateLoaded = true;
      } catch (error) {
        console.error(`[ClusteredVectorDB] Error loading cluster state from ${clusterFilePath}, will rebuild clusters:`, error);
        // Reset cluster structures before rebuilding
        this.clusters.clear();
        this.clusterCentroids.clear();
        this.clusterDimensions.clear();
        this.clusterIdCounter = 0;
      }
    } else {
      console.log('[ClusteredVectorDB] Cluster state file not found. Will rebuild clusters if vectors were loaded.');
    }

    // 4. Rebuild clusters or run K-Means if needed
    if (!clusterStateLoaded && this.memoryStorage.size > 0) {
      if (this.runKMeansOnLoad) {
        console.log('[ClusteredVectorDB] Running K-Means after load (cluster state missing/invalid)...');
        await this.runKMeans(); // Run K-Means with default settings
      } else {
        console.log('[ClusteredVectorDB] Rebuilding clusters incrementally after load (cluster state missing/invalid)...');
        this._rebuildAllClusters(); // Fallback to incremental rebuild
        console.log(`[ClusteredVectorDB] Rebuilt ${this.clusterCentroids.size} clusters incrementally.`);
      }
    }
  }

  getDistanceMetric(): DistanceMetric {
    return this.distanceMetric;
  }
  // --- Overridden Methods ---

  override addVector(id: number | string | undefined, vector: Vector, metadata?: Record<string, any>): number | string {
    const vectorId = super.addVector(id, vector, metadata); // Let parent handle storage

    const typedVector = this.memoryStorage.get(vectorId);
    if (!typedVector) return vectorId; // Should not happen

    this._assignVectorToCluster(vectorId, typedVector);

    return vectorId;
  }

  override deleteVector(id: number | string): boolean {
    const vector = this.memoryStorage.get(id); // Get vector before deleting
    const deleted = super.deleteVector(id); // Let parent handle deletion

    if (deleted && vector) {
      this._removeVectorFromCluster(id);
    }

    return deleted;
  }

  override updateVector(id: number | string, vector: Vector): boolean {
    const oldVector = this.memoryStorage.get(id);
    if (!oldVector) {
      console.warn(`Attempted to update non-existent vector ID: ${id}`);
      return false;
    }

    const deleted = super.deleteVector(id);
    if (!deleted) {
      console.warn(`Attempted to update non-existent vector ID: ${id}`);
      return false;
    }

    const vectorId = super.addVector(id, vector); // Let parent handle storage

    const typedVector = this.memoryStorage.get(vectorId);
    if (!typedVector) return false; // Should not happen

    this._assignVectorToCluster(vectorId, typedVector);

    return true;
  }

  override findNearest(
    query: Vector,
    k: number = 10,
    options: {
      filter?: (id: number | string, metadata?: Record<string, any>) => boolean;
      metric?: DistanceMetric;
    } = {}
  ): SearchResult[] {
    console.log(`[ClusteredVectorDB] [findNearest] Searching for nearest vectors... with k=${k}}`);

    const typedQuery = query instanceof Float32Array ? query : new Float32Array(query);
    const metric = options.metric ?? this.distanceMetric; // Use instance default or override
    const filter = options.filter;

    // Fallback to linear search if no clusters exist
    if (this.clusterCentroids.size === 0) {
      console.warn('No clusters found, falling back to linear search.');
      return this._linearSearch(typedQuery, k, metric, filter);
    }

    const queryDim = typedQuery.length;

    // 1. Find candidate clusters
    const clusterDistances: Array<{ key: number; dist: number }> = [];
    for (const [key, centroid] of this.clusterCentroids.entries()) {
      // Check dimension compatibility *before* calculating distance if metric requires it
      const centroidDim = this.clusterDimensions.get(key);
      if (metric === 'cosine' && centroidDim !== queryDim) {
        continue; // Skip incompatible dimensions for cosine
      }
      // Euclidean can handle mismatch (though results might be less meaningful)

      const dist = this._calculateDistance(typedQuery, centroid, metric);
      clusterDistances.push({ key, dist });
    }

    if (clusterDistances.length === 0) {
      // No compatible clusters found (e.g., cosine search with wrong dimension)
      return [];
    }

    const clustersToSearch = clusterDistances.sort((a, b) => a.dist - b.dist);
    console.log(`[ClusteredVectorDB] [findNearest] Found ${clustersToSearch.length} candidate clusters.`);
    
    // 2. Collect candidate vectors from selected clusters
    const candidateIds = new Set<number | string>();
    for (const { key } of clustersToSearch) {
      const clusterMembers = this.clusters.get(key) || [];
      for (const member of clusterMembers) {
        candidateIds.add(member.id);
      }
    }

    // 3. Perform exact search on candidates
    const results: SearchResult[] = [];
    for (const id of candidateIds) {
      const vector = this.memoryStorage.get(id);
      if (!vector) continue; // Should not happen if cluster list is sync'd

      // Apply filter if provided
      if (filter) {
        const meta = this.metadata.get(id);
        if (!filter(id, meta)) {
          continue;
        }
      }

      // Double-check dimension compatibility for the specific metric
      if (metric === 'cosine' && vector.length !== queryDim) {
        continue;
      }

      const dist = this._calculateDistance(typedQuery, vector, metric);
      results.push({ id, dist });
    }
    console.log(`[ClusteredVectorDB] [findNearest] Found ${results.length} candidates.`);

    // 4. Sort final results and return top k
    return results.sort((a, b) => a.dist - b.dist).slice(0, k);
  }

  // --- Clustering Logic ---

  private _assignVectorToCluster(vectorId: number | string, vector: Float32Array): void {
    const vectorDim = vector.length;

    // Handle the very first vector
    if (this.clusterCentroids.size === 0) {
      this._createNewCluster(vectorId, vector);
      return;
    }

    // Find the best cluster (considering dimensions and distance)
    let bestClusterKey: number | null = null;
    let minDist = Infinity;

    for (const [key, centroid] of this.clusterCentroids.entries()) {
      const clusterDim = this.clusterDimensions.get(key);

      // Strict dimension check for cosine, optional for Euclidean (centroids should ideally match vector dims)
      if (this.distanceMetric === 'cosine' && clusterDim !== vectorDim) {
        continue;
      }
      // Could add a check here for Euclidean too if strict dimension matching per cluster is desired

      const dist = this._calculateDistance(vector, centroid, this.distanceMetric);
      if (dist < minDist) {
        minDist = dist;
        bestClusterKey = key;
      }
    }

    // Decide whether to create a new cluster or add to the best existing one
    let assignedKey: number;
    if (bestClusterKey !== null && this.clusterCentroids.size < this.maxClusters) {
      const clusterMembers = this.clusters.get(bestClusterKey) || [];
      const needsNewCluster =
        // Reason 1: Cluster is getting too large
        clusterMembers.length >= this.targetClusterSize * this.newClusterThresholdFactor ||
        // Reason 2: Vector is too far from the closest centroid
        minDist > this.newClusterDistanceThreshold;

      if (!needsNewCluster) {
        assignedKey = bestClusterKey;
      } else {
        // Create a new cluster if conditions met
        assignedKey = this._createNewCluster(vectorId, vector);
        // Don't add to the list below, it's done in _createNewCluster
        return; // Exit early as it's handled
      }
    } else {
      // No suitable existing cluster found, or max clusters reached, or first vector for this dimension
      assignedKey = this._createNewCluster(vectorId, vector);
      // Don't add to the list below, it's done in _createNewCluster
      return; // Exit early as it's handled
    }

    // Add to the chosen existing cluster
    const clusterMembers = this.clusters.get(assignedKey);
    if (clusterMembers) {
      // Should always exist if assignedKey is from existing
      clusterMembers.push({ id: vectorId });
      // Update centroid incrementally (more efficient than recalculating)
      this._updateCentroidIncrementally(assignedKey, vector, 'add');
    } else {
      // This case should ideally not be reached if logic above is correct
      console.error(`Cluster ${assignedKey} not found when trying to add vector ${vectorId}`);
      // Fallback: create cluster anyway?
      assignedKey = this._createNewCluster(vectorId, vector);
    }
  }

  private _createNewCluster(initialVectorId: number | string, initialVector: Float32Array): number {
    const newKey = this.clusterIdCounter++;
    this.clusters.set(newKey, [{ id: initialVectorId }]); // Store only ID
    // Centroid starts as the first vector in the cluster
    this.clusterCentroids.set(newKey, initialVector.slice()); // Use slice to copy
    this.clusterDimensions.set(newKey, initialVector.length);
    this.emit('cluster:create', {
      clusterId: newKey,
      vectorId: initialVectorId,
    });
    return newKey;
  }

  private _removeVectorFromCluster(vectorId: number | string): void {
    let foundClusterKey: number | null = null;
    let indexToRemove: number | null = null;

    // Find the cluster containing the vector
    for (const [key, members] of this.clusters.entries()) {
      const index = members.findIndex((m) => m.id === vectorId);
      if (index !== -1) {
        foundClusterKey = key;
        indexToRemove = index;
        break;
      }
    }

    if (foundClusterKey !== null && indexToRemove !== null) {
      const members = this.clusters.get(foundClusterKey)!;
      const vectorToRemove = this.memoryStorage.get(vectorId) ?? null; // Get vector data for centroid update

      // Remove from member list
      members.splice(indexToRemove, 1);

      // Update centroid or remove cluster if empty
      if (members.length > 0 && vectorToRemove) {
        // Update centroid incrementally
        this._updateCentroidIncrementally(foundClusterKey, vectorToRemove, 'remove');
      } else {
        // Cluster is now empty, remove it
        this.clusters.delete(foundClusterKey);
        this.clusterCentroids.delete(foundClusterKey);
        this.clusterDimensions.delete(foundClusterKey);
        this.emit('cluster:delete', { clusterId: foundClusterKey });
      }
    } else {
      console.warn(`Vector ${vectorId} not found in any cluster during deletion.`);
    }
  }

  // More efficient centroid update without iterating all members
  private _updateCentroidIncrementally(clusterKey: number, vector: Float32Array, operation: 'add' | 'remove'): void {
    const centroid = this.clusterCentroids.get(clusterKey);
    const members = this.clusters.get(clusterKey);

    if (!centroid || !members) {
      console.error(`Cannot update centroid for non-existent cluster ${clusterKey}`);
      return;
    }
    if (centroid.length !== vector.length) {
      console.error(`Dimension mismatch during incremental centroid update for cluster ${clusterKey}`);
      // Maybe trigger full rebuild for this cluster?
      this._recalculateCentroid(clusterKey); // Fallback to full recalc
      return;
    }

    const currentSize = operation === 'add' ? members.length - 1 : members.length + 1;
    const newSize = members.length;

    if (newSize === 0 || currentSize < 0) {
      // This should be handled by cluster deletion logic, but as a safeguard:
      if (newSize === 0) {
        this.clusters.delete(clusterKey);
        this.clusterCentroids.delete(clusterKey);
        this.clusterDimensions.delete(clusterKey);
      }
      return;
    }

    if (operation === 'add') {
      // new_centroid = (old_centroid * old_size + new_vector) / new_size
      for (let i = 0; i < centroid.length; i++) {
        centroid[i] = (centroid[i] * currentSize + vector[i]) / newSize;
      }
    } else {
      // operation === 'remove'
      // new_centroid = (old_centroid * old_size - removed_vector) / new_size
      for (let i = 0; i < centroid.length; i++) {
        centroid[i] = (centroid[i] * currentSize - vector[i]) / newSize;
      }
    }

    // No need to set back into map as we modified the array in place
    // this.clusterCentroids.set(clusterKey, centroid);
  }

  // Fallback centroid calculation
  private _recalculateCentroid(clusterKey: number): void {
    const members = this.clusters.get(clusterKey);
    if (!members || members.length === 0) {
      // Remove cluster if empty during recalculation
      this.clusters.delete(clusterKey);
      this.clusterCentroids.delete(clusterKey);
      this.clusterDimensions.delete(clusterKey);
      return;
    }

    let firstVector: Float32Array | null = null;
    const memberVectors: Float32Array[] = [];

    // Gather vectors (inefficient, use only as fallback)
    for (const member of members) {
      const vec = this.memoryStorage.get(member.id);
      if (vec) {
        if (!firstVector) firstVector = vec;
        memberVectors.push(vec);
      } else {
        console.warn(`Vector ${member.id} not found in memoryStorage during centroid recalc for cluster ${clusterKey}.`);
      }
    }

    if (!firstVector || memberVectors.length === 0) {
      // Cluster effectively empty
      this.clusters.delete(clusterKey);
      this.clusterCentroids.delete(clusterKey);
      this.clusterDimensions.delete(clusterKey);
      return;
    }

    const dimensions = firstVector.length;
    const centroid = new Float32Array(dimensions);

    // Calculate sum
    for (const vector of memberVectors) {
      if (vector.length !== dimensions) {
        console.error(`Inconsistent dimensions within cluster ${clusterKey} during recalc. Expected ${dimensions}, got ${vector.length}`);
        // How to handle? Skip vector? Abort?
        continue;
      }
      for (let i = 0; i < dimensions; i++) {
        centroid[i] += vector[i];
      }
    }

    // Calculate average
    const count = memberVectors.length;
    if (count > 0) {
      for (let i = 0; i < dimensions; i++) {
        centroid[i] /= count;
      }
    }

    this.clusterCentroids.set(clusterKey, centroid);
    this.clusterDimensions.set(clusterKey, dimensions); // Ensure dimension is correct
  }

  // Method to rebuild all clusters from scratch (e.g., after loading)
  private _rebuildAllClusters(): void {
    this.clusters.clear();
    this.clusterCentroids.clear();
    this.clusterDimensions.clear();
    this.clusterIdCounter = 0;

    // Iterate through all vectors in memory storage and re-assign them
    for (const [id, vector] of this.memoryStorage.entries()) {
      // Use the assignment logic, which handles creating new clusters
      // This is less efficient than a bulk k-means, but reuses existing logic
      this._assignVectorToCluster(id, vector);
    }
  }

  // --- K-Means Implementation ---

  /**
   * Runs the K-Means clustering algorithm to potentially improve cluster quality.
   * This is computationally more expensive than incremental updates.
   *
   * @param k - The target number of clusters. Defaults to the current number of clusters or a minimum of 1.
   * @param maxIterations - Maximum number of iterations for the algorithm. Defaults to instance configuration.
   * @returns A promise that resolves when K-Means completes.
   */
  async runKMeans(k?: number, maxIterations?: number): Promise<void> {
    if (this.memoryStorage.size === 0) {
      console.log('[ClusteredVectorDB] Skipping K-Means: No vectors in the database.');
      return;
    }

    const targetK = k ?? Math.max(1, this.clusterCentroids.size); // Default to current cluster count or 1
    const iterations = maxIterations ?? this.kmeansMaxIterations;

    console.log(`[ClusteredVectorDB] Starting K-Means with k=${targetK}, maxIterations=${iterations}...`);
    this.emit('kmeans:start', { k: targetK, iterations }); // Emit start event
    const startTime = Date.now();

    try {
      const vectors = Array.from(this.memoryStorage.values());
      this.kmeans = new KMeans(targetK, iterations);
      const centroids = await this.kmeans.cluster(vectors);
      this._updateClustersFromKMeans(Array.from(this.memoryStorage.entries()), new Map<number | string, number>(), centroids);
      const duration = Date.now() - startTime;
      console.log(`[ClusteredVectorDB] K-Means finished in ${duration}ms. New cluster count: ${this.clusterCentroids.size}`);
      this.emit('kmeans:complete', { k: this.clusterCentroids.size, iterations });
    } catch (error) {
      console.error('[ClusteredVectorDB] Error during K-Means execution:', error);
      this.emit('kmeans:error', { error });
      // Optionally re-throw or handle the error
    }
  }

  private _updateClustersFromKMeans(
    allVectors: [number | string, Float32Array][],
    assignments: Map<number | string, number>, // vectorId -> centroidIndex
    finalCentroids: Float32Array[]
  ): void {
    // Clear existing cluster structures
    this.clusters.clear();
    this.clusterCentroids.clear();
    this.clusterDimensions.clear();
    this.clusterIdCounter = 0; // Reset counter, new keys will be assigned

    const centroidIndexToClusterKey: Map<number, number> = new Map();

    // Create new cluster structures based on final centroids
    for (let i = 0; i < finalCentroids.length; i++) {
      const centroid = finalCentroids[i];
      const newKey = this.clusterIdCounter++;
      centroidIndexToClusterKey.set(i, newKey); // Map K-Means index to new DB cluster key

      this.clusters.set(newKey, []); // Initialize empty member list { id: vectorId }[]
      this.clusterCentroids.set(newKey, centroid); // Already a copy
      this.clusterDimensions.set(newKey, centroid.length);
    }

    // Populate the member lists based on assignments
    for (const [vectorId, vector] of allVectors) {
      let bestCentroidIndex = -1;
      let minDist = Infinity;

      for (let i = 0; i < finalCentroids.length; i++) {
        const centroid = finalCentroids[i];
        // Ensure dimension compatibility if needed by metric
        if (this.distanceMetric === 'cosine' && vector.length !== centroid.length) {
          continue;
        }
        const dist = this._calculateDistance(vector, centroid, this.distanceMetric);
        if (dist < minDist) {
          minDist = dist;
          bestCentroidIndex = i;
        }
      }
      const centroidIndex = bestCentroidIndex;
      if (centroidIndex !== undefined && centroidIndex !== -1) {
        const clusterKey = centroidIndexToClusterKey.get(centroidIndex);
        if (clusterKey !== undefined) {
          const members = this.clusters.get(clusterKey);
          members?.push({ id: vectorId }); // Add vector ID object
        } else {
          // Should not happen if mapping is correct
          console.warn(`[ClusteredVectorDB] K-Means Update: Could not find cluster key for centroid index ${centroidIndex}`);
        }
      } else {
        // Vector wasn't assigned (e.g., dimension mismatch)
        console.warn(`[ClusteredVectorDB] K-Means Update: Vector ${vectorId} has no assignment.`);
        // Decide how to handle unassigned vectors: create separate cluster? Ignore?
      }
    }

    // Optional: Clean up any clusters that ended up empty despite having a centroid
    const keysToDelete: number[] = [];
    for (const [key, members] of this.clusters.entries()) {
      if (members.length === 0) {
        keysToDelete.push(key);
      }
    }
    for (const key of keysToDelete) {
      this.clusters.delete(key);
      this.clusterCentroids.delete(key);
      this.clusterDimensions.delete(key);
      console.log(`[ClusteredVectorDB] K-Means Update: Removed empty cluster ${key}.`);
    }
  }

  // --- Stats (Override) ---

  override getStats(): DBStats {
    const baseStats = super.getStats(); // Get stats from VectorDB

    const clusterSizes: Record<number, number> = {};
    let totalVectorsInClusters = 0;
    this.clusters.forEach((members, key) => {
      clusterSizes[key] = members.length;
      totalVectorsInClusters += members.length;
    });

    const clusterDims: Record<number, number> = {};
    this.clusterDimensions.forEach((dim, key) => {
      clusterDims[key] = dim;
    });

    baseStats.clusters = {
      count: this.clusterCentroids.size,
      avgSize: this.clusterCentroids.size > 0 ? totalVectorsInClusters / this.clusterCentroids.size : 0,
      dimensions: clusterDims, // Store dimension per cluster key
      distribution: Object.entries(clusterSizes).map(([keyStr, size]) => {
        const key = parseInt(keyStr, 10); // Ensure key is number
        const centroid = this.clusterCentroids.get(key);
        const members = this.clusters.get(key) || []; // Get members for this cluster
        return {
          id: key, // Cluster ID
          size,
          dimension: this.clusterDimensions.get(key) || 0, // Get stored dimension
          // Calculate norm only if centroid exists
          centroidNorm: centroid ? this._calculateNorm(centroid) : 0,
          members: members, // Add the list of members (vector IDs)
        };
      }),
    };

    // Add clustering overhead to memory estimate
    let clusterOverhead = 0;
    this.clusterCentroids.forEach((c) => (clusterOverhead += c.byteLength)); // Centroid memory
    clusterOverhead += this.clusters.size * 16; // Map overhead
    clusterOverhead += this.clusterDimensions.size * 8; // Map overhead
    // Estimate overhead for member lists (crude: assume ~8 bytes per ID reference)
    this.clusters.forEach((m) => (clusterOverhead += m.length * 8));

    baseStats.memoryUsage = (baseStats.memoryUsage ?? 0) + clusterOverhead;

    return baseStats;
  }

  override async close(): Promise<void> {
    await super.close(); // Call parent close (saves data, clears base maps)
    // Parent clear methods already handle memoryStorage, metadata, vectorDimensions

    // Clear clustering structures
    this.clusters.clear();
    this.clusterCentroids.clear();
    this.clusterDimensions.clear();
    // No need to emit 'db:close' again, parent does it
  }

  // --- Public Cluster Info Method ---
  getClusterInfo(): Array<{
    id: number;
    centroid: Float32Array;
    size: number;
    dimension: number;
  }> {
    const result = [];
    for (const [key, centroid] of this.clusterCentroids.entries()) {
      const size = this.clusters.get(key)?.length ?? 0;
      const dimension = this.clusterDimensions.get(key) ?? centroid.length; // Use stored dim or calculate
      result.push({ id: key, centroid, size, dimension });
    }
    return result;
  }
  /**
 * Extract relationships between vectors based on distance or custom criteria.
 *
 * @param threshold - The maximum distance between vectors to consider them related.
 * @param metric - Distance metric to use (e.g., 'cosine', 'euclidean').
 * @returns An array of relationships, where each relationship links two vector IDs, their distance, and optional metadata.
 */
public extractRelationships(
  threshold: number,
  metric: DistanceMetric = this.distanceMetric
): Array<{ 
  vector1: number | string; 
  vector2: number | string; 
  distance: number;
  metadata1?: Record<string, any>;
  metadata2?: Record<string, any>;
}> {
  const relationships: Array<{ 
    vector1: number | string; 
    vector2: number | string; 
    distance: number;
    metadata1?: Record<string, any>;
    metadata2?: Record<string, any>;
  }> = [];

  // Iterate over all vectors
  const vectorEntries = Array.from(this.memoryStorage.entries());
  for (let i = 0; i < vectorEntries.length; i++) {
    const [id1, vector1] = vectorEntries[i];

    for (let j = i + 1; j < vectorEntries.length; j++) {
      const [id2, vector2] = vectorEntries[j];

      // Ensure dimension compatibility
      if (vector1.length !== vector2.length) {
        console.warn(`Dimension mismatch between vector ${id1} and ${id2}, skipping.`);
        continue;
      }

      // Calculate distance
      const distance = this._calculateDistance(vector1, vector2, metric);

      // Check if the distance is within the threshold
      if (distance <= threshold) {
        // Get metadata for both vectors if available
        const metadata1 = this.metadata.get(id1);
        const metadata2 = this.metadata.get(id2);
        
        relationships.push({ 
          vector1: id1, 
          vector2: id2, 
          distance,
          metadata1: metadata1 ? { ...metadata1 } : undefined,
          metadata2: metadata2 ? { ...metadata2 } : undefined 
        });
      }
    }
  }

  console.log(`[ClusteredVectorDB] Extracted ${relationships.length} relationships.`);
  return relationships;
}
/**
   * Extract communities of related vectors based on distance threshold.
   * Uses cluster information to optimize the community detection process.
   * 
   * @param threshold - The maximum distance between vectors to consider them related
   * @param metric - Distance metric to use (e.g., 'cosine', 'euclidean')
   * @returns Array of communities, where each community is an array of related vector information
   */
  override extractCommunities(
    threshold: number,
    metric: DistanceMetric = this.distanceMetric
  ): Array<Array<{
    id: number | string;
    metadata?: Record<string, any>;
  }>> {
    console.log(`[ClusteredVectorDB] Extracting vector communities with threshold ${threshold}...`);
    
    // We can optimize by first checking distances between cluster centroids
    // Only compare vectors in clusters whose centroids are within (2 * threshold) distance
    // This is an approximation that works because of the triangle inequality property
    
    const clusterAdjacency = new Map<number, Set<number>>();
    
    // Build cluster adjacency graph
    for (const [keyA, centroidA] of this.clusterCentroids.entries()) {
      clusterAdjacency.set(keyA, new Set());
      
      for (const [keyB, centroidB] of this.clusterCentroids.entries()) {
        if (keyA === keyB) continue; // Skip self
        
        // Skip if dimension mismatch for cosine
        if (metric === 'cosine' && centroidA.length !== centroidB.length) {
          continue;
        }
        
        // Calculate inter-cluster distance
        const distance = this._calculateDistance(centroidA, centroidB, metric);
        
        // Use 2*threshold as a conservative bound due to triangle inequality
        if (distance <= 2 * threshold) {
          clusterAdjacency.get(keyA)?.add(keyB);
        }
      }
    }
    
    // Build the vector graph, but only consider vectors in nearby clusters
    const graph = new Map<number | string, Set<number | string>>();
    
    // Initialize graph with empty adjacency lists
    for (const [id] of this.memoryStorage.entries()) {
      graph.set(id, new Set());
    }
    
    // For each cluster
    for (const [clusterKey, members] of this.clusters.entries()) {
      const relatedClusters = new Set([clusterKey, ...(clusterAdjacency.get(clusterKey) || [])]);
      
      // Get all vectors in this cluster
      const clusterVectors = members.map(m => m.id);
      
      // For each vector in this cluster
      for (const vectorId of clusterVectors) {
        const vector = this.memoryStorage.get(vectorId);
        if (!vector) continue;
        
        // Compare with vectors in related clusters
        for (const relatedClusterKey of relatedClusters) {
          const relatedMembers = this.clusters.get(relatedClusterKey) || [];
          
          for (const relatedMember of relatedMembers) {
            const relatedId = relatedMember.id;
            
            // Skip self comparison
            if (vectorId === relatedId) continue;
            
            // Skip if already checked (undirected graph)
            if (graph.get(vectorId)?.has(relatedId)) continue;
            
            const relatedVector = this.memoryStorage.get(relatedId);
            if (!relatedVector) continue;
            
            // Ensure dimension compatibility
            if (vector.length !== relatedVector.length) {
              continue;
            }
            
            // Calculate distance
            const distance = this._calculateDistance(vector, relatedVector, metric);
            
            // Add edge if distance is within threshold
            if (distance <= threshold) {
              graph.get(vectorId)?.add(relatedId);
              graph.get(relatedId)?.add(vectorId);
            }
          }
        }
      }
    }
    
    // Use depth-first search to find connected components (communities)
    const visited = new Set<number | string>();
    const communities: Array<Array<{
      id: number | string;
      metadata?: Record<string, any>;
    }>> = [];
    
    for (const [id] of graph.entries()) {
      if (!visited.has(id)) {
        const community: Array<{
          id: number | string;
          metadata?: Record<string, any>;
        }> = [];
        
        // DFS to find all connected vectors
        const dfs = (nodeId: number | string) => {
          visited.add(nodeId);
          const metadata = this.metadata.get(nodeId);
          community.push({
            id: nodeId,
            metadata: metadata ? { ...metadata } : undefined
          });
          
          // Visit all neighbors
          const neighbors = graph.get(nodeId) || new Set();
          for (const neighbor of neighbors) {
            if (!visited.has(neighbor)) {
              dfs(neighbor);
            }
          }
        };
        
        dfs(id);
        
        // Only include communities with at least 2 vectors
        if (community.length > 1) {
          communities.push(community);
        }
      }
    }
    
    console.log(`[ClusteredVectorDB] Found ${communities.length} communities`);
    return communities;
  }
}
// --- END OF FILE clustered_vector_db.ts ---
