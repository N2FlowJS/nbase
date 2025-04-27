import { promises as fs } from "fs";
import config from "../config";
import {
  BuildIndexOptions,
  LoadIndexOptions,
  LSHOptions,
  SearchResult,
  Vector,
} from "../types";
import { ClusteredVectorDB } from "../vector/clustered_vector_db";
import { log } from "../utils/log";

/**
 * Locality-Sensitive Hashing (LSH) implementation
 * Used for approximate nearest neighbor search by hashing similar vectors
 * to the same buckets with high probability.
 * Supports vectors of different dimensions.
 */
/**
 * Locality-Sensitive Hashing (LSH) implementation for approximate nearest neighbor search.
 * 
 * LSH accelerates vector similarity search by hashing similar vectors into the same buckets.
 * This implementation uses random hyperplanes to partition the vector space, supporting:
 * - Multi-dimensional vectors (vectors of different sizes)
 * - Multi-probing to increase recall
 * - Automatic index building
 * - Serialization for persistence
 * 
 * @example
 * ```typescript
 * // Create a new LSH index
 * const lsh = new LSH(vectorDB, {
 *   dimensions: 1536,
 *   numberOfHashes: 8, 
 *   numberOfBuckets: 150
 * });
 * 
 * // Build the index
 * await lsh.buildIndex({
 *   progressCallback: (progress) => console.log(`Indexing: ${progress * 100}%`)
 * });
 * 
 * // Query for nearest neighbors
 * const results = lsh.findNearest(queryVector, 10);
 * ```
 * 
 * @remarks
 * The implementation uses random hyperplane hashing, where vectors are assigned to buckets
 * based on which side of random hyperplanes they fall. Vectors that are close to each other
 * in the original space have a higher probability of being assigned to the same bucket.
 * 
 * For improved recall, consider using multi-probing which checks neighboring buckets.
 * For higher precision, increase the number of hash functions (numberOfHashes).
 * For better performance but potentially lower recall, increase numberOfBuckets.
 * 
 * @see {@link BuildIndexOptions} for index building options
 * @see {@link LSHOptions} for constructor options
 */
class LSH {
  private db: ClusteredVectorDB;
  private defaultDimensions: number;
  private numberOfHashes: number;
  private numberOfBuckets: number;
  private hashFunctions: Map<number, Float32Array[][]>;
  private buckets: Map<number, Map<number, number | string[]>[]>;
  private vectorDimensions: Map<number | string, number>;
  private dimensionGroups: Map<number, Set<number | string>>;
  private allowMismatchedDimensions: boolean;
  private initialized: boolean = false;

  constructor(db: ClusteredVectorDB, options: LSHOptions) {
    this.db = db;

    // Set default parameters
    this.defaultDimensions =
      options.dimensions ||
      this.db.vectorSize() ||
      config.defaults.vectorSize ||
      1024;
    this.numberOfHashes = options.numberOfHashes || 10;
    this.numberOfBuckets = options.numberOfBuckets || 100;
    this.allowMismatchedDimensions =
      options.allowMismatchedDimensions !== false;

    // Initialize data structures for multi-dimensional support
    this.hashFunctions = new Map<number, Float32Array[][]>();
    this.buckets = new Map<number, Map<number, number | string[]>[]>();
    this.vectorDimensions = new Map<number | string, number>();
    this.dimensionGroups = new Map<number, Set<number | string>>();

    // Generate hash functions for default dimension
    this._generateHashFunctions(this.defaultDimensions);
  }

  /**
   * Generate random hyperplanes for LSH
   * @param dimension - The vector dimension to generate hash functions for
   */
  private _generateHashFunctions(dimension: number): void {
    // Skip if hash functions already exist for this dimension
    if (this.hashFunctions.has(dimension)) {
      return;
    }

    const hyperplanes: Float32Array[][] = [];

    for (let i = 0; i < this.numberOfHashes; i++) {
      const hashFunctions: Float32Array[] = [];

      for (let j = 0; j < this.numberOfBuckets; j++) {
        // Generate a random hyperplane (normal vector)
        const hyperplane = new Float32Array(dimension);
        for (let d = 0; d < dimension; d++) {
          // Use normal distribution for better results
          hyperplane[d] = this._randomNormal();
        }

        hashFunctions.push(hyperplane);
      }

      hyperplanes.push(hashFunctions);
    }

    this.hashFunctions.set(dimension, hyperplanes);

    // Initialize buckets for this dimension
    const dimensionBuckets = Array(this.numberOfHashes)
      .fill(null)
      .map(() => new Map());
    this.buckets.set(dimension, dimensionBuckets);
  }

  /**
   * Standard normal distribution using Box-Muller transform
   */
  private _randomNormal(): number {
    let u = 0,
      v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  }

  /**
   * Compute hash for a vector
   * @param vector - Vector to hash
   * @returns Array of hash values
   */
  private _hashVector(vector: Vector): { dimension: number; hashes: number[] } {
    const dimension = vector.length;

    // Generate hash functions for this dimension if they don't exist
    if (!this.hashFunctions.has(dimension)) {
      this._generateHashFunctions(dimension);
    }

    const hashFunctions = this.hashFunctions.get(dimension)!;
    const hashes: number[] = [];

    for (let i = 0; i < this.numberOfHashes; i++) {
      const hyperplanes = hashFunctions[i];
      let hash = 0;

      // Compute hash by checking which side of each hyperplane the vector falls on
      for (let j = 0; j < hyperplanes.length && j < 31; j++) {
        const hyperplane = hyperplanes[j];

        // Compute dot product
        let dotProduct = 0;
        for (let d = 0; d < dimension; d++) {
          dotProduct += vector[d] * hyperplane[d];
        }

        // Set the corresponding bit based on sign of dot product
        if (dotProduct >= 0) {
          hash |= 1 << j;
        }
      }

      hashes.push(hash % this.numberOfBuckets);
    }

    return { dimension, hashes };
  }

  /**
   * Index a vector
   * @param id - Vector identifier
   * @param vector - Vector to index
   * @returns Vector ID
   */
  indexVector(id: number | string, vector: Vector): number | string {
    const { dimension, hashes } = this._hashVector(vector);

    // Store dimension information
    this.vectorDimensions.set(id, dimension);

    // Add to dimension group
    if (!this.dimensionGroups.has(dimension)) {
      this.dimensionGroups.set(dimension, new Set());
    }
    this.dimensionGroups.get(dimension)!.add(id);

    // Get buckets for this dimension
    const dimensionBuckets = this.buckets.get(dimension);
    if (!dimensionBuckets) {
      console.warn(`No buckets for dimension ${dimension}`);
      return id;
    }

    // Add to each hash table
    for (let i = 0; i < hashes.length; i++) {
      const hash = hashes[i];
      const bucket = dimensionBuckets[i];

      if (!bucket.has(hash)) {
        bucket.set(hash, []);
      }

      const ids = bucket.get(hash) as (number | string)[];
      ids.push(id);
    }

    return id;
  }

  /**
   * Build index for all vectors in database
   * @param options - Build options
   */
  async buildIndex(options: BuildIndexOptions = {}): Promise<void> {
    const progressCallback = options.progressCallback || (() => {});
    const useDimensionGroups = options.dimensionGroups !== false;

    // Reset all data structures
    this.hashFunctions.clear();
    this.buckets.clear();
    this.vectorDimensions.clear();
    this.dimensionGroups.clear();

    // Get all vectors
    const ids = Array.from(this.db.memoryStorage.keys());
    const totalVectors = ids.length;

    if (totalVectors === 0) {
      console.log("No vectors to index");
      return;
    }

    // Phase 1: Collect dimensions
    progressCallback(0);
    let processedCount = 0;

    for (let i = 0; i < ids.length; i++) {
      const id = ids[i];
      let dimension: number;

      // Try to get dimension from db.getVectorDimension if available (faster)
      if (this.db.getVectorDimension) {
        dimension = this.db.getVectorDimension(id) || 0;
      } else {
        // Fallback to getting dimension from vector
        const vector = this.db.getVector(id);
        dimension = vector ? vector.length : 0;
      }

      if (dimension > 0) {
        // Store dimension info
        this.vectorDimensions.set(id, dimension);

        // Group by dimension
        if (!this.dimensionGroups.has(dimension)) {
          this.dimensionGroups.set(dimension, new Set());
        }
        this.dimensionGroups.get(dimension)!.add(id);
      }

      // Report progress for phase 1 (0-10%)
      if (i % 1000 === 0) {
        progressCallback((i / totalVectors) * 0.1);
      }
    }

    // Initialize hash functions for each dimension
    for (const dimension of this.dimensionGroups.keys()) {
      this._generateHashFunctions(dimension);
    }

    // Phase 2: Index vectors
    processedCount = 0;

    // If using dimension groups, process each group separately
    if (useDimensionGroups) {
      for (const [dimension, idSet] of this.dimensionGroups.entries()) {
        const idsInDimension = Array.from(idSet);
        log('info' , `dimension: ${dimension} in ids`)

        for (let i = 0; i < idsInDimension.length; i++) {
          const id = idsInDimension[i];
          const vector = this.db.getVector(id);

          if (vector) {
            this.indexVector(id, vector);
          }

          processedCount++;

          // Report progress for phase 2 (10-100%)
          if (processedCount % 100 === 0) {
            progressCallback(0.1 + (processedCount / totalVectors) * 0.9);
          }
        }
      }
    } else {
      // Process all vectors regardless of dimension
      for (let i = 0; i < ids.length; i++) {
        const id = ids[i];
        const vector = this.db.getVector(id);

        if (vector) {
          this.indexVector(id, vector);
        }

        // Report progress for phase 2 (10-100%)
        if (i % 100 === 0) {
          progressCallback(0.1 + (i / totalVectors) * 0.9);
        }
      }
    }

    this.initialized = true;
    progressCallback(1.0);
  }

  /**
   * Query for approximate nearest neighbors
   * @param vector - Query vector
   * @param multiProbe - Number of neighboring buckets to check (0 for exact bucket only)
   * @param options - Query options
   * @returns Array of candidate IDs
   */
  query(
    vector: Vector,
    multiProbe: number = 0,
    options: { exactDimensions?: boolean } = {}
  ): (number | string)[] {
    const { dimension, hashes } = this._hashVector(vector);
    const exactDimensions = options.exactDimensions || false;
    const candidateIds = new Set<number | string>();

    // If exact dimensions is true, only query the matching dimension
    if (exactDimensions) {
      // If no buckets for this dimension, return empty results
      if (!this.buckets.has(dimension)) {
        return [];
      }

      const dimensionBuckets = this.buckets.get(dimension)!;

      for (let i = 0; i < hashes.length; i++) {
        const hash = hashes[i];
        const bucket = dimensionBuckets[i];

        // Get exact bucket
        if (bucket.has(hash)) {
          for (const id of bucket.get(hash) as (number | string)[]) {
            candidateIds.add(id);
          }
        }

        // Multi-probe LSH: check neighboring buckets
        if (multiProbe > 0) {
          for (let j = 1; j <= multiProbe; j++) {
            const probeBucket1 = (hash + j) % this.numberOfBuckets;
            const probeBucket2 =
              (hash - j + this.numberOfBuckets) % this.numberOfBuckets;

            if (bucket.has(probeBucket1)) {
              for (const id of bucket.get(probeBucket1) as (
                | number
                | string
              )[]) {
                candidateIds.add(id);
              }
            }

            if (bucket.has(probeBucket2)) {
              for (const id of bucket.get(probeBucket2) as (
                | number
                | string
              )[]) {
                candidateIds.add(id);
              }
            }
          }
        }
      }
    } else {
      // Query all dimensions or matching dimension based on allowMismatchedDimensions
      const dimensionsToQuery = this.allowMismatchedDimensions
        ? Array.from(this.buckets.keys())
        : [dimension];

      for (const dim of dimensionsToQuery) {
        // If no buckets for this dimension, skip
        if (!this.buckets.has(dim)) continue;

        const dimensionBuckets = this.buckets.get(dim)!;

        // Use the hash of the query vector for the default dimension
        // This is a simplification - ideally we'd recompute hashes for each dimension
        for (
          let i = 0;
          i < Math.min(hashes.length, dimensionBuckets.length);
          i++
        ) {
          // Adapt hash to the current dimension's bucket count
          const hash = hashes[i] % this.numberOfBuckets;
          const bucket = dimensionBuckets[i];

          // Get exact bucket
          if (bucket.has(hash)) {
            for (const id of bucket.get(hash) as (number | string)[]) {
              candidateIds.add(id);
            }
          }

          // Multi-probe LSH: check neighboring buckets
          if (multiProbe > 0) {
            for (let j = 1; j <= multiProbe; j++) {
              const probeBucket1 = (hash + j) % this.numberOfBuckets;
              const probeBucket2 =
                (hash - j + this.numberOfBuckets) % this.numberOfBuckets;

              if (bucket.has(probeBucket1)) {
                for (const id of bucket.get(probeBucket1) as (
                  | number
                  | string
                )[]) {
                  candidateIds.add(id);
                }
              }

              if (bucket.has(probeBucket2)) {
                for (const id of bucket.get(probeBucket2) as (
                  | number
                  | string
                )[]) {
                  candidateIds.add(id);
                }
              }
            }
          }
        }
      }
    }

    return Array.from(candidateIds);
  }

  /**
   * Find approximate nearest neighbors by first filtering with LSH
   * then refining with exact distance
   * @param query - Query vector
   * @param k - Number of nearest neighbors to return
   * @param options - Search options
   * @returns Array of search results
   */
  findNearest(
    query: Vector,
    k: number = config.defaults.k,
    options: {
      filter?: (id: number | string) => boolean;
      exactDimensions?: boolean;
    } = {}
  ): SearchResult[] {
    const typedQuery =
      query instanceof Float32Array ? query : new Float32Array(query);
    const filter = options.filter || (() => true);
    const exactDimensions = options.exactDimensions || false;

    // Fall back to linear search if not initialized
    if (!this.initialized) {
      return this._linearSearch(typedQuery, k, options);
    }

    // Get candidate IDs using LSH
    const candidateIds = this.query(typedQuery, 2, { exactDimensions });

    if (candidateIds.length === 0) {
      return [];
    }

    // Compute exact distances for candidates
    const distances: SearchResult[] = [];
    for (const id of candidateIds) {
      // Apply filter
      if (!filter(id)) continue;

      const vector = this.db.getVector(id);
      if (!vector) continue;

      // Skip vectors with different dimensions in exactDimensions mode
      if (exactDimensions && vector.length !== typedQuery.length) continue;

      const dist = this._distance(typedQuery, vector);
      distances.push({ id, dist });
    }

    // Sort by distance and return top k
    return distances.sort((a, b) => a.dist - b.dist).slice(0, k);
  }

  /**
   * Compute Euclidean distance between vectors
   * @private
   */
  private _distance(a: Vector, b: Vector): number {
    const len = Math.min(a.length, b.length);
    let sum = 0;

    // Process 4 elements at a time for better performance
    for (let i = 0; i < len - 3; i += 4) {
      const d1 = a[i] - b[i];
      const d2 = a[i + 1] - b[i + 1];
      const d3 = a[i + 2] - b[i + 2];
      const d4 = a[i + 3] - b[i + 3];

      sum += d1 * d1 + d2 * d2 + d3 * d3 + d4 * d4;
    }

    // Handle remaining elements
    for (let i = len - (len % 4); i < len; i++) {
      const diff = a[i] - b[i];
      sum += diff * diff;
    }

    // Add small penalty for dimension mismatch
    const dimDiff = Math.abs(a.length - b.length);
    if (dimDiff > 0) {
      sum += dimDiff * 0.01;
    }

    return Math.sqrt(sum);
  }

  /**
   * Perform linear search when no index is available
   * @private
   */
  private _linearSearch(
    query: Vector,
    k: number,
    options: {
      filter?: (id: number | string) => boolean;
      exactDimensions?: boolean;
    }
  ): SearchResult[] {
    const results: SearchResult[] = [];
    const filter = options.filter || (() => true);
    const exactDimensions = options.exactDimensions || false;
    const queryDim = query.length;

    for (const [id, vector] of this.db.memoryStorage.entries()) {
      // Skip if filter excludes this ID
      if (!filter(id)) continue;

      // Skip vectors with different dimensions if exactDimensions is true
      if (exactDimensions && vector.length !== queryDim) continue;

      const dist = this._distance(query, vector);
      results.push({ id, dist });
    }

    // Sort by distance and limit to k results
    return results.sort((a, b) => a.dist - b.dist).slice(0, k);
  }

  /**
   * Get index statistics
   * @returns Statistics about the LSH index
   */
  getStats(): Record<string, any> {
    const stats = {
      numberOfHashes: this.numberOfHashes,
      numberOfBuckets: this.numberOfBuckets,
      defaultDimensions: this.defaultDimensions,
      totalItems: 0,
      bucketsUsed: 0,
      avgBucketSize: 0,
      maxBucketSize: 0,
      vectorsPerDimension: {} as Record<number, number>,
      bucketsPerDimension: {} as Record<number, number>,
      initialized: this.initialized,
      allowMismatchedDimensions: this.allowMismatchedDimensions,
    };

    // Count vectors by dimension
    for (const [dimension, vectors] of this.dimensionGroups.entries()) {
      stats.vectorsPerDimension[dimension] = vectors.size;
    }

    // Count buckets and calculate statistics per dimension
    for (const [dimension, dimensionBuckets] of this.buckets.entries()) {
      let dimBucketsUsed = 0;
      let dimTotalItems = 0;
      let dimMaxBucketSize = 0;

      for (const bucket of dimensionBuckets) {
        dimBucketsUsed += bucket.size;

        for (const items of bucket.values()) {
          if (Array.isArray(items)) {
            dimTotalItems += items.length;
            dimMaxBucketSize = Math.max(dimMaxBucketSize, items.length);
          }
        }
      }

      // Add to overall stats
      stats.bucketsUsed += dimBucketsUsed;
      stats.totalItems += dimTotalItems;
      stats.maxBucketSize = Math.max(stats.maxBucketSize, dimMaxBucketSize);

      // Store dimension-specific stats
      stats.bucketsPerDimension[dimension] = dimBucketsUsed;
    }

    // Calculate average bucket size
    stats.avgBucketSize =
      stats.bucketsUsed > 0 ? stats.totalItems / stats.bucketsUsed : 0;

    return stats;
  }

  /**
   * Serialize the LSH index to JSON
   * @returns Serialized index data
   */
  serialize(): string {
    // Convert Maps to serializable objects
    const hashFunctionsData: Record<number, number[][][]> = {};
    const bucketsData: Record<
      number,
      Record<number, Record<number, (number | string)[]>>
    > = {};
    const vectorDimensionsData: [string | number, number][] = Array.from(
      this.vectorDimensions.entries()
    );

    // Convert hash functions
    for (const [dimension, functions] of this.hashFunctions.entries()) {
      hashFunctionsData[dimension] = functions.map((table) =>
        table.map((hyperplane) => Array.from(hyperplane))
      );
    }

    // Convert buckets
    for (const [dimension, tables] of this.buckets.entries()) {
      bucketsData[dimension] = {};

      tables.forEach((table, tableIndex) => {
        bucketsData[dimension][tableIndex] = {};

        for (const [hash, ids] of table.entries()) {
          bucketsData[dimension][tableIndex][hash] = Array.isArray(ids)
            ? ids
            : [ids];
        }
      });
    }

    const data = {
      defaultDimensions: this.defaultDimensions,
      numberOfHashes: this.numberOfHashes,
      numberOfBuckets: this.numberOfBuckets,
      allowMismatchedDimensions: this.allowMismatchedDimensions,
      hashFunctions: hashFunctionsData,
      buckets: bucketsData,
      vectorDimensions: vectorDimensionsData,
      version: 1, // For future compatibility
    };

    return JSON.stringify(data);
  }

  /**
   * Save index to file
   * @param filePath - Path to save the index
   */
  async saveIndex(filePath: string): Promise<void> {
    const data = this.serialize();
    await fs.writeFile(filePath, data, "utf8");
  }

  /**
   * Load serialized LSH index
   * @param json - Serialized LSH index
   * @param db - Vector database
   * @returns LSH instance
   */
  static deserialize(json: string, db: ClusteredVectorDB): LSH {
    const data = JSON.parse(json);

    // Create LSH instance with basic parameters
    const lsh = new LSH(db, {
      dimensions: data.defaultDimensions,
      numberOfHashes: data.numberOfHashes,
      numberOfBuckets: data.numberOfBuckets,
      allowMismatchedDimensions: data.allowMismatchedDimensions,
    });

    // Restore hash functions
    for (const [dimensionStr, functions] of Object.entries(
      data.hashFunctions
    )) {
      const dimension = parseInt(dimensionStr, 10);
      const typedFunctions: Float32Array[][] = [];

      // Convert arrays to Float32Arrays
      for (const table of functions as number[][][]) {
        const typedTable: Float32Array[] = [];

        for (const hyperplane of table) {
          typedTable.push(new Float32Array(hyperplane));
        }

        typedFunctions.push(typedTable);
      }

      lsh.hashFunctions.set(dimension, typedFunctions);
    }

    // Restore buckets
    for (const [dimensionStr, tables] of Object.entries(data.buckets)) {
      const dimension = parseInt(dimensionStr, 10);
      const dimensionBuckets: Map<number, number | string[]>[] = [];

      for (const [tableIndexStr, hashTable] of Object.entries(
        tables as Record<string, Record<string, (number | string)[]>>
      )) {
        const tableIndex = parseInt(tableIndexStr, 10);
        const bucketMap = new Map<number, number | string[]>();

        // Ensure we have enough tables
        while (dimensionBuckets.length <= tableIndex) {
          dimensionBuckets.push(new Map());
        }

        // Restore each hash bucket
        for (const [hashStr, ids] of Object.entries(hashTable)) {
          const hash = parseInt(hashStr, 10);
          bucketMap.set(hash, ids as string[]);
        }

        dimensionBuckets[tableIndex] = bucketMap;
      }

      lsh.buckets.set(dimension, dimensionBuckets);
    }

    // Restore vector dimensions
    for (const [id, dimension] of data.vectorDimensions) {
      lsh.vectorDimensions.set(id, dimension);

      // Rebuild dimension groups
      if (!lsh.dimensionGroups.has(dimension)) {
        lsh.dimensionGroups.set(dimension, new Set());
      }
      lsh.dimensionGroups.get(dimension)!.add(id);
    }

    lsh.initialized = true;
    return lsh;
  }

  /**
   * Load index from file
   * @param filePath - Path to load the index from
   * @param db - Vector database
   * @param options - Load options
   * @returns LSH instance
   */
  static async loadIndex(
    filePath: string,
    db: ClusteredVectorDB,
    options: LoadIndexOptions = {}
  ): Promise<LSH> {
    const data = await fs.readFile(filePath, "utf8");
    const lsh = LSH.deserialize(data, db);

    // Apply options
    if (options.allowMismatchedDimensions !== undefined) {
      lsh.allowMismatchedDimensions = options.allowMismatchedDimensions;
    }

    return lsh;
  }
}

export default LSH;
