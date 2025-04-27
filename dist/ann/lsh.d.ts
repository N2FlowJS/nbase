import { BuildIndexOptions, LoadIndexOptions, LSHOptions, SearchResult, Vector } from "../types";
import { ClusteredVectorDB } from "../vector/clustered_vector_db";
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
declare class LSH {
    private db;
    private defaultDimensions;
    private numberOfHashes;
    private numberOfBuckets;
    private hashFunctions;
    private buckets;
    private vectorDimensions;
    private dimensionGroups;
    private allowMismatchedDimensions;
    private initialized;
    constructor(db: ClusteredVectorDB, options: LSHOptions);
    /**
     * Generate random hyperplanes for LSH
     * @param dimension - The vector dimension to generate hash functions for
     */
    private _generateHashFunctions;
    /**
     * Standard normal distribution using Box-Muller transform
     */
    private _randomNormal;
    /**
     * Compute hash for a vector
     * @param vector - Vector to hash
     * @returns Array of hash values
     */
    private _hashVector;
    /**
     * Index a vector
     * @param id - Vector identifier
     * @param vector - Vector to index
     * @returns Vector ID
     */
    indexVector(id: number | string, vector: Vector): number | string;
    /**
     * Build index for all vectors in database
     * @param options - Build options
     */
    buildIndex(options?: BuildIndexOptions): Promise<void>;
    /**
     * Query for approximate nearest neighbors
     * @param vector - Query vector
     * @param multiProbe - Number of neighboring buckets to check (0 for exact bucket only)
     * @param options - Query options
     * @returns Array of candidate IDs
     */
    query(vector: Vector, multiProbe?: number, options?: {
        exactDimensions?: boolean;
    }): (number | string)[];
    /**
     * Find approximate nearest neighbors by first filtering with LSH
     * then refining with exact distance
     * @param query - Query vector
     * @param k - Number of nearest neighbors to return
     * @param options - Search options
     * @returns Array of search results
     */
    findNearest(query: Vector, k?: number, options?: {
        filter?: (id: number | string) => boolean;
        exactDimensions?: boolean;
    }): SearchResult[];
    /**
     * Compute Euclidean distance between vectors
     * @private
     */
    private _distance;
    /**
     * Perform linear search when no index is available
     * @private
     */
    private _linearSearch;
    /**
     * Get index statistics
     * @returns Statistics about the LSH index
     */
    getStats(): Record<string, any>;
    /**
     * Serialize the LSH index to JSON
     * @returns Serialized index data
     */
    serialize(): string;
    /**
     * Save index to file
     * @param filePath - Path to save the index
     */
    saveIndex(filePath: string): Promise<void>;
    /**
     * Load serialized LSH index
     * @param json - Serialized LSH index
     * @param db - Vector database
     * @returns LSH instance
     */
    static deserialize(json: string, db: ClusteredVectorDB): LSH;
    /**
     * Load index from file
     * @param filePath - Path to load the index from
     * @param db - Vector database
     * @param options - Load options
     * @returns LSH instance
     */
    static loadIndex(filePath: string, db: ClusteredVectorDB, options?: LoadIndexOptions): Promise<LSH>;
}
export default LSH;
