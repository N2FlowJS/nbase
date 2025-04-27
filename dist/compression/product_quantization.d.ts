import { LoadModelOptions, PQOptions, TrainingOptions, Vector } from '../types';
/**
 * Product Quantization implementation for vector compression
 * Supports dynamic vector dimensions for improved flexibility
 */
export declare class ProductQuantization {
    defaultVectorSize: number;
    subvectorSize: number;
    numSubvectors: number;
    numClusters: number;
    trained: boolean;
    dynamicDimensions: boolean;
    minSubquantizers: number;
    centroids: Float32Array[][];
    private dimensionModels;
    private codes;
    private dimensionCodes;
    private vectorDimensions;
    constructor(options?: PQOptions);
    /**
     * Calculate optimal subvector parameters based on vector dimension
     * @param vectorDimension Size of the vectors
     * @returns Optimal subvector configuration
     * @private
     */
    private _calculateOptimalSubvectorParams;
    /**
     * Validate and adjust subvector configuration if necessary
     * @private
     */
    private _validateAndAdjustSubvectorConfig;
    /**
     * Calculate subvector parameters for a specific dimension
     * @param dimension Vector dimension
     * @returns Subvector parameters
     * @private
     */
    private _calculateSubvectorParams;
    /**
     * Train product quantizer on a set of vectors
     * Method overloads to allow more flexible calling patterns
     */
    train(vectors: Vector[], options: TrainingOptions): Promise<void>;
    train(vectors: Vector[], dimension?: number, options?: TrainingOptions): Promise<void>;
    /**
     * Serialize the PQ model to JSON
     * @returns Serialized model
     */
    serialize(): string;
    /**
     * Save the PQ model to a file
     * @param filePath Path to save the model
     */
    saveModel(filePath: string): Promise<void>;
    /**
     * Load a serialized PQ model
     * @param json Serialized model data or path to model file
     * @param options Load options
     * @returns ProductQuantization instance
     */
    static loadModel(json: string, options?: LoadModelOptions): Promise<ProductQuantization>;
    /**
     * Compute distance tables for a query vector
     * This pre-computes distances between query subvectors and centroids
     * for efficient distance approximation
     *
     * @param query The query vector
     * @returns Distance tables as Float32Array[]
     */
    computeDistanceTables(query: Vector): Float32Array[];
    /**
     * Calculate the approximate distance between a query and a PQ code
     * using pre-computed distance tables
     *
     * @param tables Distance tables from computeDistanceTables
     * @param code PQ code representing the vector
     * @returns Approximate distance
     */
    distanceWithTables(tables: Float32Array[], code: Uint8Array): number;
    /**
     * Encode a vector using product quantization
     * @param vector Vector to encode
     * @param dimension Target dimension (optional)
     * @returns Encoded vector as Uint8Array
     */
    encode(vector: Vector, dimension?: number): Uint8Array;
    /**
     * Store encoded vector for an ID
     * @param id Vector ID
     * @param code Encoded vector
     * @param dimension Vector dimension (optional)
     */
    storeCode(id: string | number, code: Uint8Array, dimension?: number): void;
    /**
     * Get code for a vector ID if it exists
     * @param id Vector ID
     * @param dimension Specific dimension to check (optional)
     */
    getCode(id: string | number, dimension?: number): Uint8Array | undefined;
    /**
     * Store vector encodings for fast lookup
     * @param codes Map of codes
     * @param dimension Dimension of these codes (optional)
     */
    setCodes(codes: Map<string | number, Uint8Array> | Record<string, Uint8Array>, dimension?: number): void;
    /**
     * Get compression ratio compared to original vectors
     * @param dimension Vector dimension to calculate for (optional)
     */
    getCompressionRatio(dimension?: number): number;
    /**
     * Get statistics about the product quantization
     * @returns Object with PQ statistics
     */
    getStats(): Record<string, any>;
    /**
     * Get estimated memory usage of compressed vectors and centroids
     * @returns Memory usage in bytes
     */
    getMemoryUsage(): number;
}
