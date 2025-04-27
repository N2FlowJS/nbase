import { promises as fs } from 'fs';
import config from '../config';
import { LoadModelOptions, PQOptions, TrainingOptions, Vector } from '../types';
import { KMeans } from './kmeans';

/**
 * Product Quantization implementation for vector compression
 * Supports dynamic vector dimensions for improved flexibility
 */
export class ProductQuantization {
  public defaultVectorSize: number;
  public subvectorSize: number;
  public numSubvectors: number;
  public numClusters: number;
  public trained: boolean = false;
  public dynamicDimensions: boolean;
  public minSubquantizers: number; // Minimum number of subquantizers to use

  // Centroids for default dimension
  public centroids: Float32Array[][] = [];

  // Dimension-specific models
  private dimensionModels: Map<
    number,
    {
      centroids: Float32Array[][];
      numSubvectors: number;
      trained: boolean;
    }
  > = new Map();

  // Maps vector IDs to their PQ codes for each dimension
  private codes: Map<string | number, Uint8Array> = new Map();
  private dimensionCodes: Map<number, Map<string | number, Uint8Array>> = new Map();

  // Track vector dimensions
  private vectorDimensions: Map<string | number, number> = new Map();

  constructor(options: PQOptions = {}) {
    this.defaultVectorSize = options.vectorSize || config.defaults.vectorSize || 1024;
    this.dynamicDimensions = options.dynamicDimensions !== false;
    this.minSubquantizers = options.minSubquantizers || 8; // Default to at least 8 subquantizers
    this.numClusters = options.numClusters || 256;

    // Calculate subvector dimensions with automatic optimization
    if (options.numSubvectors && options.subvectorSize) {
      // Both specified - validate and adjust if needed
      this.numSubvectors = options.numSubvectors;
      this.subvectorSize = options.subvectorSize;

      // Verify the configuration is valid
      this._validateAndAdjustSubvectorConfig();
    } else if (options.numSubvectors) {
      // Only numSubvectors specified
      this.numSubvectors = options.numSubvectors;
      this.subvectorSize = Math.ceil(this.defaultVectorSize / this.numSubvectors);

      // Verify and adjust if needed
      this._validateAndAdjustSubvectorConfig();
    } else if (options.subvectorSize) {
      // Only subvectorSize specified
      this.subvectorSize = options.subvectorSize;
      this.numSubvectors = Math.ceil(this.defaultVectorSize / this.subvectorSize);

      // Ensure we have at least minSubquantizers
      if (this.numSubvectors < this.minSubquantizers) {
        this.numSubvectors = this.minSubquantizers;
        this.subvectorSize = Math.ceil(this.defaultVectorSize / this.numSubvectors);
      }
    } else {
      // Auto-calculate optimal values
      const optimal = this._calculateOptimalSubvectorParams(this.defaultVectorSize);
      this.subvectorSize = optimal.subvectorSize;
      this.numSubvectors = optimal.numSubvectors;
    }
  }

  /**
   * Calculate optimal subvector parameters based on vector dimension
   * @param vectorDimension Size of the vectors
   * @returns Optimal subvector configuration
   * @private
   */
  private _calculateOptimalSubvectorParams(vectorDimension: number): { subvectorSize: number; numSubvectors: number } {
    // Rules for determining optimal parameters:
    // 1. For very small vectors (< 32), use fewer subquantizers
    // 2. For medium vectors, aim for 8-32 subquantizers
    // 3. For large vectors (> 512), ensure at least 32 subquantizers

    let numSubvectors: number;

    if (vectorDimension <= 32) {
      // For small vectors: 4-8 subquantizers
      numSubvectors = Math.max(4, Math.min(8, Math.floor(vectorDimension / 4)));
    } else if (vectorDimension <= 256) {
      // For medium vectors: 8-16 subquantizers
      numSubvectors = Math.max(8, Math.min(16, Math.floor(vectorDimension / 16)));
    } else if (vectorDimension <= 768) {
      // For large vectors: 16-32 subquantizers
      numSubvectors = Math.max(16, Math.min(32, Math.floor(vectorDimension / 24)));
    } else {
      // For very large vectors: 32+ subquantizers
      numSubvectors = Math.max(32, Math.floor(vectorDimension / 32));
    }

    // Ensure we have at least the minimum number of subquantizers
    numSubvectors = Math.max(numSubvectors, this.minSubquantizers);

    // Calculate subvector size based on number of subquantizers
    const subvectorSize = Math.ceil(vectorDimension / numSubvectors);

    return { subvectorSize, numSubvectors };
  }

  /**
   * Validate and adjust subvector configuration if necessary
   * @private
   */
  private _validateAndAdjustSubvectorConfig(): void {
    // Check if current configuration satisfies minimum subquantizers
    if (this.numSubvectors < this.minSubquantizers) {
      // Recalculate with minimum subquantizers
      this.numSubvectors = this.minSubquantizers;
      this.subvectorSize = Math.ceil(this.defaultVectorSize / this.numSubvectors);
      console.warn(`Adjusted subvector configuration to ensure at least ${this.minSubquantizers} subquantizers. New config: ${this.numSubvectors} subquantizers of size ${this.subvectorSize}`);
    }

    // Verify total vector size is covered
    const totalCoverage = this.subvectorSize * this.numSubvectors;
    if (totalCoverage < this.defaultVectorSize) {
      // Adjust subvector size to ensure full coverage
      this.subvectorSize = Math.ceil(this.defaultVectorSize / this.numSubvectors);
      console.warn(`Adjusted subvector size to ensure full vector coverage. New subvector size: ${this.subvectorSize}`);
    }
  }

  /**
   * Calculate subvector parameters for a specific dimension
   * @param dimension Vector dimension
   * @returns Subvector parameters
   * @private
   */
  private _calculateSubvectorParams(dimension: number): { subvectorSize: number; numSubvectors: number } {
    if (!this.dynamicDimensions) {
      // For non-dynamic mode, use default parameters
      return {
        subvectorSize: this.subvectorSize,
        numSubvectors: this.numSubvectors,
      };
    }

    // For dynamic mode, calculate optimal parameters
    return this._calculateOptimalSubvectorParams(dimension);
  }

  /**
   * Train product quantizer on a set of vectors
   * Method overloads to allow more flexible calling patterns
   */
  async train(vectors: Vector[], options: TrainingOptions): Promise<void>;
  async train(vectors: Vector[], dimension?: number, options?: TrainingOptions): Promise<void>;
  async train(vectors: Vector[], dimensionOrOptions?: number | TrainingOptions, maybeOptions?: TrainingOptions): Promise<void> {
    // Parse arguments - handle both calling patterns
    let dimension: number | undefined;
    let options: TrainingOptions = {};

    if (typeof dimensionOrOptions === 'number') {
      // Called with train(vectors, dimension, options)
      dimension = dimensionOrOptions;
      options = maybeOptions || {};
    } else if (dimensionOrOptions && typeof dimensionOrOptions === 'object') {
      // Called with train(vectors, options)
      options = dimensionOrOptions;
    }

    const { progressCallback } = options;

    if (vectors.length === 0) {
      throw new Error('Cannot train on empty vector set');
    }

    // Determine vector dimension from training data
    const vectorDim = vectors[0].length;

    // If dimension parameter is specified, we're training a dimension-specific model
    if (dimension !== undefined && this.dynamicDimensions) {
      // Calculate optimal parameters for this dimension
      const params = this._calculateOptimalSubvectorParams(dimension);
      const numSubvectors = params.numSubvectors;
      const subvectorSize = params.subvectorSize;

      console.log(`Training dimension-specific model for dim ${dimension}: using ${numSubvectors} subquantizers of size ${subvectorSize}`);

      // Initialize dimension-specific model
      const centroids = new Array(numSubvectors);

      // Initialize kmeans instance for clustering
      const kmeans = new KMeans(this.numClusters, 100, 0.001);

      try {
        // Train each subquantizer
        for (let m = 0; m < numSubvectors; m++) {
          // Extract subvectors for this subquantizer
          const subvectors: Vector[] = [];
          const start = m * subvectorSize;
          const end = Math.min(start + subvectorSize, dimension);

          for (const vector of vectors) {
            // Extract subvector and add to training set
            const subvector = vector.slice(start, end);
            subvectors.push(subvector);
          }

          // Train k-means on this set of subvectors
          if (progressCallback) {
            progressCallback(m / numSubvectors);
          }

          // Cluster subvectors
          centroids[m] = await kmeans.cluster(subvectors);

          // Allow other operations to proceed
          await new Promise((resolve) => setTimeout(resolve, 0));
        }

        // Only set the trained flag AFTER all subquantizers are trained
        const dimModel = {
          centroids,
          numSubvectors,
          trained: true, // Set trained flag here after all processing is done
        };

        // Store dimension-specific model
        this.dimensionModels.set(dimension, dimModel);

        if (progressCallback) {
          progressCallback(1.0);
        }
      } catch (error) {
        console.error(`Error during PQ training for dimension ${dimension}:`, error);
        throw error;
      }
    } else {
      // Train default model
      // If dimension doesn't match default, recalculate parameters
      if (vectorDim !== this.defaultVectorSize) {
        const params = this._calculateOptimalSubvectorParams(vectorDim);
        this.subvectorSize = params.subvectorSize;
        this.numSubvectors = params.numSubvectors;
        this.defaultVectorSize = vectorDim;

        console.log(`Adjusted to vector dimension ${vectorDim}: using ${this.numSubvectors} subquantizers of size ${this.subvectorSize}`);
      }

      // Validate configuration before training
      this._validateAndAdjustSubvectorConfig();

      // Initialize centroids array
      this.centroids = new Array(this.numSubvectors);

      // Initialize kmeans instance for clustering
      const kmeans = new KMeans(this.numClusters, 100, 0.001);

      try {
        // Train each subquantizer
        for (let m = 0; m < this.numSubvectors; m++) {
          // Extract subvectors for this subquantizer
          const subvectors: Vector[] = [];
          const start = m * this.subvectorSize;
          const end = Math.min(start + this.subvectorSize, vectorDim);

          for (const vector of vectors) {
            // Extract subvector and add to training set
            const subvector = vector.slice(start, end);
            subvectors.push(subvector);
          }

          // Train k-means on this set of subvectors
          if (progressCallback) {
            progressCallback(m / this.numSubvectors);
          }

          // Cluster subvectors
          this.centroids[m] = await kmeans.cluster(subvectors);

          // Allow other operations to proceed
          await new Promise((resolve) => setTimeout(resolve, 0));
        }

        // Set the trained flag only AFTER all centroids are properly initialized
        let allCentroidsValid = true;
        for (const subq of this.centroids) {
          if (!subq || subq.length !== this.numClusters) {
            allCentroidsValid = false;
            break;
          }
        }

        this.trained = allCentroidsValid;

        if (progressCallback) {
          progressCallback(1.0);
        }
      } catch (error) {
        // Mark as not trained in case of error
        this.trained = false;
        console.error('Error during PQ training:', error);
        throw error;
      }
    }
  }

  /**
   * Serialize the PQ model to JSON
   * @returns Serialized model
   */
  serialize(): string {
    const data: any = {
      defaultVectorSize: this.defaultVectorSize,
      subvectorSize: this.subvectorSize,
      numSubvectors: this.numSubvectors,
      numClusters: this.numClusters,
      trained: this.trained,
      dynamicDimensions: this.dynamicDimensions,
      minSubquantizers: this.minSubquantizers,
      version: 1,
    };

    // Serialize default model centroids
    if (this.trained) {
      data.centroids = this.centroids.map((subq) => subq.map((centroid) => Array.from(centroid)));
    }

    // Serialize dimension-specific models
    data.dimensionModels = {};
    for (const [dim, model] of this.dimensionModels.entries()) {
      if (model.trained) {
        data.dimensionModels[dim] = {
          numSubvectors: model.numSubvectors,
          centroids: model.centroids.map((subq) => subq.map((centroid) => Array.from(centroid))),
          trained: true,
        };
      }
    }

    return JSON.stringify(data);
  }

  /**
   * Save the PQ model to a file
   * @param filePath Path to save the model
   */
  async saveModel(filePath: string): Promise<void> {
    const data = this.serialize();
    await fs.writeFile(filePath, data, 'utf8');
  }

  /**
   * Load a serialized PQ model
   * @param json Serialized model data or path to model file
   * @param options Load options
   * @returns ProductQuantization instance
   */
  static async loadModel(json: string, options: LoadModelOptions = {}): Promise<ProductQuantization> {
    let data: any;

    // Check if json is a file path
    if (json.endsWith('.json')) {
      try {
        const fileContent = await fs.readFile(json, 'utf8');
        data = JSON.parse(fileContent);
      } catch (error: unknown) {
        throw new Error(`Failed to load PQ model from file: ${error instanceof Error ? error.message : 'Unknown error'}`);
      }
    } else {
      // Parse JSON string directly
      try {
        data = JSON.parse(json);
      } catch (error: unknown) {
        throw new Error(`Failed to parse PQ model JSON: ${error instanceof Error ? error.message : 'Unknown error'}`);
      }
    }

    // Validate model data
    if (!data.version || !data.defaultVectorSize || !data.numSubvectors) {
      throw new Error('Invalid PQ model format');
    }

    // Create new PQ instance with model parameters
    const pq = new ProductQuantization({
      vectorSize: data.defaultVectorSize,
      numSubvectors: data.numSubvectors,
      subvectorSize: data.subvectorSize,
      numClusters: data.numClusters,
      dynamicDimensions: options.dynamicDimensions !== false && data.dynamicDimensions !== false,
      minSubquantizers: data.minSubquantizers,
    });

    // Load default model centroids
    if (data.trained && data.centroids) {
      pq.trained = true;
      pq.centroids = data.centroids.map((subq: number[][]) => subq.map((centroid) => new Float32Array(centroid)));
    }

    // Load dimension-specific models
    if (data.dimensionModels) {
      for (const [dimStr, modelData] of Object.entries(data.dimensionModels)) {
        const dim = parseInt(dimStr, 10);
        const model = modelData as any;

        if (model.trained && model.centroids) {
          pq.dimensionModels.set(dim, {
            numSubvectors: model.numSubvectors,
            centroids: model.centroids.map((subq: number[][]) => subq.map((centroid) => new Float32Array(centroid))),
            trained: true,
          });
        }
      }
    }

    return pq;
  }

  /**
   * Compute distance tables for a query vector
   * This pre-computes distances between query subvectors and centroids
   * for efficient distance approximation
   *
   * @param query The query vector
   * @returns Distance tables as Float32Array[]
   */
  computeDistanceTables(query: Vector): Float32Array[] {
    if (!this.trained) {
      throw new Error('Product quantizer not trained');
    }

    const dimension = query.length;
    let modelCentroids: Float32Array[][];
    let numSubvectors: number;
    let subvectorSize: number;

    // Determine which model to use based on vector dimension
    if (this.dynamicDimensions && dimension !== this.defaultVectorSize) {
      // Try to find a trained model for this dimension
      if (this.dimensionModels.has(dimension) && this.dimensionModels.get(dimension)!.trained) {
        const model = this.dimensionModels.get(dimension)!;
        modelCentroids = model.centroids;
        numSubvectors = model.numSubvectors;
        const params = this._calculateSubvectorParams(dimension);
        subvectorSize = params.subvectorSize;
      } else {
        // Fall back to default model
        modelCentroids = this.centroids;
        numSubvectors = this.numSubvectors;
        subvectorSize = this.subvectorSize;
        console.warn(`No specific model found for dimension ${dimension}, using default model`);
      }
    } else {
      // Use default model
      modelCentroids = this.centroids;
      numSubvectors = this.numSubvectors;
      subvectorSize = this.subvectorSize;
    }

    // Create distance tables for each subvector
    const tables: Float32Array[] = [];

    for (let m = 0; m < numSubvectors; m++) {
      // Extract subvector from query
      const start = m * subvectorSize;
      const end = Math.min(start + subvectorSize, dimension);
      const length = end - start;

      // Create table for this subquantizer
      const table = new Float32Array(this.numClusters);

      // For each centroid in this subquantizer
      for (let i = 0; i < this.numClusters; i++) {
        // Get centroid vector
        const centroid = modelCentroids[m][i];

        // Skip if centroid doesn't exist (can happen if numClusters > training data)
        if (!centroid) continue;

        // Compute squared distance
        let dist = 0;
        for (let j = 0; j < length; j++) {
          const qval = typeof query[start + j] === 'number' ? query[start + j] : 0;
          const cval = j < centroid.length ? centroid[j] : 0;
          const diff = qval - cval;
          dist += diff * diff;
        }

        // Store distance in table
        table[i] = dist;
      }

      tables.push(table);
    }

    return tables;
  }

  /**
   * Calculate the approximate distance between a query and a PQ code
   * using pre-computed distance tables
   *
   * @param tables Distance tables from computeDistanceTables
   * @param code PQ code representing the vector
   * @returns Approximate distance
   */
  distanceWithTables(tables: Float32Array[], code: Uint8Array): number {
    if (!this.trained) {
      throw new Error('Product quantizer not trained');
    }

    // Sum up distances from tables
    let dist = 0;
    const numSubquantizers = Math.min(tables.length, code.length);

    for (let m = 0; m < numSubquantizers; m++) {
      const centroidIndex = code[m];
      dist += tables[m][centroidIndex];
    }

    return dist;
  }

  /**
   * Encode a vector using product quantization
   * @param vector Vector to encode
   * @param dimension Target dimension (optional)
   * @returns Encoded vector as Uint8Array
   */
  encode(vector: Vector, dimension?: number): Uint8Array {
    if (!this.trained) {
      throw new Error('Product quantizer not trained');
    }

    // Add validation to double check centroids are initialized
    if (!this.centroids || this.centroids.length === 0) {
      this.trained = false; // Reset trained flag if centroids aren't actually available
      throw new Error('Product quantizer centroids not properly initialized');
    }

    // Determine which model to use
    let modelCentroids: Float32Array[][];
    let numSubvectors: number;
    let subvectorSize: number;

    // Handle dynamic dimensions if provided
    const vectorDim = vector.length;
    if (dimension !== undefined || (this.dynamicDimensions && vectorDim !== this.defaultVectorSize)) {
      const targetDim = dimension || vectorDim;

      // Try to find a trained model for this dimension
      if (this.dimensionModels.has(targetDim) && this.dimensionModels.get(targetDim)!.trained) {
        const model = this.dimensionModels.get(targetDim)!;
        modelCentroids = model.centroids;
        numSubvectors = model.numSubvectors;
        const params = this._calculateSubvectorParams(targetDim);
        subvectorSize = params.subvectorSize;
      } else {
        // Fall back to default model
        modelCentroids = this.centroids;
        numSubvectors = this.numSubvectors;
        subvectorSize = this.subvectorSize;
      }
    } else {
      // Use default model
      modelCentroids = this.centroids;
      numSubvectors = this.numSubvectors;
      subvectorSize = this.subvectorSize;
    }

    // Create code array
    const code = new Uint8Array(numSubvectors);

    // Encode each subvector
    for (let m = 0; m < numSubvectors; m++) {
      // Extract subvector
      const start = m * subvectorSize;
      const end = Math.min(start + subvectorSize, vectorDim);

      // Find nearest centroid for this subvector
      let bestDist = Infinity;
      let bestIndex = 0;

      for (let i = 0; i < this.numClusters; i++) {
        const centroid = modelCentroids[m][i];
        // Skip if centroid doesn't exist
        if (!centroid) continue;

        // Compute distance to centroid
        let dist = 0;
        for (let j = 0; j < end - start; j++) {
          if (start + j >= vectorDim) break;
          const vval = typeof vector[start + j] === 'number' ? vector[start + j] : 0;
          const cval = j < centroid.length ? centroid[j] : 0;
          const diff = vval - cval;
          dist += diff * diff;
        }

        // Update best match
        if (dist < bestDist) {
          bestDist = dist;
          bestIndex = i;
        }
      }

      // Store centroid index
      code[m] = bestIndex;
    }

    return code;
  }

  /**
   * Store encoded vector for an ID
   * @param id Vector ID
   * @param code Encoded vector
   * @param dimension Vector dimension (optional)
   */
  storeCode(id: string | number, code: Uint8Array, dimension?: number): void {
    // Store in the main codes map for backward compatibility
    this.codes.set(id, code);

    // If dimension is specified, also store in dimension-specific map
    if (dimension !== undefined) {
      if (!this.dimensionCodes.has(dimension)) {
        this.dimensionCodes.set(dimension, new Map());
      }
      this.dimensionCodes.get(dimension)!.set(id, code);
      this.vectorDimensions.set(id, dimension);
    } else if (this.vectorDimensions.has(id)) {
      // If we know the dimension from previous operations
      const dim = this.vectorDimensions.get(id)!;
      if (!this.dimensionCodes.has(dim)) {
        this.dimensionCodes.set(dim, new Map());
      }
      this.dimensionCodes.get(dim)!.set(id, code);
    }
  }

  /**
   * Get code for a vector ID if it exists
   * @param id Vector ID
   * @param dimension Specific dimension to check (optional)
   */
  getCode(id: string | number, dimension?: number): Uint8Array | undefined {
    // If dimension is specified, check dimension-specific map first
    if (dimension !== undefined && this.dimensionCodes.has(dimension)) {
      const code = this.dimensionCodes.get(dimension)!.get(id);
      if (code) return code;
    }

    // Try to find the code in the dimension map if we know its dimension
    if (this.vectorDimensions.has(id)) {
      const dim = this.vectorDimensions.get(id)!;
      if (this.dimensionCodes.has(dim)) {
        const code = this.dimensionCodes.get(dim)!.get(id);
        if (code) return code;
      }
    }

    // Fall back to the main codes map
    return this.codes.get(id);
  }

  /**
   * Store vector encodings for fast lookup
   * @param codes Map of codes
   * @param dimension Dimension of these codes (optional)
   */
  setCodes(codes: Map<string | number, Uint8Array> | Record<string, Uint8Array>, dimension?: number): void {
    // Convert to Map if needed
    const codeMap = codes instanceof Map ? codes : new Map(Object.entries(codes).map(([id, code]) => [/^\d+$/.test(id) ? parseInt(id, 10) : id, code]));

    // If dimension is specified, store in dimension-specific map
    if (dimension !== undefined) {
      if (!this.dimensionCodes.has(dimension)) {
        this.dimensionCodes.set(dimension, new Map());
      }
      const dimCodes = this.dimensionCodes.get(dimension)!;

      for (const [id, code] of codeMap.entries()) {
        dimCodes.set(id, code);
        this.vectorDimensions.set(id, dimension);
      }
    }

    // Always update the main codes map for backward compatibility
    this.codes = new Map([...this.codes, ...codeMap]);
  }

  /**
   * Get compression ratio compared to original vectors
   * @param dimension Vector dimension to calculate for (optional)
   */
  getCompressionRatio(dimension?: number): number {
    const vectorSize = dimension || this.defaultVectorSize;
    const originalSize = vectorSize * 4; // 4 bytes per float

    // Find the appropriate model for this dimension
    let numSubvectors: number;

    if (dimension && this.dimensionModels.has(dimension)) {
      numSubvectors = this.dimensionModels.get(dimension)!.numSubvectors;
    } else {
      numSubvectors = this.numSubvectors;
    }

    const compressedSize = numSubvectors; // 1 byte per subvector
    return originalSize / compressedSize;
  }

  /**
   * Get statistics about the product quantization
   * @returns Object with PQ statistics
   */
  getStats(): Record<string, any> {
    // Count vectors per dimension
    const vectorsPerDimension: Record<number, number> = {};
    for (const dim of this.vectorDimensions.values()) {
      vectorsPerDimension[dim] = (vectorsPerDimension[dim] || 0) + 1;
    }

    // Count models per dimension
    const modelsPerDimension: Record<number, any> = {};
    for (const [dim, model] of this.dimensionModels.entries()) {
      if (model.trained) {
        const { subvectorSize } = this._calculateSubvectorParams(dim);
        modelsPerDimension[dim] = {
          numSubvectors: model.numSubvectors,
          subvectorSize,
          trained: true,
        };
      }
    }

    return {
      trained: this.trained,
      defaultVectorSize: this.defaultVectorSize,
      subvectorSize: this.subvectorSize,
      numSubvectors: this.numSubvectors,
      numClusters: this.numClusters,
      dynamicDimensions: this.dynamicDimensions,
      minSubquantizers: this.minSubquantizers,
      dimensions: {
        models: modelsPerDimension,
        vectors: vectorsPerDimension,
        unique: Object.keys(vectorsPerDimension).length,
      },
      compression: {
        ratio: this.getCompressionRatio(),
        originalSize: this.defaultVectorSize * 4, // bytes per float
        compressedSize: this.numSubvectors, // bytes per encoded vector
      },
      memory: {
        centroids: this.trained ? this.centroids.reduce((sum, subq) => sum + subq.length, 0) * this.subvectorSize * 4 : 0,
        dimensionModels: this.dimensionModels.size,
        codes: this.codes.size,
        totalCodes: this.codes.size + Array.from(this.dimensionCodes.values()).reduce((sum, map) => sum + map.size, 0),
      },
      vectors: {
        encoded: this.codes.size,
        dimensions: this.vectorDimensions.size,
      },
    };
  }

  /**
   * Get estimated memory usage of compressed vectors and centroids
   * @returns Memory usage in bytes
   */
  getMemoryUsage(): number {
    // Calculate size of centroids for the default model
    let centroidSize = 0;

    // Each centroid is a Float32Array (4 bytes per float)
    if (this.trained) {
      // Size of default model centroids
      centroidSize += this.centroids.reduce((sum, subquantizer) => sum + subquantizer.reduce((subSum, centroid) => subSum + centroid.length * 4, 0), 0);
    }

    // Add size of dimension-specific models
    for (const [dimension, model] of this.dimensionModels.entries()) {
      if (model.trained) {
        centroidSize += model.centroids.reduce((sum, subquantizer) => sum + subquantizer.reduce((subSum, centroid) => subSum + centroid.length * 4, 0), 0);
      }
    }

    // Calculate size of codes (1 byte per subquantizer per vector)
    let codeSize = 0;

    // Size of codes in the main map
    codeSize += this.codes.size * this.numSubvectors;

    // Size of dimension-specific codes
    for (const [dimension, codeMap] of this.dimensionCodes.entries()) {
      const numSubvectors = this.dimensionModels.has(dimension) ? this.dimensionModels.get(dimension)!.numSubvectors : Math.ceil(dimension / this.subvectorSize);

      codeSize += codeMap.size * numSubvectors;
    }

    // Sum total memory usage
    return centroidSize + codeSize;
  }
}
