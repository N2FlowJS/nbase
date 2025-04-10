// --- START OF FILE vector_db.ts ---

import { EventEmitter } from 'events';
import path from 'path';
import { promises as fsPromises, existsSync } from 'fs';
import zlib from 'zlib';
import { promisify } from 'util';

const gzip = promisify(zlib.gzip);
const gunzip = promisify(zlib.gunzip);

import config from '../config'; // Assuming config exists and has defaults
import { Vector, DBStats, SearchResult, VectorData, DistanceMetric, TypedEventEmitter, VectorDBEventData } from '../types';

// Function to serialize Float32Array to Buffer
function serializeVector(vector: Float32Array): Buffer {
  // Create a Buffer from the underlying ArrayBuffer of the Float32Array
  return Buffer.from(vector.buffer, vector.byteOffset, vector.byteLength);
}

// Function to deserialize Buffer to Float32Array
function deserializeVector(buffer: Buffer, dimension: number): Float32Array {
  if (buffer.length !== dimension * 4) {
    // Basic sanity check (4 bytes per float32)
    throw new Error(`Buffer length ${buffer.length} does not match expected size for dimension ${dimension}`);
  }
  // Create a Float32Array viewing the same memory as the Buffer
  // Ensure proper alignment and byte offset handling if the buffer comes from a larger allocation
  return new Float32Array(buffer.buffer, buffer.byteOffset, dimension);
}

// --- VectorDB Base Class (Optimized Persistence) ---

/**
 * Vector database for storing and retrieving high-dimensional vectors with associated metadata.
 * Provides efficient in-memory storage with optional persistence to disk.
 *
 * Key features:
 * - Store vectors with numeric or string IDs
 * - Associate metadata with vectors
 * - Find nearest neighbors using different distance metrics
 * - Persist database to disk with optional compression
 * - Event-based architecture for operation monitoring
 *
 * @example
 * ```typescript
 * // Create an in-memory database with default vector size of 384
 * const db = new VectorDB(384);
 *
 * // Add vectors with metadata
 * const id = db.addVector(undefined, new Float32Array([0.1, 0.2, 0.3]), { source: 'example' });
 *
 * // Find similar vectors
 * const results = db.findNearest([0.1, 0.15, 0.25], 5, { metric: 'cosine' });
 *
 * // Save to disk
 * db.setPath('./vector_data');
 * await db.save();
 * ```
 *
 * @fires vector:add - When a vector is added
 * @fires vectors:bulkAdd - When multiple vectors are added
 * @fires vector:delete - When a vector is deleted
 * @fires metadata:add - When metadata is added to a vector
 * @fires metadata:update - When vector metadata is updated
 * @fires db:save - When the database is saved to disk
 * @fires db:load - When the database is loaded from disk
 * @fires db:close - When the database is closed
 *
 * @extends EventEmitter
 */
export class VectorDB extends (EventEmitter as new () => TypedEventEmitter<VectorDBEventData>) {
  public defaultVectorSize: number | null = null;
  public memoryStorage: Map<number | string, Float32Array>;
  protected metadata: Map<number | string, Record<string, any>>;
  protected vectorDimensions: Map<number | string, number>; // Keep track of individual dimensions
  protected idCounter: number;
  protected dbPath: string | null; // Base path (without extension)
  protected savePromise: Promise<void> | null = null;
  protected isClosed: boolean = false;
  protected useCompression: boolean; // Option for compression
  public isReady: boolean = false; // Flag to indicate if the database is ready for operations
  constructor(suggestedVectorSize: number | null = null, dbPath: string | null = null, options: { useCompression?: boolean } = {}) {
    super();

    this.defaultVectorSize = suggestedVectorSize;
    this.memoryStorage = new Map();
    this.metadata = new Map();
    this.vectorDimensions = new Map();
    this.idCounter = 1;
    this.dbPath = dbPath;
    this.useCompression = options.useCompression ?? false; // Default to no compression

    if (dbPath) {
      console.log(`[VectorDB] Constructor Loading database from ${dbPath}...`);

      this.load()
        .catch((err) => {
          // Only log error if file likely existed but failed to load
          if (err.code !== 'ENOENT') {
            console.error(`Error loading database from ${dbPath}:`, err);
          } else {
            console.log(`Database files not found at ${dbPath}, starting fresh.`);
          }
        })
        .finally(() => {
          this.isReady = true; // Set ready flag after load attempt
        });
    }
  }
  getIdCounter(): number {
    return this.idCounter;
  }
  // --- Core Methods (Mostly Unchanged, check ID handling) ---

  setPath(dbPath: string): void {
    this.dbPath = dbPath;
  }

  vectorSize(): number {
    // (Keep existing logic for determining most common size)
    if (this.defaultVectorSize !== null) {
      return this.defaultVectorSize;
    }
    if (this.memoryStorage.size === 0) {
      return config.defaults.vectorSize || 0;
    }
    const dimensionCounts = new Map<number, number>();
    let maxCount = 0;
    let mostCommonSize = config.defaults.vectorSize || 0;
    for (const dim of this.vectorDimensions.values()) {
      const count = (dimensionCounts.get(dim) || 0) + 1;
      dimensionCounts.set(dim, count);
      if (count > maxCount) {
        maxCount = count;
        mostCommonSize = dim;
      }
    }
    return mostCommonSize;
  }

  addVector(
    id: number | string | undefined,
    vector: Vector,
    metadata?: Record<string, any> // Allow adding metadata directly
  ): number | string {
    let vectorId = id !== undefined ? id : this.idCounter++;

    // Optional: Standardize ID to string for internal consistency?
    // vectorId = String(vectorId);

    if (this.memoryStorage.has(vectorId)) {
      console.warn(`Vector with ID ${vectorId} already exists. Overwriting.`);
      // Decide if overwrite is desired or should throw error
    }

    const typedVector = vector instanceof Float32Array ? vector : new Float32Array(vector);
    const dimension = typedVector.length;

    if (this.memoryStorage.size === 0 && this.defaultVectorSize === null) {
      this.defaultVectorSize = dimension;
    }

    this.memoryStorage.set(vectorId, typedVector);
    this.vectorDimensions.set(vectorId, dimension); // Store dimension

    if (metadata) {
      this.metadata.set(vectorId, metadata);
    }

    // Update idCounter if a numeric ID is provided explicitly
    if (typeof vectorId === 'number' && vectorId >= this.idCounter) {
      this.idCounter = vectorId + 1;
    }

    this.emit('vector:add', { id: vectorId, dimensions: dimension });
    return vectorId;
  }

  bulkAdd(vectors: VectorData[]): number {
    console.log(`[VectorDB] Bulk adding ${vectors.length} vectors...`);

    let addedCount = 0;
    const addedIds: (number | string)[] = [];

    for (const item of vectors) {
      try {
        // Pass metadata if available in VectorData type
        const id = this.addVector(item.id, item.vector, item.metadata);
        addedCount++;
        addedIds.push(id);
      } catch (error) {
        console.error(`Error adding vector ${item.id}:`, error);
      }
    }

    // Verify vectors were actually added to memory storage
    if (addedCount > 0 && this.memoryStorage.size === 0) {
      console.warn('[VectorDB] Warning: bulkAdd reported success but memoryStorage is empty');
    } else {
      console.log(`[VectorDB] Successfully added ${addedCount} vectors to memory storage. Storage size: ${this.memoryStorage.size}`);
    }

    this.emit('vectors:bulkAdd', { count: addedCount, ids: addedIds });
    return addedCount;
  }

  getVector(id: number | string): Float32Array | null {
    // Keep existing logic (try exact, then conversions)
    return this.memoryStorage.get(id) ?? null; // Simpler check with nullish coalescing
    // Consider removing automatic type conversion for stricter behavior if desired
  }

  hasVector(id: number | string): boolean {
    // Keep existing logic
    return this.memoryStorage.has(id);
    // Consider removing automatic type conversion
  }

  deleteVector(id: number | string): boolean {
    const deleted = this.memoryStorage.delete(id);
    if (deleted) {
      this.metadata.delete(id);
      this.vectorDimensions.delete(id); // Remove dimension info
      this.emit('vector:delete', { id });
    }
    return deleted;
  }

  updateVector(id: number | string, vector: Vector): boolean {
    if (!this.memoryStorage.has(id)) {
      console.warn(`Attempted to update non-existent vector ID: ${id}`);
      return false;
    }

    const typedVector = vector instanceof Float32Array ? vector : new Float32Array(vector);
    const dimension = typedVector.length;

    this.memoryStorage.set(id, typedVector);
    this.vectorDimensions.set(id, dimension); // Update dimension

    this.emit('vector:update', { id, dimensions: dimension });
    return true;
  }

  addMetadata(id: number | string, data: Record<string, any>): void {
    if (!this.memoryStorage.has(id)) {
      // Use hasVector for consistency?
      throw new Error(`Vector with ID ${id} not found`);
    }
    this.metadata.set(id, data);
    this.emit('metadata:add', { id, metadata: data });
  }

  getMetadata(id: number | string): Record<string, any> | null {
    // Keep existing logic
    return this.metadata.get(id) ?? null;
    // Consider removing automatic type conversion
  }

  updateMetadata(id: number | string, data: Record<string, any> | ((current: Record<string, any> | null) => Record<string, any>)): boolean {
    // Keep existing logic
    if (!this.memoryStorage.has(id)) {
      console.warn(`Attempted to update metadata for non-existent vector ID: ${id}`);
      return false; // Or throw error
    }
    const current = this.metadata.get(id) || null;
    let updated: Record<string, any>;
    if (typeof data === 'function') {
      updated = data(current);
    } else {
      updated = { ...(current || {}), ...data }; // Ensure current is not null
    }
    this.metadata.set(id, updated);
    this.emit('metadata:update', { id, metadata: updated });
    return true;
  }

  getVectorDimension(id: number | string): number | null {
    // Direct lookup is now primary source
    return this.vectorDimensions.get(id) ?? null;
    // Consider removing automatic type conversion
  }

  // --- Distance Calculations ---

  protected _calculateNorm(vector: Float32Array): number {
    let sum = 0;
    for (let i = 0; i < vector.length; i++) {
      sum += vector[i] * vector[i];
    }
    return Math.sqrt(sum);
  }

  protected _dotProduct(a: Float32Array, b: Float32Array): number {
    const len = Math.min(a.length, b.length); // Handle dimension mismatch
    let dot = 0;
    for (let i = 0; i < len; i++) {
      dot += a[i] * b[i];
    }
    return dot;
  }

  protected _euclideanDistance(a: Float32Array, b: Float32Array): number {
    const len = Math.min(a.length, b.length);
    let sum = 0;
    for (let i = 0; i < len; i++) {
      const d = a[i] - b[i];
      sum += d * d;
    }
    // Optional: Penalty for dimension mismatch (consider if really needed)
    const dimDiff = Math.abs(a.length - b.length);
    if (dimDiff > 0) {
      // Simple penalty, maybe make this configurable or remove
      sum += dimDiff * (config.defaults.dimensionMismatchPenalty ?? 0.01);
    }
    return Math.sqrt(sum);
  }

  // Cosine Similarity returns similarity (higher is better).
  // Often 1 - similarity is used as a distance metric (lower is better).
  protected _cosineDistance(a: Float32Array, b: Float32Array): number {
    const normA = this._calculateNorm(a);
    const normB = this._calculateNorm(b);
    if (normA === 0 || normB === 0) {
      return 1.0; // Handle zero vectors - maximally distant
    }
    // Ensure dimensions match for a meaningful cosine similarity
    if (a.length !== b.length) {
      // Or handle as per _euclideanDistance mismatch logic?
      // Returning max distance is safer if dimensions must match.
      console.warn(`Cosine distance called on vectors with different dimensions (${a.length} vs ${b.length}). Returning max distance.`);
      return 1.0;
    }
    const dot = this._dotProduct(a, b);
    // Clamp the result to [-1, 1] due to potential floating point inaccuracies
    const similarity = Math.max(-1.0, Math.min(1.0, dot / (normA * normB)));
    return 1.0 - similarity; // Convert similarity to distance
  }

  protected _calculateDistance(a: Float32Array, b: Float32Array, metric: DistanceMetric): number {
    switch (metric) {
      case 'cosine':
        return this._cosineDistance(a, b);
      case 'euclidean':
      default: // Default to Euclidean
        return this._euclideanDistance(a, b);
    }
  }

  // --- Search (Linear Scan - Base Implementation) ---

  findNearest(
    query: Vector,
    k: number = 10,
    options: {
      filter?: (id: number | string, metadata?: Record<string, any>) => boolean;
      metric?: DistanceMetric; // Allow specifying metric
    } = {}
  ): SearchResult[] {
    const typedQuery = query instanceof Float32Array ? query : new Float32Array(query);
    const metric = options.metric ?? 'euclidean'; // Default metric

    return this._linearSearch(typedQuery, k, metric, options.filter);
  }

  protected _linearSearch(query: Float32Array, k: number, metric: DistanceMetric, filter?: (id: number | string, metadata?: Record<string, any>) => boolean): SearchResult[] {
    const results: SearchResult[] = [];
    const queryDim = query.length;

    for (const [id, vector] of this.memoryStorage.entries()) {
      // Filter first (if provided) to potentially skip distance calculation
      if (filter) {
        const meta = this.metadata.get(id);
        if (!filter(id, meta)) {
          continue;
        }
      }

      // Important: Ensure dimension compatibility based on metric
      if (metric === 'cosine' && vector.length !== queryDim) {
        // Cosine requires same dimensions
        continue;
      }
      // Euclidean can handle different dimensions (with penalty)

      const dist = this._calculateDistance(query, vector, metric);
      results.push({ id, dist });
    }

    // Sort by distance and limit to k results
    // Note: For cosine distance (1-similarity), lower is better, so sort ascending still works
    return results.sort((a, b) => a.dist - b.dist).slice(0, k);
  }

  // --- Optimized Persistence ---

  protected _getMetaFilePath(): string {
    if (!this.dbPath) throw new Error('DB path not set');
    return path.join(this.dbPath, 'meta.json') + (this.useCompression ? '.gz' : '');
  }

  protected _getVectorFilePath(): string {
    if (!this.dbPath) throw new Error('DB path not set');
    return path.join(this.dbPath, 'vec.bin' + (this.useCompression ? '.gz' : ''));
  }

  async save(): Promise<void> {
    console.log('[VectorDB] Saving database...');

    if (!this.dbPath) {
      console.warn('[VectorDB] No dbPath specified, skipping save.');
      return;
    }
    if (this.isClosed) {
      console.warn('[VectorDB] Attempted to save a closed database.');
      return;
    }
    console.log(`[VectorDB] Saving to ${this.dbPath}`);

    // Only log and return existing promise if a save is already in progress
    if (this.savePromise) {
      console.log(`[VectorDB] Save already in progress, waiting...`);
      return this.savePromise;
    }

    this.savePromise = (async () => {
      const metaFilePath = this._getMetaFilePath();
      const vectorFilePath = this._getVectorFilePath();
      console.log('[VectorDB] Meta file path:', metaFilePath);
      console.log('[VectorDB] Vector file path:', vectorFilePath);

      try {
        // Ensure directory exists
        await fsPromises.mkdir(path.dirname(metaFilePath), { recursive: true });

        console.log('[VectorDB] Meta file path:', metaFilePath);
        console.log('[VectorDB] Vector file path:', vectorFilePath);

        const metaData: Record<string, any> = {};
        this.metadata.forEach((value, key) => {
          // Ensure keys are strings for JSON compatibility
          metaData[String(key)] = value;
        });

        const vectorInfo: Array<{
          id: number | string;
          offset: number;
          length: number;
          dim: number;
        }> = [];
        const vectorBuffers: Buffer[] = [];
        let currentOffset = 0;

        // 1. Prepare vector data and metadata structure
        console.log(`[VectorDB] Preparing vector data for saving with ${this.memoryStorage.size} vectors...`);

        for (const [id, vector] of this.memoryStorage.entries()) {
          const vectorBuffer = serializeVector(vector);
          vectorBuffers.push(vectorBuffer);
          vectorInfo.push({
            id: id,
            offset: currentOffset,
            length: vectorBuffer.length, // Store byte length
            dim: vector.length, // Store dimension
          });
          currentOffset += vectorBuffer.length;
        }

        const saveData = {
          version: 1, // Add a version number for future format changes
          defaultVectorSize: this.defaultVectorSize,
          idCounter: this.idCounter,
          vectors: vectorInfo,
          metadata: metaData,
        };
        console.log(`[VectorDB] Vector data prepared for saving: ${vectorInfo.length} vectors`);
        // 2. Write metadata file
        console.log(`[VectorDB] Writing metadata file to: ${metaFilePath} with ${vectorInfo.length} vectors`);
        // Ensure metadata is JSON-serializable
        let metaContent: string | Buffer = JSON.stringify(saveData);
        if (this.useCompression) {
          metaContent = await gzip(metaContent);
        }
        console.log('[VectorDB] Writing meta file to:', metaFilePath);
        await fsPromises.writeFile(metaFilePath, metaContent);
        console.log('[VectorDB] Meta file written successfully.');
        // 3. Write vector data file
        let vectorContent: Buffer | Buffer[] = Buffer.concat(vectorBuffers);
        if (this.useCompression) {
          vectorContent = await gzip(vectorContent);
        }
        console.log(`[VectorDB] Writing vector file to: ${vectorFilePath} (${vectorBuffers.length} vectors, ${vectorContent.length} bytes)`);

        await fsPromises.writeFile(vectorFilePath, vectorContent);
        console.log('[VectorDB] Vector file written successfully.');
        // 4. Emit save event

        this.emit('db:save', {
          path: this.dbPath || 'DB path not set',
          count: this.memoryStorage.size,
        });
        console.log('[VectorDB] Save event emitted successfully.');
      } catch (error) {
        console.error(`Error saving database to ${this.dbPath}:`, error);
        throw error; // Re-throw to indicate failure
      } finally {
        this.savePromise = null; // Release lock
      }
    })();
    console.log('[VectorDB] Save promise created.');
    return this.savePromise;
  }

  async load(): Promise<void> {
    if (!this.dbPath) {
      throw new Error('Database path not specified for loading.');
    }
    if (this.isClosed) {
      throw new Error('Cannot load into a closed database.');
    }

    const metaFilePath = this._getMetaFilePath();
    const vectorFilePath = this._getVectorFilePath();

    // Check if files exist first to avoid unnecessary error logging for new databases
    const metaExists = existsSync(metaFilePath);
    const vecExists = existsSync(vectorFilePath);

    // If both files don't exist, this is likely a new database
    if (!metaExists && !vecExists) {
      console.log(`[VectorDB] Database files not found at ${this.dbPath}. Starting new database.`);
      return; // Exit early for a new database initialization
    }

    try {
      // 1. Read and parse metadata file
      let metaContentBuffer = await fsPromises.readFile(metaFilePath);
      if (this.useCompression) {
        metaContentBuffer = await gunzip(metaContentBuffer);
      }
      const saveData = JSON.parse(metaContentBuffer.toString('utf8'));

      if (saveData.version !== 1) {
        throw new Error(`Unsupported database format version: ${saveData.version}`);
      }

      // 2. Read vector data file
      let vectorDataBuffer = await fsPromises.readFile(vectorFilePath);
      if (this.useCompression) {
        vectorDataBuffer = await gunzip(vectorDataBuffer);
      }

      // 3. Clear existing data
      this.memoryStorage.clear();
      this.metadata.clear();
      this.vectorDimensions.clear();

      // 4. Load data into memory
      this.defaultVectorSize = saveData.defaultVectorSize;
      this.idCounter = saveData.idCounter ?? 1; // Use saved counter or default

      for (const vecInfo of saveData.vectors) {
        const { id, offset, length, dim } = vecInfo;
        if (offset + length > vectorDataBuffer.length) {
          console.error(`Invalid offset/length for vector ${id}. Offset: ${offset}, Length: ${length}, Buffer Size: ${vectorDataBuffer.length}`);
          continue; // Skip corrupted entry
        }
        const vectorSlice = vectorDataBuffer.slice(offset, offset + length);
        const vector = deserializeVector(vectorSlice, dim);

        this.memoryStorage.set(id, vector);
        this.vectorDimensions.set(id, dim);

        // Load metadata (handle string keys from JSON)
        const meta = saveData.metadata[String(id)];
        if (meta) {
          this.metadata.set(id, meta);
        }
      }

      this.emit('db:load', {
        path: this.dbPath,
        count: this.memoryStorage.size,
      });
      console.log(`[VectorDB] Loaded ${this.memoryStorage.size} vectors from ${this.dbPath}`);
    } catch (error: any) {
      if (error.code === 'ENOENT') {
        // Files not found is expected for a new DB, don't throw
        console.log(`Database files not found at ${this.dbPath}. Starting new database.`);
        return; // Don't re-throw ENOENT
      }
      console.error(`Error loading database from ${this.dbPath}:`, error);
      throw error; // Re-throw other errors
    }
  }

  // --- Stats and Lifecycle ---

  getStats(): DBStats {
    // (Keep existing logic, ensure it uses this.vectorDimensions)
    const dimensionCounts: Record<number, number> = {};
    for (const dim of this.vectorDimensions.values()) {
      dimensionCounts[dim] = (dimensionCounts[dim] || 0) + 1;
    }

    // Recalculate memory usage based on actual stored data
    let vectorMemory = 0;
    this.memoryStorage.forEach((vec) => (vectorMemory += vec.byteLength));

    let metadataMemory = 0;
    try {
      // Estimate metadata size (crude)
      this.metadata.forEach((meta) => (metadataMemory += JSON.stringify(meta).length * 2));
    } catch (e) {
      console.warn('Could not estimate metadata size:', e);
    }

    const baseStats: DBStats = {
      vectorCount: this.memoryStorage.size,
      vectorSize: this.vectorSize(),
      defaultVectorSize: this.defaultVectorSize ?? 0, // Use 0 if null
      metadataCount: this.metadata.size,
      dimensions: {
        counts: dimensionCounts,
        unique: Object.keys(dimensionCounts).length,
      },
      // More accurate memory usage estimate
      memoryUsage:
        vectorMemory +
        metadataMemory +
        this.memoryStorage.size * 16 + // Estimate Map overhead for vectors
        this.metadata.size * 16 + // Estimate Map overhead for metadata
        this.vectorDimensions.size * 8, // Estimate Map overhead for dimensions
      // Placeholder for cluster stats (filled by subclass)
      clusters: { count: 0, avgSize: 0, distribution: [], dimensions: {} },
    };

    // Compatibility check for older DBStats type if needed
    if (!(baseStats.clusters as any).dimensions) {
      (baseStats.clusters as any).dimensions = {};
    }

    return baseStats;
  }

  protected _estimateMemoryUsage(): number {
    // This method is now effectively replaced by the calculation within getStats()
    // Kept for potential internal use or backward compatibility if needed elsewhere
    let vectorMemory = 0;
    this.memoryStorage.forEach((vec) => (vectorMemory += vec.byteLength));
    let metadataMemory = 0;
    try {
      this.metadata.forEach((meta) => (metadataMemory += JSON.stringify(meta).length * 2));
    } catch (e) {
      /* ignore */
    }
    const dimensionOverhead = this.vectorDimensions.size * 8;
    return vectorMemory + metadataMemory + dimensionOverhead + (this.memoryStorage.size + this.metadata.size) * 16; // Rough map overhead
  }
  /**
   * Gets a list of metadata entries that match specified criteria.
   *
   * @param criteria Can be:
   *   - A string: field name to check for existence
   *   - An array of strings: multiple field names to check for existence
   *   - An object: key-value pairs where each key must exist and match the specified value
   * @param values Optional value(s) to match against the field(s) when using string/array input
   * @returns Array of {id, metadata} objects for entries that match the criteria
   *
   * @example
   * ```typescript
   * // Get all metadata entries that have a 'source' field
   * const allWithSource = db.getMetadataWithField('source');
   *
   * // Get metadata entries where 'category' equals 'article'
   * const articles = db.getMetadataWithField('category', 'article');
   *
   * // Get entries that have both 'author' and 'title' fields
   * const authoredContent = db.getMetadataWithField(['author', 'title']);
   *
   * // Get entries where 'type' is 'book' AND 'published' is true
   * const publishedBooks = db.getMetadataWithField(['type', 'published'], ['book', true]);
   *
   * // Using object syntax (recommended): type='book' AND published=true
   * const publishedBooks = db.getMetadataWithField({ type: 'book', published: true });
   * ```
   */
  getMetadataWithField(
    criteria: string | string[] | Record<string, any>,
    values?: any | any[],
    options?: { limit?: number } // Optional limit for results
  ): Array<{ id: number | string; metadata: Record<string, any> }> {
    const results: Array<{ id: number | string; metadata: Record<string, any> }> = [];

    // Handle object criteria format (new format)
    if (criteria !== null && typeof criteria === 'object' && !Array.isArray(criteria)) {
      const criteriaObj = criteria as Record<string, any>;
      const fields = Object.keys(criteriaObj);

      this.metadata.forEach((meta, id) => {
        let match = true;

        // Check if all fields exist and match their values
        for (const field of fields) {
          if (!(field in meta) || meta[field] !== criteriaObj[field]) {
            match = false;
            break;
          }
        }

        if (match) {
          results.push({ id, metadata: { ...meta } }); // Return a copy of metadata
        }
      });
      if (options?.limit) {
        return results.slice(0, options.limit); // Limit results if specified
      }
      return results;
    }

    // Handle legacy string/array format
    const fieldArray = Array.isArray(criteria) ? criteria : [criteria];
    const valueArray = values !== undefined ? (Array.isArray(values) ? values : [values]) : undefined;

    // If values are provided, ensure the length matches fields
    if (valueArray !== undefined && valueArray.length !== fieldArray.length) {
      console.warn('Values array length does not match fields array length. Some value checks will be ignored.');
    }

    this.metadata.forEach((meta, id) => {
      let match = true;

      // Check all fields exist and match values if provided
      for (let i = 0; i < fieldArray.length; i++) {
        const field = fieldArray[i];

        if (!(field in meta)) {
          match = false;
          break;
        }

        // If values are provided, check if the field value matches
        if (valueArray !== undefined && i < valueArray.length && meta[field] !== valueArray[i]) {
          match = false;
          break;
        }
      }

      if (match) {
        results.push({ id, metadata: { ...meta } }); // Return a copy of metadata
      }
    });

    return results;
  }

  getVectorCount(): number {
    return this.memoryStorage.size;
  }

  async close(): Promise<void> {
    if (this.isClosed) return;
    this.isClosed = true; // Mark as closed immediately

    try {
      if (this.dbPath) {
        await this.save(); // Attempt to save on close
      }
    } catch (error) {
      console.error('Error saving database during close:', error);
    } finally {
      // Clear memory regardless of save success
      this.memoryStorage.clear();
      this.metadata.clear();
      this.vectorDimensions.clear();
      this.emit('db:close', {});
      console.log('Database closed.');
    }
  }
}
// --- END OF FILE vector_db.ts ---
