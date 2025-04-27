"use strict";
// --- START OF FILE vector_db.ts ---
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.VectorDB = void 0;
const events_1 = require("events");
const path_1 = __importDefault(require("path"));
const fs_1 = require("fs");
const zlib_1 = __importDefault(require("zlib"));
const util_1 = require("util");
const log_1 = require("../utils/log");
const gzip = (0, util_1.promisify)(zlib_1.default.gzip);
const gunzip = (0, util_1.promisify)(zlib_1.default.gunzip);
const config_1 = __importDefault(require("../config")); // Assuming config exists and has defaults
// Function to serialize Float32Array to Buffer
function serializeVector(vector) {
    // Create a Buffer from the underlying ArrayBuffer of the Float32Array
    return Buffer.from(vector.buffer, vector.byteOffset, vector.byteLength);
}
// Function to deserialize Buffer to Float32Array
function deserializeVector(buffer, dimension) {
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
class VectorDB extends events_1.EventEmitter {
    constructor(suggestedVectorSize = null, dbPath = null, options = {}) {
        super();
        this.defaultVectorSize = null;
        this.savePromise = null;
        this.isClosed = false;
        this.isReady = false; // Flag to indicate if the database is ready for operations
        this.defaultVectorSize = suggestedVectorSize;
        this.memoryStorage = new Map();
        this.metadata = new Map();
        this.vectorDimensions = new Map();
        this.idCounter = 1;
        this.dbPath = dbPath;
        this.useCompression = options.useCompression ?? false; // Default to no compression
        if (dbPath) {
            (0, log_1.log)('info', `[VectorDB] Constructor Loading database from ${dbPath}...`);
            this.load()
                .catch((err) => {
                // Only log error if file likely existed but failed to load
                if (err.code !== 'ENOENT') {
                    (0, log_1.log)('error', `Error loading database from ${dbPath}:`, err);
                }
                else {
                    (0, log_1.log)('info', `Database files not found at ${dbPath}, starting fresh.`);
                }
            })
                .finally(() => {
                this.isReady = true; // Set ready flag after load attempt
            });
        }
    }
    getIdCounter() {
        return this.idCounter;
    }
    // --- Core Methods (Mostly Unchanged, check ID handling) ---
    setPath(dbPath) {
        this.dbPath = dbPath;
    }
    vectorSize() {
        // (Keep existing logic for determining most common size)
        if (this.defaultVectorSize !== null) {
            return this.defaultVectorSize;
        }
        if (this.memoryStorage.size === 0) {
            return config_1.default.defaults.vectorSize || 0;
        }
        const dimensionCounts = new Map();
        let maxCount = 0;
        let mostCommonSize = config_1.default.defaults.vectorSize || 0;
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
    addVector(id, vector, metadata // Allow adding metadata directly
    ) {
        let vectorId = id !== undefined ? id : this.idCounter++;
        // Optional: Standardize ID to string for internal consistency?
        // vectorId = String(vectorId);
        if (this.memoryStorage.has(vectorId)) {
            (0, log_1.log)('warn', `Vector with ID ${vectorId} already exists. Overwriting.`);
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
    bulkAdd(vectors) {
        (0, log_1.log)('info', `[VectorDB] Bulk adding ${vectors.length} vectors...`);
        let addedCount = 0;
        const addedIds = [];
        for (const item of vectors) {
            try {
                // Pass metadata if available in VectorData type
                const id = this.addVector(item.id, item.vector, item.metadata);
                addedCount++;
                addedIds.push(id);
            }
            catch (error) {
                (0, log_1.log)('error', `Error adding vector ${item.id}:`, error);
            }
        }
        // Verify vectors were actually added to memory storage
        if (addedCount > 0 && this.memoryStorage.size === 0) {
            (0, log_1.log)('warn', '[VectorDB] Warning: bulkAdd reported success but memoryStorage is empty');
        }
        else {
            (0, log_1.log)('info', `[VectorDB] Successfully added ${addedCount} vectors to memory storage. Storage size: ${this.memoryStorage.size}`);
        }
        this.emit('vectors:bulkAdd', { count: addedCount, ids: addedIds });
        return addedCount;
    }
    getVector(id) {
        // Keep existing logic (try exact, then conversions)
        return this.memoryStorage.get(id) ?? null; // Simpler check with nullish coalescing
        // Consider removing automatic type conversion for stricter behavior if desired
    }
    hasVector(id) {
        // Keep existing logic
        return this.memoryStorage.has(id);
        // Consider removing automatic type conversion
    }
    deleteVector(id) {
        const deleted = this.memoryStorage.delete(id);
        if (deleted) {
            this.metadata.delete(id);
            this.vectorDimensions.delete(id); // Remove dimension info
            this.emit('vector:delete', { id });
        }
        return deleted;
    }
    updateVector(id, vector) {
        if (!this.memoryStorage.has(id)) {
            (0, log_1.log)('warn', `Attempted to update non-existent vector ID: ${id}`);
            return false;
        }
        const typedVector = vector instanceof Float32Array ? vector : new Float32Array(vector);
        const dimension = typedVector.length;
        this.memoryStorage.set(id, typedVector);
        this.vectorDimensions.set(id, dimension); // Update dimension
        this.emit('vector:update', { id, dimensions: dimension });
        return true;
    }
    addMetadata(id, data) {
        if (!this.memoryStorage.has(id)) {
            // Use hasVector for consistency?
            throw new Error(`Vector with ID ${id} not found`);
        }
        this.metadata.set(id, data);
        this.emit('metadata:add', { id, metadata: data });
    }
    getMetadata(id) {
        // Keep existing logic
        return this.metadata.get(id) ?? null;
        // Consider removing automatic type conversion
    }
    updateMetadata(id, data) {
        // Keep existing logic
        if (!this.memoryStorage.has(id)) {
            (0, log_1.log)('warn', `Attempted to update metadata for non-existent vector ID: ${id}`);
            return false; // Or throw error
        }
        const current = this.metadata.get(id) || null;
        let updated;
        if (typeof data === 'function') {
            updated = data(current);
        }
        else {
            updated = { ...(current || {}), ...data }; // Ensure current is not null
        }
        this.metadata.set(id, updated);
        this.emit('metadata:update', { id, metadata: updated });
        return true;
    }
    getVectorDimension(id) {
        // Direct lookup is now primary source
        return this.vectorDimensions.get(id) ?? null;
        // Consider removing automatic type conversion
    }
    // --- Distance Calculations ---
    _calculateNorm(vector) {
        let sum = 0;
        for (let i = 0; i < vector.length; i++) {
            sum += vector[i] * vector[i];
        }
        return Math.sqrt(sum);
    }
    _dotProduct(a, b) {
        const len = Math.min(a.length, b.length); // Handle dimension mismatch
        let dot = 0;
        for (let i = 0; i < len; i++) {
            dot += a[i] * b[i];
        }
        return dot;
    }
    _euclideanDistance(a, b) {
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
            sum += dimDiff * (config_1.default.defaults.dimensionMismatchPenalty ?? 0.01);
        }
        return Math.sqrt(sum);
    }
    // Cosine Similarity returns similarity (higher is better).
    // Often 1 - similarity is used as a distance metric (lower is better).
    _cosineDistance(a, b) {
        const normA = this._calculateNorm(a);
        const normB = this._calculateNorm(b);
        if (normA === 0 || normB === 0) {
            return 1.0; // Handle zero vectors - maximally distant
        }
        // Ensure dimensions match for a meaningful cosine similarity
        if (a.length !== b.length) {
            // Or handle as per _euclideanDistance mismatch logic?
            // Returning max distance is safer if dimensions must match.
            (0, log_1.log)('warn', `Cosine distance called on vectors with different dimensions (${a.length} vs ${b.length}). Returning max distance.`);
            return 1.0;
        }
        const dot = this._dotProduct(a, b);
        // Clamp the result to [-1, 1] due to potential floating point inaccuracies
        const similarity = Math.max(-1.0, Math.min(1.0, dot / (normA * normB)));
        return 1.0 - similarity; // Convert similarity to distance
    }
    _calculateDistance(a, b, metric) {
        switch (metric) {
            case 'cosine':
                return this._cosineDistance(a, b);
            case 'euclidean':
            default: // Default to Euclidean
                return this._euclideanDistance(a, b);
        }
    }
    // --- Search (Linear Scan - Base Implementation) ---
    findNearest(query, k = 10, options = {}) {
        const typedQuery = query instanceof Float32Array ? query : new Float32Array(query);
        const metric = options.metric ?? 'euclidean'; // Default metric
        return this._linearSearch(typedQuery, k, metric, options.filter);
    }
    _linearSearch(query, k, metric, filter) {
        const results = [];
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
    _getMetaFilePath() {
        if (!this.dbPath)
            throw new Error('DB path not set');
        return path_1.default.join(this.dbPath, 'meta.json') + (this.useCompression ? '.gz' : '');
    }
    _getVectorFilePath() {
        if (!this.dbPath)
            throw new Error('DB path not set');
        return path_1.default.join(this.dbPath, 'vec.bin' + (this.useCompression ? '.gz' : ''));
    }
    async save() {
        (0, log_1.log)('info', '[VectorDB] Saving database...');
        if (!this.dbPath) {
            (0, log_1.log)('warn', '[VectorDB] No dbPath specified, skipping save.');
            return;
        }
        if (this.isClosed) {
            (0, log_1.log)('warn', '[VectorDB] Attempted to save a closed database.');
            return;
        }
        (0, log_1.log)('info', `[VectorDB] Saving to ${this.dbPath}`);
        // Only log and return existing promise if a save is already in progress
        if (this.savePromise) {
            (0, log_1.log)('info', `[VectorDB] Save already in progress, waiting...`);
            return this.savePromise;
        }
        this.savePromise = (async () => {
            const metaFilePath = this._getMetaFilePath();
            const vectorFilePath = this._getVectorFilePath();
            (0, log_1.log)('info', '[VectorDB] Meta file path:', metaFilePath);
            (0, log_1.log)('info', '[VectorDB] Vector file path:', vectorFilePath);
            try {
                // Ensure directory exists
                await fs_1.promises.mkdir(path_1.default.dirname(metaFilePath), { recursive: true });
                (0, log_1.log)('info', '[VectorDB] Meta file path:', metaFilePath);
                (0, log_1.log)('info', '[VectorDB] Vector file path:', vectorFilePath);
                const metaData = {};
                this.metadata.forEach((value, key) => {
                    // Ensure keys are strings for JSON compatibility
                    metaData[String(key)] = value;
                });
                const vectorInfo = [];
                const vectorBuffers = [];
                let currentOffset = 0;
                // 1. Prepare vector data and metadata structure
                (0, log_1.log)('info', `[VectorDB] Preparing vector data for saving with ${this.memoryStorage.size} vectors...`);
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
                (0, log_1.log)('info', `[VectorDB] Vector data prepared for saving: ${vectorInfo.length} vectors`);
                // 2. Write metadata file
                (0, log_1.log)('info', `[VectorDB] Writing metadata file to: ${metaFilePath} with ${vectorInfo.length} vectors`);
                // Ensure metadata is JSON-serializable
                let metaContent = JSON.stringify(saveData);
                if (this.useCompression) {
                    metaContent = await gzip(metaContent);
                }
                (0, log_1.log)('info', '[VectorDB] Writing meta file to:', metaFilePath);
                await fs_1.promises.writeFile(metaFilePath, metaContent);
                (0, log_1.log)('info', '[VectorDB] Meta file written successfully.');
                // 3. Write vector data file
                let vectorContent = Buffer.concat(vectorBuffers);
                if (this.useCompression) {
                    vectorContent = await gzip(vectorContent);
                }
                (0, log_1.log)('info', `[VectorDB] Writing vector file to: ${vectorFilePath} (${vectorBuffers.length} vectors, ${vectorContent.length} bytes)`);
                await fs_1.promises.writeFile(vectorFilePath, vectorContent);
                (0, log_1.log)('info', '[VectorDB] Vector file written successfully.');
                // 4. Emit save event
                this.emit('db:save', {
                    path: this.dbPath || 'DB path not set',
                    count: this.memoryStorage.size,
                });
                (0, log_1.log)('info', '[VectorDB] Save event emitted successfully.');
            }
            catch (error) {
                (0, log_1.log)('error', `Error saving database to ${this.dbPath}:`, error);
                throw error; // Re-throw to indicate failure
            }
            finally {
                this.savePromise = null; // Release lock
            }
        })();
        (0, log_1.log)('info', '[VectorDB] Save promise created.');
        return this.savePromise;
    }
    async load() {
        if (!this.dbPath) {
            throw new Error('Database path not specified for loading.');
        }
        if (this.isClosed) {
            throw new Error('Cannot load into a closed database.');
        }
        const metaFilePath = this._getMetaFilePath();
        const vectorFilePath = this._getVectorFilePath();
        // Check if files exist first to avoid unnecessary error logging for new databases
        const metaExists = (0, fs_1.existsSync)(metaFilePath);
        const vecExists = (0, fs_1.existsSync)(vectorFilePath);
        // If both files don't exist, this is likely a new database
        if (!metaExists && !vecExists) {
            (0, log_1.log)('info', `[VectorDB] Database files not found at ${this.dbPath}. Starting new database.`);
            return; // Exit early for a new database initialization
        }
        try {
            // 1. Read and parse metadata file
            let metaContentBuffer = await fs_1.promises.readFile(metaFilePath);
            if (this.useCompression) {
                metaContentBuffer = await gunzip(metaContentBuffer);
            }
            const saveData = JSON.parse(metaContentBuffer.toString('utf8'));
            if (saveData.version !== 1) {
                throw new Error(`Unsupported database format version: ${saveData.version}`);
            }
            // 2. Read vector data file
            let vectorDataBuffer = await fs_1.promises.readFile(vectorFilePath);
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
                    (0, log_1.log)('error', `Invalid offset/length for vector ${id}. Offset: ${offset}, Length: ${length}, Buffer Size: ${vectorDataBuffer.length}`);
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
            (0, log_1.log)('info', `[VectorDB] Loaded ${this.memoryStorage.size} vectors from ${this.dbPath}`);
        }
        catch (error) {
            if (error.code === 'ENOENT') {
                // Files not found is expected for a new DB, don't throw
                (0, log_1.log)('info', `Database files not found at ${this.dbPath}. Starting new database.`);
                return; // Don't re-throw ENOENT
            }
            (0, log_1.log)('error', `Error loading database from ${this.dbPath}:`, error);
            throw error; // Re-throw other errors
        }
    }
    // --- Stats and Lifecycle ---
    getStats() {
        // (Keep existing logic, ensure it uses this.vectorDimensions)
        const dimensionCounts = {};
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
        }
        catch (e) {
            (0, log_1.log)('warn', 'Could not estimate metadata size:', e);
        }
        const baseStats = {
            vectorCount: this.memoryStorage.size,
            vectorSize: this.vectorSize(),
            defaultVectorSize: this.defaultVectorSize ?? 0, // Use 0 if null
            metadataCount: this.metadata.size,
            dimensions: {
                counts: dimensionCounts,
                unique: Object.keys(dimensionCounts).length,
            },
            // More accurate memory usage estimate
            memoryUsage: vectorMemory +
                metadataMemory +
                this.memoryStorage.size * 16 + // Estimate Map overhead for vectors
                this.metadata.size * 16 + // Estimate Map overhead for metadata
                this.vectorDimensions.size * 8, // Estimate Map overhead for dimensions
            // Placeholder for cluster stats (filled by subclass)
            clusters: { count: 0, avgSize: 0, distribution: [], dimensions: {} },
        };
        // Compatibility check for older DBStats type if needed
        if (!baseStats.clusters.dimensions) {
            baseStats.clusters.dimensions = {};
        }
        return baseStats;
    }
    _estimateMemoryUsage() {
        // This method is now effectively replaced by the calculation within getStats()
        // Kept for potential internal use or backward compatibility if needed elsewhere
        let vectorMemory = 0;
        this.memoryStorage.forEach((vec) => (vectorMemory += vec.byteLength));
        let metadataMemory = 0;
        try {
            this.metadata.forEach((meta) => (metadataMemory += JSON.stringify(meta).length * 2));
        }
        catch (e) {
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
    getMetadataWithField(criteria, values, options // Optional limit for results
    ) {
        const results = [];
        // Handle object criteria format (new format)
        if (criteria !== null && typeof criteria === 'object' && !Array.isArray(criteria)) {
            const criteriaObj = criteria;
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
            (0, log_1.log)('warn', 'Values array length does not match fields array length. Some value checks will be ignored.');
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
    getVectorCount() {
        return this.memoryStorage.size;
    }
    /**
     * Extract relationships between vectors based on distance or custom criteria.
     *
     * @param threshold - The maximum distance between vectors to consider them related.
     * @param metric - Distance metric to use (e.g., 'cosine', 'euclidean').
     * @returns An array of relationships, where each relationship links two vector IDs, their distance, and optional metadata.
     */
    extractRelationships(threshold, metric = 'euclidean') {
        const relationships = [];
        // Iterate over all vectors
        const vectorEntries = Array.from(this.memoryStorage.entries());
        for (let i = 0; i < vectorEntries.length; i++) {
            const [id1, vector1] = vectorEntries[i];
            for (let j = i + 1; j < vectorEntries.length; j++) {
                const [id2, vector2] = vectorEntries[j];
                // Ensure dimension compatibility
                if (vector1.length !== vector2.length) {
                    (0, log_1.log)('warn', `Dimension mismatch between vector ${id1} and ${id2}, skipping.`);
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
        (0, log_1.log)('info', `[VectorDB] Extracted ${relationships.length} relationships.`);
        return relationships;
    }
    /**
     * Extract communities of related vectors based on distance threshold.
     * A community is a group of vectors where each vector is related to at least one other vector in the group.
     *
     * @param threshold - The maximum distance between vectors to consider them related
     * @param metric - Distance metric to use (e.g., 'cosine', 'euclidean')
     * @returns Array of communities, where each community is an array of related vector information
     */
    extractCommunities(threshold, metric = 'euclidean') {
        (0, log_1.log)('info', `[VectorDB] Extracting vector communities with threshold ${threshold}...`);
        // First build a graph representation where each vector is a node
        // and edges exist between vectors with distance <= threshold
        const graph = new Map();
        const vectorEntries = Array.from(this.memoryStorage.entries());
        // Initialize the graph with empty adjacency lists
        for (const [id] of vectorEntries) {
            graph.set(id, new Set());
        }
        // Build edges
        for (let i = 0; i < vectorEntries.length; i++) {
            const [id1, vector1] = vectorEntries[i];
            for (let j = i + 1; j < vectorEntries.length; j++) {
                const [id2, vector2] = vectorEntries[j];
                // Ensure dimension compatibility
                if (vector1.length !== vector2.length) {
                    continue;
                }
                // Calculate distance
                const distance = this._calculateDistance(vector1, vector2, metric);
                // Add edge if distance is within threshold
                if (distance <= threshold) {
                    graph.get(id1)?.add(id2);
                    graph.get(id2)?.add(id1);
                }
            }
        }
        // Use depth-first search to find connected components (communities)
        const visited = new Set();
        const communities = [];
        for (const [id] of graph.entries()) {
            if (!visited.has(id)) {
                const community = [];
                // DFS to find all connected vectors
                const dfs = (nodeId) => {
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
        (0, log_1.log)('info', `[VectorDB] Found ${communities.length} communities`);
        return communities;
    }
    async close() {
        if (this.isClosed)
            return;
        this.isClosed = true; // Mark as closed immediately
        try {
            if (this.dbPath) {
                await this.save(); // Attempt to save on close
            }
        }
        catch (error) {
            (0, log_1.log)('error', 'Error saving database during close:', error);
        }
        finally {
            // Clear memory regardless of save success
            this.memoryStorage.clear();
            this.metadata.clear();
            this.vectorDimensions.clear();
            this.emit('db:close', {});
            (0, log_1.log)('info', 'Database closed.');
        }
    }
}
exports.VectorDB = VectorDB;
// --- END OF FILE vector_db.ts ---
//# sourceMappingURL=vector_db.js.map