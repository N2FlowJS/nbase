import { Vector, DBStats, SearchResult, VectorData, DistanceMetric, TypedEventEmitter, VectorDBEventData } from '../types';
declare const VectorDB_base: new () => TypedEventEmitter<VectorDBEventData>;
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
export declare class VectorDB extends VectorDB_base {
    defaultVectorSize: number | null;
    memoryStorage: Map<number | string, Float32Array>;
    protected metadata: Map<number | string, Record<string, any>>;
    protected vectorDimensions: Map<number | string, number>;
    protected idCounter: number;
    protected dbPath: string | null;
    protected savePromise: Promise<void> | null;
    protected isClosed: boolean;
    protected useCompression: boolean;
    isReady: boolean;
    constructor(suggestedVectorSize?: number | null, dbPath?: string | null, options?: {
        useCompression?: boolean;
    });
    getIdCounter(): number;
    setPath(dbPath: string): void;
    vectorSize(): number;
    addVector(id: number | string | undefined, vector: Vector, metadata?: Record<string, any>): number | string;
    bulkAdd(vectors: VectorData[]): number;
    getVector(id: number | string): Float32Array | null;
    hasVector(id: number | string): boolean;
    deleteVector(id: number | string): boolean;
    updateVector(id: number | string, vector: Vector): boolean;
    addMetadata(id: number | string, data: Record<string, any>): void;
    getMetadata(id: number | string): Record<string, any> | null;
    updateMetadata(id: number | string, data: Record<string, any> | ((current: Record<string, any> | null) => Record<string, any>)): boolean;
    getVectorDimension(id: number | string): number | null;
    protected _calculateNorm(vector: Float32Array): number;
    protected _dotProduct(a: Float32Array, b: Float32Array): number;
    protected _euclideanDistance(a: Float32Array, b: Float32Array): number;
    protected _cosineDistance(a: Float32Array, b: Float32Array): number;
    protected _calculateDistance(a: Float32Array, b: Float32Array, metric: DistanceMetric): number;
    findNearest(query: Vector, k?: number, options?: {
        filter?: (id: number | string, metadata?: Record<string, any>) => boolean;
        metric?: DistanceMetric;
    }): SearchResult[];
    protected _linearSearch(query: Float32Array, k: number, metric: DistanceMetric, filter?: (id: number | string, metadata?: Record<string, any>) => boolean): SearchResult[];
    protected _getMetaFilePath(): string;
    protected _getVectorFilePath(): string;
    save(): Promise<void>;
    load(): Promise<void>;
    getStats(): DBStats;
    protected _estimateMemoryUsage(): number;
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
    getMetadataWithField(criteria: string | string[] | Record<string, any>, values?: any | any[], options?: {
        limit?: number;
    }): Array<{
        id: number | string;
        metadata: Record<string, any>;
    }>;
    getVectorCount(): number;
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
     * A community is a group of vectors where each vector is related to at least one other vector in the group.
     *
     * @param threshold - The maximum distance between vectors to consider them related
     * @param metric - Distance metric to use (e.g., 'cosine', 'euclidean')
     * @returns Array of communities, where each community is an array of related vector information
     */
    extractCommunities(threshold: number, metric?: DistanceMetric): Array<Array<{
        id: number | string;
        metadata?: Record<string, any>;
    }>>;
    close(): Promise<void>;
}
export {};
