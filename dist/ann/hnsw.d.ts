import { BuildIndexHNSWOptions, HNSWOptions, HNSWStats, LoadIndexHNSWOptions, SearchOptions, SearchResult, Vector } from '../types';
import { ClusteredVectorDB } from '../vector/clustered_vector_db';
/**
 * Hierarchical Navigable Small World (HNSW) graph for approximate nearest neighbor search
 * Optimized for performance
 */
/**
 * Hierarchical Navigable Small World (HNSW) index implementation for approximate nearest neighbor search.
 *
 * HNSW is an algorithm for efficient approximate nearest neighbor search in high-dimensional spaces.
 * It creates a multi-layered graph structure that allows for faster search by navigating through
 * a hierarchy of increasingly dense graphs.
 *
 * Key features:
 * - Dimension-aware mode: Optimizes searches for vectors of the same dimension
 * - Efficient incremental updates: Add/remove vectors without rebuilding the entire index
 * - Configurable precision via efConstruction and efSearch parameters
 * - Soft deletion support: Vectors can be marked for deletion without rebuilding
 * - Serialization/deserialization for persistent storage
 *
 * The implementation is optimized for both memory efficiency and search performance,
 * with specialized handling for different vector dimensions when in dimension-aware mode.
 *
 * @example
 * ```typescript
 * // Create a new HNSW index
 * const hnsw = new HNSW(vectorDatabase, {
 *   M: 16,                // Max connections per node (default: 16)
 *   efConstruction: 200,  // Size of dynamic candidate list during construction (default: 200)
 *   efSearch: 50,         // Size of dynamic candidate list during search (default: 50)
 *   dimensionAware: true  // Whether to optimize for vectors of the same dimension (default: true)
 * });
 *
 * // Build the index with all vectors in the database
 * await hnsw.buildIndex({
 *   progressCallback: (progress) => console.log(`Indexing: ${progress * 100}%`)
 * });
 *
 * // Search for nearest neighbors
 * const results = hnsw.findNearest(queryVector, 10);
 * ```
 */
declare class HNSW {
    private db;
    private M;
    private efConstruction;
    private efSearch;
    private maxLevel;
    private levelProbability;
    private distanceFunc;
    private entryPointId;
    private nodes;
    private nodeToLevel;
    private nodeDimensions;
    private dimensionGroups;
    private dimensionEntryPoints;
    private timer;
    private initialized;
    private dimensionAware;
    private deletedNodes;
    constructor(db: ClusteredVectorDB, options?: HNSWOptions);
    /**
     * Add a vector to the HNSW graph
     * @param id - Vector identifier
     * @param vector - Vector to add
     * @returns Added vector ID
     */
    addVector(id: number | string, vector: Vector): number | string;
    /**
     * Mark a vector as deleted in the HNSW graph
     * This method marks nodes for deletion without immediately removing them from the graph
     * @param id - Vector identifier to mark as deleted
     * @returns True if the vector was marked for deletion, false if not found
     */
    markDelete(id: number | string): boolean;
    /**
     * Update entry point after the current entry point was deleted
     * @private
     */
    private _updateEntryPointAfterDeletion;
    /**
     * Update dimension entry point after the current entry point for that dimension was deleted
     * @private
     */
    private _updateDimensionEntryPointAfterDeletion;
    /**
     * Search with a specific entry point
     * @private
     */
    private _searchWithEntryPoint;
    /**
     * Add a single point to the HNSW index
     * For incremental updates to an existing index
     * @param vector - Vector to add
     * @param id - Vector identifier
     * @returns Added vector ID
     */
    addPoint(vector: Vector, id: number | string): number | string;
    /**
     * Get the number of nodes in the HNSW graph
     * @returns Number of nodes
     */
    getNodeCount(): number;
    /**
     * Find k nearest neighbors to the query vector
     * @param query - Query vector
     * @param k - Number of neighbors to find
     * @param options - Search options
     * @returns Array of nearest neighbors
     */
    findNearest(query: Vector, k?: number, options?: SearchOptions & {
        exactDimensions?: boolean;
    }): SearchResult[];
    /**
     * Fallback linear search implementation
     * @private
     */
    private _linearSearch;
    /**
     * Build the HNSW index for all vectors in the database
     * @param options - Build options
     */
    buildIndex(options?: BuildIndexHNSWOptions): Promise<void>;
    /**
     * Create a new node in the graph
     * @private
     */
    private _createNode;
    /**
     * Get connections for a node at a specific level
     * @private
     */
    private _getConnections;
    /**
     * Add bidirectional connections between nodes
     * @private
     */
    private _addConnectionsForNode;
    /**
     * Select up to M nearest neighbors for a node
     * @private
     */
    private _selectNeighbors;
    /**
     * Prune connections to maintain at most M connections per node
     * @private
     */
    private _pruneConnections;
    /**
     * Calculate random level for a new node
     * @private
     */
    private _randomLevel;
    /**
     * Calculate distance between two nodes
     * @private
     */
    private _distance;
    /**
     * Calculate distance from query to a node
     * @private
     */
    private _distanceToQuery;
    /**
     * Get HNSW statistics
     * @returns Stats object with graph information
     */
    getStats(): HNSWStats;
    /**
     * Serialize HNSW graph to JSON
     * @returns JSON string representation of the graph
     */
    serialize(): string;
    /**
     * Deserialize HNSW graph from JSON
     * @param json - JSON string representation of the graph
     * @returns HNSW instance
     */
    static deserialize(json: string, db: ClusteredVectorDB): HNSW;
    /**
     * Save HNSW index to disk
     * @param filePath - Path to save the index
     */
    saveIndex(filePath: string): Promise<void>;
    /**
     * Load HNSW index from disk
     * @param filePath - Path to load the index from
     * @param db - Vector database
     */
    static loadIndex(filePath: string, db: ClusteredVectorDB, options?: LoadIndexHNSWOptions): Promise<HNSW>;
    /**
     * Clean up resources
     */
    close(): void;
}
export default HNSW;
