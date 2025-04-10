import { ClusteredVectorDB } from '../vector/clustered_vector_db';
import { Vector, SearchResult, SearchOptions, HNSWNode, HNSWOptions, BuildIndexHNSWOptions, HNSWStats, LoadIndexHNSWOptions } from '../types';
import { createTimer } from '../utils/profiling';
import { promises as fs } from 'fs';
import { log } from 'console';

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
class HNSW {
  private db: ClusteredVectorDB;
  private M: number;
  private efConstruction: number;
  private efSearch: number;
  private maxLevel: number;
  private levelProbability: number;
  private distanceFunc: (a: Vector, b: Vector) => number;
  private entryPointId: number | string | null;
  private nodes: Map<number | string, HNSWNode>;
  private nodeToLevel: Map<number | string, number>;
  private nodeDimensions: Map<number | string, number>;
  private dimensionGroups: Map<number, Set<number | string>>;
  private dimensionEntryPoints: Map<number, number | string>;
  private timer: ReturnType<typeof createTimer>;
  private initialized: boolean;
  private dimensionAware: boolean;
  private deletedNodes: Set<number | string>; // Track deleted nodes

  constructor(db: ClusteredVectorDB, options: HNSWOptions = {}) {
    this.db = db;

    // Set HNSW parameters
    this.M = options.M || 16;
    this.efConstruction = options.efConstruction || 200;
    this.efSearch = options.efSearch || 50;
    this.maxLevel = options.maxLevel || 16;
    this.levelProbability = options.levelProbability || 0.5;
    this.entryPointId = options.entryPointId || null;
    this.dimensionAware = options.dimensionAware !== false;

    // Customizable distance function - Inlined Euclidean Distance for performance
    this.distanceFunc = (a: Vector, b: Vector) => {
      let sum = 0;
      const len = Math.min(a.length, b.length);
      // Inlined loop for performance
      for (let i = 0; i < len; i++) {
        const diff = a[i] - b[i];
        sum += diff * diff;
      }
      // Dimension penalty - only if dimensionAware
      if (this.dimensionAware) {
        const dimDiff = Math.abs(a.length - b.length);
        if (dimDiff > 0) {
          sum += dimDiff * 0.01;
        }
      }
      return Math.sqrt(sum);
    };

    // Initialize data structures
    this.nodes = new Map();
    this.nodeToLevel = new Map();
    this.nodeDimensions = new Map();
    this.dimensionGroups = new Map();
    this.dimensionEntryPoints = new Map();
    this.timer = createTimer();
    this.initialized = false;
    this.deletedNodes = new Set(); // Initialize deletedNodes set
  }

  /**
   * Add a vector to the HNSW graph
   * @param id - Vector identifier
   * @param vector - Vector to add
   * @returns Added vector ID
   */
  addVector(id: number | string, vector: Vector): number | string {
    // Verify the vector exists in the database (optimized get)
    const dbVector = this.db.getVector(id); // No need for || vector, assume vector is in DB
    if (!dbVector) {
      throw new Error(`Vector with id ${id} not found in database`);
    }

    // Store vector dimension
    const dimension = dbVector.length;
    this.nodeDimensions.set(id, dimension);

    // Add to dimension group (optimized Set operations)
    let dimensionSet = this.dimensionGroups.get(dimension);
    if (!dimensionSet) {
      dimensionSet = new Set();
      this.dimensionGroups.set(dimension, dimensionSet);
    }
    dimensionSet.add(id);

    // If this is the first vector in this dimension group, make it the entry point for this dimension
    if (dimensionSet.size === 1) {
      this.dimensionEntryPoints.set(dimension, id);
    }

    // If this is the first vector overall, make it the global entry point
    if (!this.entryPointId) {
      this.entryPointId = id;
      const level = this._randomLevel();
      this._createNode(id, level, dimension);
      this.initialized = true;
      return id;
    }

    // Random level for new node
    const randLevel = this._randomLevel();
    this._createNode(id, randLevel, dimension);

    // Connect the new node into the graph
    // For dimension-aware mode, use entry point from the same dimension group if available
    let entryPointId = this.dimensionAware ? this.dimensionEntryPoints.get(dimension) || this.entryPointId : this.entryPointId;

    // Only proceed with graph building if we have an entry point with the same dimension
    // or if we're not in dimension-aware mode
    if (entryPointId) {
      let currObj = entryPointId;
      let currDist = this._distance(id, entryPointId);

      // Get max level in the graph (optimized level retrieval)
      const entryLevel = this.nodeToLevel.get(entryPointId) || 0;

      // Work down from the entry level to the level of the new node
      for (let level = Math.min(entryLevel, randLevel); level >= 0; level--) {
        // Find closest neighbors at the current level
        let changed = true;

        // Greedy search for the closest element
        while (changed) {
          changed = false;

          // Get node connections at level (optimized retrieval)
          const neighbors = this._getConnections(currObj, level);

          for (const neighborId of neighbors) {
            // Skip neighbors with different dimensions in dimension-aware mode (optimized dimension check)
            if (this.dimensionAware && this.nodeDimensions.get(neighborId) !== dimension) {
              continue;
            }

            const dist = this._distance(id, neighborId);

            if (dist < currDist) {
              currDist = dist;
              currObj = neighborId;
              changed = true;
            }
          }
        }

        if (level <= randLevel) {
          // Add edges at this level
          this._addConnectionsForNode(id, level, currObj);
        }
      }

      // Update dimension entry point if new node is at a higher level
      const dimEntryPoint = this.dimensionEntryPoints.get(dimension);
      if (dimEntryPoint && randLevel > (this.nodeToLevel.get(dimEntryPoint) || 0)) {
        this.dimensionEntryPoints.set(dimension, id);
      }

      // Update global entry point if new node is at a higher level
      if (randLevel > (this.nodeToLevel.get(this.entryPointId) || 0)) {
        this.entryPointId = id;
      }
    }

    return id;
  }

  /**
   * Mark a vector as deleted in the HNSW graph
   * This method marks nodes for deletion without immediately removing them from the graph
   * @param id - Vector identifier to mark as deleted
   * @returns True if the vector was marked for deletion, false if not found
   */
  markDelete(id: number | string): boolean {
    if (!this.nodes.has(id)) {
      console.warn(`[HNSW] Vector with id ${id} not found to delete`);
      return false;
    }

    // Mark the node as deleted
    this.deletedNodes.add(id);

    // If the deleted node is the entry point, find a new entry point
    if (this.entryPointId === id) {
      console.log(`[HNSW] Entry point ${id} was deleted, finding new entry point...`);
      this._updateEntryPointAfterDeletion();
    }

    // If the deleted node is a dimension entry point, update that too
    const nodeDimension = this.nodeDimensions.get(id);
    if (nodeDimension !== undefined && this.dimensionEntryPoints.get(nodeDimension) === id) {
      console.log(`[HNSW] Dimension entry point for dimension ${nodeDimension} was deleted, finding new entry point...`);
      this._updateDimensionEntryPointAfterDeletion(nodeDimension);
    }

    return true;
  }

  /**
   * Update entry point after the current entry point was deleted
   * @private
   */
  private _updateEntryPointAfterDeletion(): void {
    // Reset entry point
    this.entryPointId = null;

    // Find the node with the highest level that isn't deleted
    let maxLevel = -1;
    let newEntryPoint: number | string | null = null;

    for (const [nodeId, level] of this.nodeToLevel.entries()) {
      if (this.deletedNodes.has(nodeId)) continue;

      if (level > maxLevel) {
        maxLevel = level;
        newEntryPoint = nodeId;
      }
    }

    // Set new entry point if found
    if (newEntryPoint !== null) {
      this.entryPointId = newEntryPoint;
      console.log(`[HNSW] New entry point set to ${newEntryPoint} with level ${maxLevel}`);
    } else {
      console.warn('[HNSW] No valid entry point found after deletion.');
    }
  }

  /**
   * Update dimension entry point after the current entry point for that dimension was deleted
   * @private
   */
  private _updateDimensionEntryPointAfterDeletion(dimension: number): void {
    // Get the set of nodes for this dimension
    const dimensionNodes = this.dimensionGroups.get(dimension);
    if (!dimensionNodes || dimensionNodes.size === 0) {
      // No more nodes in this dimension
      this.dimensionEntryPoints.delete(dimension);
      console.log(`[HNSW] No more nodes in dimension ${dimension}, removing entry point.`);
      return;
    }

    // Find a new entry point for this dimension (highest level node)
    let maxLevel = -1;
    let newEntryPoint: number | string | null = null;

    for (const nodeId of dimensionNodes) {
      if (this.deletedNodes.has(nodeId)) continue;

      const level = this.nodeToLevel.get(nodeId) || 0;
      if (level > maxLevel) {
        maxLevel = level;
        newEntryPoint = nodeId;
      }
    }

    // Set new dimension entry point if found
    if (newEntryPoint !== null) {
      this.dimensionEntryPoints.set(dimension, newEntryPoint);
      console.log(`[HNSW] New dimension ${dimension} entry point set to ${newEntryPoint} with level ${maxLevel}`);
    } else {
      this.dimensionEntryPoints.delete(dimension);
      console.warn(`[HNSW] No valid entry point found for dimension ${dimension} after deletion.`);
    }
  }

  /**
   * Search with a specific entry point
   * @private
   */
  private _searchWithEntryPoint(entryPoint: number | string, query: Vector, k: number, options: SearchOptions & { exactDimensions?: boolean } = {}): SearchResult[] {
    const timer = this.timer;
    const queryDimension = query.length;
    const exactDimensions = options.exactDimensions || false;
    const filter = options.filter || (() => true);

    // Get entry level (optimized level retrieval)
    const entryLevel = this.nodeToLevel.get(entryPoint) || 0;

    let currObj = entryPoint;
    let currDist = this._distanceToQuery(query, entryPoint);

    // Search from top level down
    for (let i = entryLevel; i > 0; i--) {
      // Greedy search at this level
      let changed = true;

      while (changed) {
        changed = false;

        // Get node connections at level (optimized retrieval)
        const neighbors = this._getConnections(currObj, i);

        for (const neighborId of neighbors) {
          // Skip deleted nodes
          if (this.deletedNodes.has(neighborId)) continue;

          // Skip neighbors with different dimensions if exactDimensions is true (optimized dimension check)
          if (exactDimensions) {
            const neighborDim = this.nodeDimensions.get(neighborId);
            if (neighborDim !== queryDimension) continue;
          }

          // Skip if filter excludes this ID
          if (!filter(neighborId)) continue;

          const dist = this._distanceToQuery(query, neighborId);

          if (dist < currDist) {
            currDist = dist;
            currObj = neighborId;
            changed = true;
          }
        }
      }
    }

    // Beam search at the bottom level
    const ef = Math.max(k, this.efSearch);
    const visited = new Set<number | string>();
    const candidates = new Map<number | string, number>(); // id -> distance
    const results = new Map<number | string, number>(); // id -> distance

    // Initialize with entry point
    candidates.set(currObj, currDist);
    results.set(currObj, currDist);
    visited.add(currObj);

    // Main loop
    while (candidates.size > 0) {
      // Find closest candidate
      let closest: number | string | null = null;
      let minDist = Infinity;
      for (const [id, dist] of candidates.entries()) {
        if (dist < minDist) {
          minDist = dist;
          closest = id;
        }
      }
      if (closest === null) break;

      // If furthest result is closer than closest candidate, we're done
      if (results.size >= ef) {
        let furthestResultDist = -Infinity;
        for (const dist of results.values()) {
          if (dist > furthestResultDist) {
            furthestResultDist = dist;
          }
        }
        if (minDist > furthestResultDist) {
          break;
        }
      }

      // Remove from candidates
      candidates.delete(closest);

      // Get nearest neighbors for this candidate
      const connections = this._getConnections(closest, 0);

      for (const neighborId of connections) {
        // Skip deleted nodes
        if (this.deletedNodes.has(neighborId)) continue;

        // Skip neighbors with different dimensions if exactDimensions is true
        if (exactDimensions) {
          const neighborDim = this.nodeDimensions.get(neighborId);
          if (neighborDim !== queryDimension) continue;
        }

        // Skip if filter excludes this ID
        if (!filter(neighborId)) continue;

        if (!visited.has(neighborId)) {
          visited.add(neighborId);
          const dist = this._distanceToQuery(query, neighborId);

          // Add to results if results is not full or if it's closer than furthest result
          let furthestResultDist = -Infinity;
          let furthestId: number | string | null = null;

          if (results.size >= ef) {
            // Find furthest result
            for (const [resultId, resultDist] of results.entries()) {
              if (resultDist > furthestResultDist) {
                furthestResultDist = resultDist;
                furthestId = resultId;
              }
            }

            if (dist < furthestResultDist) {
              // Replace furthest result
              if (furthestId !== null) {
                results.delete(furthestId);
              }
              results.set(neighborId, dist);
              candidates.set(neighborId, dist);
            }
          } else {
            // Results not full yet
            results.set(neighborId, dist);
            candidates.set(neighborId, dist);
          }
        }
      }
    }

    // Convert results to array and sort by distance
    let resultsList = Array.from(results.entries())
      .sort((a, b) => a[1] - b[1])
      .map(([id, dist]) => ({ id, dist }));

    // Take top k results
    resultsList = resultsList.slice(0, k);

    timer.stop('hnsw_search');
    return resultsList;
  }

  /**
   * Add a single point to the HNSW index
   * For incremental updates to an existing index
   * @param vector - Vector to add
   * @param id - Vector identifier
   * @returns Added vector ID
   */
  addPoint(vector: Vector, id: number | string): number | string {
    // Skip if the node is already in the index
    if (this.nodes.has(id)) {
      console.warn(`[HNSW] Vector with id ${id} already exists in the index`);
      return id;
    }

    // Remove from deleted nodes if it was marked as deleted
    if (this.deletedNodes.has(id)) {
      this.deletedNodes.delete(id);
      console.log(`[HNSW] Vector with id ${id} was previously deleted, undeleting it`);
    }

    // Store vector dimension
    const dimension = vector.length;
    this.nodeDimensions.set(id, dimension);

    // Add to dimension group
    let dimensionSet = this.dimensionGroups.get(dimension);
    if (!dimensionSet) {
      dimensionSet = new Set();
      this.dimensionGroups.set(dimension, dimensionSet);
    }
    dimensionSet.add(id);

    // If this is the first vector in this dimension group, make it the entry point for this dimension
    if (dimensionSet.size === 1) {
      this.dimensionEntryPoints.set(dimension, id);
    }

    // If this is the first vector overall, make it the global entry point
    if (!this.entryPointId) {
      this.entryPointId = id;
      const level = this._randomLevel();
      this._createNode(id, level, dimension);
      this.initialized = true;
      return id;
    }

    // Random level for new node
    const randLevel = this._randomLevel();
    this._createNode(id, randLevel, dimension);

    // Connect the new node into the graph
    // For dimension-aware mode, use entry point from the same dimension group if available
    let entryPointId = this.dimensionAware ? this.dimensionEntryPoints.get(dimension) || this.entryPointId : this.entryPointId;

    // Only proceed with graph building if we have an entry point with the same dimension
    // or if we're not in dimension-aware mode
    if (entryPointId) {
      let currObj = entryPointId;
      let currDist = this.distanceFunc(vector, this.db.getVector(entryPointId) || vector);

      // Get max level in the graph
      const entryLevel = this.nodeToLevel.get(entryPointId) || 0;

      // Work down from the entry level to the level of the new node
      for (let level = Math.min(entryLevel, randLevel); level >= 0; level--) {
        // Find closest neighbors at the current level
        let changed = true;

        // Greedy search for the closest element
        while (changed) {
          changed = false;

          // Get node connections at level
          const neighbors = this._getConnections(currObj, level);

          for (const neighborId of neighbors) {
            // Skip neighbors with different dimensions in dimension-aware mode
            if (this.dimensionAware && this.nodeDimensions.get(neighborId) !== dimension) {
              continue;
            }

            const neighborVector = this.db.getVector(neighborId);
            if (!neighborVector) continue;

            const dist = this.distanceFunc(vector, neighborVector);

            if (dist < currDist) {
              currDist = dist;
              currObj = neighborId;
              changed = true;
            }
          }
        }

        if (level <= randLevel) {
          // Add edges at this level based on closest elements
          this._addConnectionsForNode(id, level, currObj);
        }
      }

      // Update dimension entry point if new node is at a higher level
      const dimEntryPoint = this.dimensionEntryPoints.get(dimension);
      if (dimEntryPoint && randLevel > (this.nodeToLevel.get(dimEntryPoint) || 0)) {
        this.dimensionEntryPoints.set(dimension, id);
      }

      // Update global entry point if new node is at a higher level
      if (randLevel > (this.nodeToLevel.get(this.entryPointId) || 0)) {
        this.entryPointId = id;
      }
    }

    return id;
  }

  /**
   * Get the number of nodes in the HNSW graph
   * @returns Number of nodes
   */
  getNodeCount(): number {
    return this.nodes.size;
  }

  /**
   * Find k nearest neighbors to the query vector
   * @param query - Query vector
   * @param k - Number of neighbors to find
   * @param options - Search options
   * @returns Array of nearest neighbors
   */
  findNearest(query: Vector, k: number = 10, options: SearchOptions & { exactDimensions?: boolean } = {}): SearchResult[] {
    if (!this.entryPointId || !this.initialized) {
      // Fall back to linear search (optimized linear search call)
      return this._linearSearch(query, k, options);
    }

    const timer = this.timer;
    timer.start('hnsw_search');

    const queryDimension = query.length;
    const exactDimensions = options.exactDimensions || false;

    // Modify search methods to filter out deleted nodes
    const originalFilter = options.filter;
    options.filter = (id) => {
      // Skip deleted nodes and apply the original filter if it exists
      return !this.deletedNodes.has(id) && (originalFilter ? originalFilter(id) : true);
    };

    // For dimension-aware search with exact dimension matching
    if (this.dimensionAware && exactDimensions) {
      // Get entry point for this dimension (optimized map get)
      const dimensionEntryPoint = this.dimensionEntryPoints.get(queryDimension);

      // If we don't have an entry point for this dimension, return empty results
      if (!dimensionEntryPoint) {
        timer.stop('hnsw_search');
        return [];
      }

      // Perform search using the dimension-specific entry point (optimized search call)
      return this._searchWithEntryPoint(dimensionEntryPoint, query, k, options);
    }

    // Standard search using global entry point (optimized search call)
    return this._searchWithEntryPoint(this.entryPointId, query, k, options);
  }

  /**
   * Fallback linear search implementation
   * @private
   */
  private _linearSearch(query: Vector, k: number, options: SearchOptions = {}): SearchResult[] {
    const filter = options.filter || (() => true);
    const queryDimension = query.length;
    const results: SearchResult[] = [];

    // Optimized linear scan using for...of and direct Map iteration
    for (const [id, vector] of this.db.memoryStorage.entries()) {
      // Skip if filter excludes this ID (optimized filter call)
      if (!filter(id)) continue;
      if (vector.length !== queryDimension) continue;

      const dist = this.distanceFunc(query, vector);
      results.push({ id, dist });
    }

    return results.sort((a, b) => a.dist - b.dist).slice(0, k);
  }

  /**
   * Build the HNSW index for all vectors in the database
   * @param options - Build options
   */
  async buildIndex(options: BuildIndexHNSWOptions = {}): Promise<void> {
    const progressCallback = options.progressCallback || (() => {});
    const dimensionAware = options.dimensionAware !== false;

    // Reset the index (optimized clear operations)
    this.nodes.clear();
    this.nodeToLevel.clear();
    this.nodeDimensions.clear();
    this.dimensionGroups.clear();
    this.dimensionEntryPoints.clear();
    this.entryPointId = null;
    this.initialized = false;
    this.dimensionAware = dimensionAware;
    this.deletedNodes.clear(); // Clear deleted nodes

    // Get all vector IDs from the database (optimized key retrieval)
    const ids = Array.from(this.db.memoryStorage.keys());
    const totalVectors = ids.length;

    if (totalVectors === 0) {
      console.log('No vectors to index');
      return;
    }

    // First, scan all vectors to collect dimensions (optimized loop)
    for (let i = 0; i < ids.length; i++) {
      const id = ids[i];
      const vector = this.db.getVector(id); // Optimized vector retrieval

      if (vector) {
        const dimension = vector.length;
        this.nodeDimensions.set(id, dimension);

        // Group vectors by dimension (optimized Set operations)
        let dimensionSet = this.dimensionGroups.get(dimension);
        if (!dimensionSet) {
          dimensionSet = new Set();
          this.dimensionGroups.set(dimension, dimensionSet);
        }
        dimensionSet.add(id);
      }

      // Report progress (optimized modulo operation)
      if (i % 1000 === 0) {
        progressCallback((i / totalVectors) * 0.1); // First 10% for dimension scanning
      }
    }

    // Log dimension stats
    console.log(`Building HNSW index with ${this.dimensionGroups.size} different dimensions`);
    for (const [dimension, ids] of this.dimensionGroups.entries()) {
      console.log(`Dimension ${dimension}: ${ids.size} vectors`);
    }

    // Now build the index
    let processedCount = 0;

    // If dimension-aware, process each dimension group separately to avoid cross-dimension connections
    if (dimensionAware) {
      for (const [dimension, ids] of this.dimensionGroups.entries()) {
        // Optimized Map iteration
        const dimensionIds = Array.from(ids);

        // Process vectors in this dimension (optimized loop)
        for (let i = 0; i < dimensionIds.length; i++) {
          const id = dimensionIds[i];
          const vector = this.db.getVector(id); // Optimized vector retrieval

          if (vector) {
            this.addVector(id, vector);
          }

          processedCount++;

          // Report progress (optimized modulo operation)
          if (processedCount % 100 === 0) {
            const progress = 0.1 + (processedCount / totalVectors) * 0.9; // 10-100%
            progressCallback(progress);
          }
        }
      }
    } else {
      // Process all vectors regardless of dimension (optimized loop)
      for (let i = 0; i < ids.length; i++) {
        const id = ids[i];
        const vector = this.db.getVector(id); // Optimized vector retrieval

        if (vector) {
          this.addVector(id, vector);
        }

        // Report progress (optimized modulo operation)
        if (i % 100 === 0) {
          const progress = 0.1 + (i / totalVectors) * 0.9; // 10-100%
          progressCallback(progress);
        }
      }
    }

    this.initialized = true;
    progressCallback(1.0); // 100% complete
  }

  /**
   * Create a new node in the graph
   * @private
   */
  private _createNode(id: number | string, level: number, dimension?: number): void {
    // Initialize connections for each level (optimized Map and Set creation)
    const node: HNSWNode = {
      id,
      connections: new Map(),
      dimension,
    };

    // Create empty connection sets for each level (optimized loop)
    for (let i = 0; i <= level; i++) {
      node.connections.set(i, new Set());
    }

    this.nodes.set(id, node);
    this.nodeToLevel.set(id, level);
    if (dimension !== undefined) {
      this.nodeDimensions.set(id, dimension);
    }
  }

  /**
   * Get connections for a node at a specific level
   * @private
   */
  private _getConnections(id: number | string, level: number): Set<number | string> {
    const node = this.nodes.get(id); // Optimized Map get
    if (!node) return new Set();

    const connections = node.connections.get(level); // Optimized Map get
    return connections || new Set();
  }

  /**
   * Add bidirectional connections between nodes
   * @private
   */
  private _addConnectionsForNode(newId: number | string, level: number, currId: number | string): void {
    // Get the dimension of the new node (optimized dimension retrieval)
    const newNodeDimension = this.nodeDimensions.get(newId);

    // Find nearest neighbors for the new node at this level
    const closestIds = this._selectNeighbors(newId, level, currId);
    const newNode = this.nodes.get(newId); // Optimized Map get

    if (!newNode) return;

    // Get connections at this level for the new node (optimized retrieval and creation)
    let newConnections = newNode.connections.get(level);
    if (!newConnections) {
      newConnections = new Set();
      newNode.connections.set(level, newConnections);
    }

    // Add connections from new node to its neighbors (optimized loop)
    for (const closestId of closestIds) {
      // Skip connections between different dimensions in dimension-aware mode (optimized dimension check)
      if (this.dimensionAware) {
        const closestDimension = this.nodeDimensions.get(closestId);
        if (closestDimension !== newNodeDimension) {
          continue;
        }
      }

      newConnections.add(closestId); // Optimized Set add

      // Add backlink from neighbor to new node
      const neighborNode = this.nodes.get(closestId); // Optimized Map get
      if (neighborNode) {
        let neighborConnections = neighborNode.connections.get(level); // Optimized retrieval and creation
        if (!neighborConnections) {
          neighborConnections = new Set();
          neighborNode.connections.set(level, neighborConnections);
        }

        neighborConnections.add(newId); // Optimized Set add

        // Prune connections if needed
        this._pruneConnections(neighborNode, level);
      }
    }

    // Prune connections for the new node if needed
    this._pruneConnections(newNode, level);
  }

  /**
   * Select up to M nearest neighbors for a node
   * @private
   */
  private _selectNeighbors(id: number | string, level: number, entryPointId: number | string): Array<number | string> {
    // Get the dimension of the node (optimized dimension retrieval)
    const nodeDimension = this.nodeDimensions.get(id);

    // Find ef_construction nearest neighbors for the node
    const candidates = new Map<number | string, number>(); // id -> distance (optimized Map usage)
    const visited = new Set<number | string>(); // Optimized Set usage

    // Start with the entry point
    const entryDist = this._distance(id, entryPointId);
    candidates.set(entryPointId, entryDist); // Optimized Map set
    visited.add(entryPointId); // Optimized Set add

    // Beam search
    const results = new Map<number | string, number>(); // id -> distance (optimized Map usage)
    results.set(entryPointId, entryDist); // Optimized Map set

    // Main loop (optimized loop and candidate selection)
    while (candidates.size > 0) {
      // Find closest candidate (optimized candidate selection)
      let closest: number | string | null = null;
      let minDist = Infinity;
      for (const [candidateId, dist] of candidates.entries()) {
        // Optimized Map iteration
        if (dist < minDist) {
          minDist = dist;
          closest = candidateId;
        }
      }
      if (closest === null) break;

      // Check if we should stop search (optimized result comparison)
      let furthestResultDist = -Infinity;
      if (results.size >= this.efConstruction) {
        // Only calculate if needed
        for (const dist of results.values()) {
          // Optimized Map value iteration
          if (dist > furthestResultDist) {
            furthestResultDist = dist;
          }
        }
        if (minDist > furthestResultDist && results.size >= this.efConstruction) {
          break;
        }
      }

      // Remove from candidates (optimized Map delete)
      candidates.delete(closest);

      // Get nearest neighbors for this candidate (optimized retrieval)
      const connections = this._getConnections(closest, level);

      for (const neighborId of connections) {
        // Skip connections to nodes with different dimensions in dimension-aware mode (optimized dimension check)
        if (this.dimensionAware) {
          const neighborDimension = this.nodeDimensions.get(neighborId);
          if (neighborDimension !== nodeDimension) {
            continue;
          }
        }

        if (!visited.has(neighborId)) {
          // Optimized Set has check
          visited.add(neighborId); // Optimized Set add

          const dist = this._distance(id, neighborId);

          // Add to results if results is not full or if it's closer than furthest result (optimized result management)
          let furthestResultDist = -Infinity;
          if (results.size >= this.efConstruction) {
            // Only calculate if needed
            for (const dist of results.values()) {
              // Optimized Map value iteration
              if (dist > furthestResultDist) {
                furthestResultDist = dist;
              }
            }
          }

          if (results.size < this.efConstruction || dist < furthestResultDist) {
            if (results.size >= this.efConstruction) {
              // Optimized check, no need to find ID if not replacing
              // Find and remove furthest result (less frequent operation, optimized search for furthest)
              let furthestId: number | string | null = null;
              let maxDist = -Infinity;
              for (const [resultId, resultDist] of results.entries()) {
                // Optimized Map iteration
                if (resultDist > maxDist) {
                  maxDist = resultDist;
                  furthestId = resultId;
                }
              }
              if (furthestId !== null) {
                results.delete(furthestId); // Optimized Map delete
              }
            }
            results.set(neighborId, dist); // Optimized Map set
            candidates.set(neighborId, dist); // Optimized Map set
          }
        }
      }
    }

    // Keep only M closest neighbors (optimized sorting and mapping)
    return Array.from(results.entries())
      .sort((a, b) => a[1] - b[1]) // Sorting is still O(N log N), might be bottleneck for large ef
      .slice(0, this.M)
      .map(([id]) => id);
  }

  /**
   * Prune connections to maintain at most M connections per node
   * @private
   */
  private _pruneConnections(node: HNSWNode, level: number): void {
    const connections = node.connections.get(level); // Optimized retrieval
    if (!connections || connections.size <= this.M) return;

    // Get node's dimension (optimized dimension retrieval)
    const nodeDimension = this.nodeDimensions.get(node.id);

    // Calculate distances from this node to all its neighbors
    const distances: { id: number | string; dist: number }[] = [];

    // Optimized loop for distance calculation
    for (const neighborId of connections) {
      // Skip neighbors with different dimensions in dimension-aware mode (optimized dimension check)
      if (this.dimensionAware) {
        const neighborDimension = this.nodeDimensions.get(neighborId);
        if (neighborDimension !== nodeDimension) {
          continue;
        }
      }

      const dist = this._distance(node.id, neighborId);
      distances.push({ id: neighborId, dist });
    }

    // Sort by distance and keep only the M closest (optimized sorting and connection update)
    distances.sort((a, b) => a.dist - b.dist);

    // Create new connection set with only M closest (optimized Set creation and population)
    const newConnections = new Set<number | string>();
    for (let i = 0; i < Math.min(this.M, distances.length); i++) {
      newConnections.add(distances[i].id); // Optimized Set add
    }

    // Replace old connections with pruned set (optimized Map set)
    node.connections.set(level, newConnections);
  }

  /**
   * Calculate random level for a new node
   * @private
   */
  private _randomLevel(): number {
    // Exponential distribution with base 1/levelProbability (no changes needed, already efficient)
    let level = 0;
    while (Math.random() < this.levelProbability && level < this.maxLevel) {
      level++;
    }
    return level;
  }

  /**
   * Calculate distance between two nodes
   * @private
   */
  private _distance(id1: number | string, id2: number | string): number {
    const vec1 = this.db.getVector(id1); // Optimized vector retrieval
    const vec2 = this.db.getVector(id2); // Optimized vector retrieval

    if (!vec1 || !vec2) {
      throw new Error(`Vector not found: ${!vec1 ? id1 : id2}`);
    }

    return this.distanceFunc(vec1, vec2);
  }

  /**
   * Calculate distance from query to a node
   * @private
   */
  private _distanceToQuery(query: Vector, id: number | string): number {
    const vec = this.db.getVector(id); // Optimized vector retrieval

    if (!vec) {
      throw new Error(`Vector not found: ${id}`);
    }

    return this.distanceFunc(query, vec);
  }

  /**
   * Get HNSW statistics
   * @returns Stats object with graph information
   */
  getStats(): HNSWStats {
    const levels = Array.from(this.nodeToLevel.values()); // Optimized value retrieval
    const maxLevel = levels.length > 0 ? Math.max(...levels) : 0;

    // Count nodes per level
    const nodesPerLevel: number[] = new Array(maxLevel + 1).fill(0);
    for (const level of levels) {
      // Optimized loop
      nodesPerLevel[level]++;
    }

    // Calculate average connections per node per level
    const avgConnectionsPerLevel: number[] = [];
    for (let level = 0; level <= maxLevel; level++) {
      // Optimized loop
      let totalConnections = 0;
      let nodesWithLevel = 0;

      for (const node of this.nodes.values()) {
        // Optimized value iteration
        const connections = node.connections.get(level); // Optimized retrieval
        if (connections) {
          totalConnections += connections.size;
          nodesWithLevel++;
        }
      }

      avgConnectionsPerLevel.push(nodesWithLevel > 0 ? totalConnections / nodesWithLevel : 0);
    }

    // Count vectors by dimension
    const nodesByDimension: Record<number, number> = {};
    for (const dimension of this.nodeDimensions.values()) {
      // Optimized value iteration
      nodesByDimension[dimension] = (nodesByDimension[dimension] || 0) + 1;
    }

    return {
      totalNodes: this.nodes.size,
      maxM: this.M,
      efConstruction: this.efConstruction,
      efSearch: this.efSearch,
      levels: maxLevel + 1,
      nodesPerLevel,
      avgConnectionsPerLevel,
      entryPoint: this.entryPointId,
      dimensionAware: this.dimensionAware,
      dimensionGroups: this.dimensionGroups.size,
      dimensions: {
        counts: nodesByDimension,
        entryPoints: Object.fromEntries(this.dimensionEntryPoints),
      },
      deletedNodesCount: this.deletedNodes.size, // Add deleted nodes count to stats
    };
  }

  /**
   * Serialize HNSW graph to JSON
   * @returns JSON string representation of the graph
   */
  serialize(): string {
    const data = {
      M: this.M,
      efConstruction: this.efConstruction,
      efSearch: this.efSearch,
      maxLevel: this.maxLevel,
      levelProbability: this.levelProbability,
      entryPointId: this.entryPointId,
      dimensionAware: this.dimensionAware,
      dimensionEntryPoints: Array.from(this.dimensionEntryPoints.entries()) || [], // Optimized Map to Array conversion
      nodes:
        Array.from(this.nodes.entries()).map(([id, node]) => ({
          // Optimized Map iteration and mapping
          id,
          level: this.nodeToLevel.get(id), // Optimized retrieval
          dimension: this.nodeDimensions.get(id), // Optimized retrieval
          connections: Array.from(node.connections.entries()).map(
            // Optimized Map iteration and mapping
            ([level, connections]) => ({
              level,
              connections: Array.from(connections), // Optimized Set to Array conversion
            })
          ),
        })) || [],
    };

    return JSON.stringify(data);
  }

  /**
   * Deserialize HNSW graph from JSON
   * @param json - JSON string representation of the graph
   * @returns HNSW instance
   */
  static deserialize(json: string, db: ClusteredVectorDB): HNSW {
    // Parse JSON string (optimized error handling)
    const data = JSON.parse(json);

    const hnsw = new HNSW(db, {
      M: data.M,
      efConstruction: data.efConstruction,
      efSearch: data.efSearch,
      maxLevel: data.maxLevel,
      levelProbability: data.levelProbability,
      dimensionAware: data.dimensionAware !== false,
    });

    hnsw.entryPointId = data.entryPointId;

    // Restore dimension entry points (optimized Map set population)
    if (data.dimensionEntryPoints) {
      for (const [dimension, entryPoint] of data.dimensionEntryPoints) {
        hnsw.dimensionEntryPoints.set(Number(dimension), entryPoint);
      }
    }

    // Rebuild nodes and their connections (optimized loop and node creation)
    for (const nodeData of data.nodes) {
      const { id, level, dimension } = nodeData;

      // Create node
      hnsw._createNode(id, level, dimension);

      // Store dimension information
      if (dimension !== undefined) {
        hnsw.nodeDimensions.set(id, dimension);

        // Add to dimension group (optimized Set operations)
        let dimensionSet = hnsw.dimensionGroups.get(dimension);
        if (!dimensionSet) {
          dimensionSet = new Set();
          hnsw.dimensionGroups.set(dimension, dimensionSet);
        }
        dimensionSet.add(id);
      }

      // Add connections at each level (optimized loop and connection population)
      for (const connData of nodeData.connections) {
        // Use type assertion to inform TypeScript of the correct type
        const levelConnections = new Set<number | string>(connData.connections as (number | string)[]);
        const node = hnsw.nodes.get(id); // Optimized retrieval
        if (node) {
          node.connections.set(connData.level, levelConnections);
        }
      }
    }

    hnsw.initialized = true;

    return hnsw;
  }

  /**
   * Save HNSW index to disk
   * @param filePath - Path to save the index
   */
  async saveIndex(filePath: string): Promise<void> {
    const data = this.serialize();
    await fs.writeFile(filePath, data, 'utf8');
  }

  /**
   * Load HNSW index from disk
   * @param filePath - Path to load the index from
   * @param db - Vector database
   */
  static async loadIndex(filePath: string, db: ClusteredVectorDB, options: LoadIndexHNSWOptions = {}): Promise<HNSW> {
    const data = await fs.readFile(filePath, 'utf8');
    const hnsw = HNSW.deserialize(data, db);

    // Set dimension-aware mode from options if provided
    if (options.dimensionAware !== undefined) {
      hnsw.dimensionAware = options.dimensionAware;
    }

    return hnsw;
  }

  /**
   * Clean up resources
   */
  close(): void {
    // Clear internal data structures to free memory (optimized clear operations)
    this.nodes.clear();
    this.nodeToLevel.clear();
    this.nodeDimensions.clear();
    this.dimensionGroups.clear();
    this.dimensionEntryPoints.clear();
    this.entryPointId = null;
    this.initialized = false;
    this.deletedNodes.clear(); // Clear deleted nodes
  }
}

export default HNSW;
