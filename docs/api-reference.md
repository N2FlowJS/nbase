# NBase API Reference

## Overview

NBase provides a comprehensive API for vector database operations, including vector storage, similarity search, indexing, and clustering. The API is organized around the main `Database` class which provides a high-level interface to all functionality.

## Database Class

The `Database` class is the primary interface for interacting with NBase. It provides methods for vector operations, search, indexing, and database management.

### Constructor

```typescript
const db = new Database(options: DatabaseOptions)
```

Creates a new Database instance with the specified configuration options.

#### DatabaseOptions

```typescript
interface DatabaseOptions {
  // Vector configuration
  vectorSize?: number;                    // Default vector dimension
  cacheSize?: number;                     // Search result cache size
  maxConcurrentSearches?: number;         // Max concurrent search operations

  // Core subsystems
  indexing: IndexingConfiguration;        // Index configuration
  partitioning: PartitioningConfiguration; // Partition configuration
  clustering: ClusteringConfiguration;    // Clustering configuration
  persistence: PersistenceOptions;        // Persistence configuration

  // Optional features
  monitoring?: MonitoringConfiguration;   // Performance monitoring
  backup?: DatabaseBackUp;               // Backup configuration
}
```

### Core Methods

#### Vector Operations

##### addVector
```typescript
async addVector(
  id: string | number,
  vector: number[] | Float32Array,
  metadata?: Record<string, any>
): Promise<void>
```

Adds a single vector to the database.

**Parameters:**
- `id`: Unique identifier for the vector
- `vector`: Vector data as array or Float32Array
- `metadata`: Optional metadata object

**Example:**
```typescript
await db.addVector('doc1', [0.1, 0.2, 0.3], { title: 'Document 1' });
```

##### bulkAdd
```typescript
async bulkAdd(
  vectors: VectorData[]
): Promise<{ count: number; partitionIds: string[] }>
```

Adds multiple vectors to the database in a single operation.

**Parameters:**
- `vectors`: Array of vector data objects

**Returns:**
- `count`: Number of vectors added
- `partitionIds`: IDs of partitions where vectors were added

**Example:**
```typescript
const vectors = [
  { id: 'doc1', vector: [0.1, 0.2, 0.3], metadata: { category: 'A' } },
  { id: 'doc2', vector: [0.4, 0.5, 0.6], metadata: { category: 'B' } }
];
const result = await db.bulkAdd(vectors);
console.log(`Added ${result.count} vectors`);
```

##### deleteVector
```typescript
async deleteVector(id: string | number): Promise<boolean>
```

Deletes a vector from the database.

**Parameters:**
- `id`: ID of the vector to delete

**Returns:** `true` if vector was deleted, `false` if not found

##### getVector
```typescript
async getVector(id: string | number): Promise<{ vector: Float32Array; metadata?: Record<string, any> } | null>
```

Retrieves a vector and its metadata from the database.

**Parameters:**
- `id`: ID of the vector to retrieve

**Returns:** Vector data with metadata, or `null` if not found

##### updateMetadata
```typescript
async updateMetadata(
  id: string | number,
  metadata: Record<string, any>
): Promise<void>
```

Updates metadata for an existing vector.

**Parameters:**
- `id`: ID of the vector
- `metadata`: New metadata object (replaces existing)

#### Search Operations

##### search / findNearest
```typescript
async search(
  query: number[] | Float32Array,
  options?: UnifiedSearchOptions
): Promise<SearchResult[]>
```

Performs similarity search to find nearest neighbors.

**Parameters:**
- `query`: Query vector
- `options`: Search configuration options

**Returns:** Array of search results with IDs, distances, and metadata

**Example:**
```typescript
const results = await db.search([0.1, 0.2, 0.3], {
  k: 10,
  includeMetadata: true,
  distanceMetric: 'cosine',
  useHNSW: true
});
```

##### batchSearch
```typescript
async batchSearch(
  queries: BatchQuery[],
  options?: BatchSearchOptions
): Promise<SearchResult[][]>
```

Performs multiple search queries in parallel.

**Parameters:**
- `queries`: Array of search queries
- `options`: Batch search configuration

**Example:**
```typescript
const queries = [
  { query: [0.1, 0.2, 0.3], k: 5 },
  { query: [0.4, 0.5, 0.6], k: 3 }
];
const results = await db.batchSearch(queries);
```

#### Metadata Operations

##### getMetadata
```typescript
async getMetadata(id: string | number): Promise<Record<string, any> | null>
```

Retrieves metadata for a vector.

**Parameters:**
- `id`: Vector ID

**Returns:** Metadata object or `null` if not found

##### getMetadataWithFieldAcrossPartitions
```typescript
async getMetadataWithFieldAcrossPartitions(
  criteria: string | string[] | Record<string, any>,
  values?: any | any[],
  option?: { limit: number }
): Promise<Array<{ partitionId: string; vectorId: string | number; metadata: Record<string, any> }>>
```

Searches for vectors by metadata criteria across all partitions.

**Parameters:**
- `criteria`: Metadata field name(s) or criteria object
- `values`: Values to match (when criteria is string/array)
- `option`: Search options including result limit

#### Index Management

##### buildIndexes
```typescript
async buildIndexes(options?: BuildIndexHNSWOptions): Promise<void>
```

Builds search indexes for improved query performance.

**Parameters:**
- `options`: Index building configuration

**Example:**
```typescript
await db.buildIndexes({
  progressCallback: (progress) => console.log(`Progress: ${progress}%`)
});
```

##### saveHNSWIndices
```typescript
async saveHNSWIndices(partitionId?: string): Promise<void>
```

Saves HNSW indexes to disk.

**Parameters:**
- `partitionId`: Specific partition to save (optional, saves all if not specified)

##### loadHNSWIndices
```typescript
async loadHNSWIndices(partitionId?: string): Promise<void>
```

Loads HNSW indexes from disk.

**Parameters:**
- `partitionId`: Specific partition to load (optional, loads all if not specified)

#### Database Management

##### save
```typescript
async save(): Promise<void>
```

Saves the database state to disk.

##### close
```typescript
async close(): Promise<void>
```

Closes the database and releases all resources.

##### getStats
```typescript
async getStats(): Promise<DatabaseStats>
```

Retrieves comprehensive database statistics.

**Returns:** Detailed statistics about database state, performance, and configuration

#### Advanced Operations

##### extractRelationships
```typescript
async extractRelationships(
  threshold: number,
  options?: {
    metric?: DistanceMetric;
    partitionIds?: string[];
    includeMetadata?: boolean;
  }
): Promise<Array<{
  vector1: { id: string | number; partitionId: string; metadata?: Record<string, any> };
  vector2: { id: string | number; partitionId: string; metadata?: Record<string, any> };
  distance: number;
}>>
```

Finds relationships between vectors based on distance threshold.

**Parameters:**
- `threshold`: Maximum distance for relationship
- `options`: Configuration for relationship extraction

##### extractCommunities
```typescript
async extractCommunities(
  threshold: number,
  options?: {
    metric?: DistanceMetric;
    partitionIds?: string[];
    includeMetadata?: boolean;
  }
): Promise<Array<Array<{
  id: string | number;
  partitionId: string;
  metadata?: Record<string, any>;
}>>>
```

Extracts communities (clusters) of closely related vectors.

**Parameters:**
- `threshold`: Distance threshold for community detection
- `options`: Configuration options

## Search Options

### UnifiedSearchOptions

```typescript
interface UnifiedSearchOptions {
  // Result control
  k?: number;                          // Number of results to return
  includeMetadata?: boolean;           // Include metadata in results
  includeVectors?: boolean;            // Include vector data in results

  // Search behavior
  distanceMetric?: DistanceMetric;     // 'euclidean' | 'cosine'
  filter?: (id: string | number, metadata?: Record<string, any>) => boolean;

  // Algorithm selection
  useHNSW?: boolean;                   // Use HNSW index
  efSearch?: number;                   // HNSW search parameter

  // Advanced features
  rerank?: boolean;                    // Apply reranking
  rerankingMethod?: RerankingMethod;   // 'diversity' | 'standard' | 'weighted'
  rerankLambda?: number;               // Reranking parameter

  // Partition control
  partitionIds?: string[];             // Limit search to specific partitions

  // Performance
  skipCache?: boolean;                 // Skip result cache
  searchTimeoutMs?: number;            // Search timeout
}
```

### BatchSearchOptions

```typescript
interface BatchSearchOptions {
  maxBatchSize?: number;               // Maximum queries per batch
  prioritizeOrder?: boolean;           // Maintain result order
  groupSimilarQueries?: boolean;       // Optimize similar queries
  defaultSearchTimeout?: number;       // Timeout per query
  maxWorkers?: number;                 // Maximum worker threads
  useWorkers?: boolean;                // Enable worker threads
}
```

## Configuration Types

### IndexingConfiguration

```typescript
interface IndexingConfiguration {
  indexPath?: string;                  // Directory for index files
  buildOnStart?: boolean;              // Build indexes on startup
  autoSave?: boolean;                  // Auto-save indexes
  autoLoad?: boolean;                  // Auto-load indexes on startup
  runKMeansOnLoad?: boolean;           // Run K-means after loading

  // Index-specific options
  hnsw?: HNSWIndexConfiguration;
  lsh?: LSHIndexConfiguration;
  pq?: PQIndexConfiguration;
}
```

### PartitioningConfiguration

```typescript
interface PartitioningConfiguration {
  partitionsDir?: string;              // Directory for partitions
  partitionCapacity?: number;          // Max vectors per partition
  autoCreatePartitions?: boolean;      // Auto-create new partitions
  autoLoadPartitions?: boolean;        // Load partitions on startup
  maxActivePartitions?: number;        // Max partitions in memory
  defaultVectorSize?: number;          // Default vector dimension
}
```

### ClusteringConfiguration

```typescript
interface ClusteringConfiguration {
  clusterSize?: number;                // Target cluster size
  distanceMetric?: DistanceMetric;     // Clustering distance metric
  useCompression?: boolean;            // Enable vector compression
  kmeansMaxIterations?: number;        // K-means iterations
  newClusterThresholdFactor?: number;  // Cluster creation threshold
}
```

### PersistenceOptions

```typescript
interface PersistenceOptions {
  dbPath?: string;                     // Database storage path
  autoSave?: boolean;                  // Auto-save database
  saveIntervalMs?: number;             // Save interval in milliseconds
}
```

### MonitoringConfiguration

```typescript
interface MonitoringConfiguration {
  enable?: boolean;                    // Enable monitoring
  intervalMs?: number;                 // Metrics collection interval
  logToConsole?: boolean;              // Log to console
  enableSystemMetrics?: boolean;       // System resource metrics
  enableSearchMetrics?: boolean;       // Search performance metrics
  enableDatabaseMetrics?: boolean;     // Database state metrics
}
```

## Result Types

### SearchResult

```typescript
interface SearchResult {
  id: string | number;                 // Vector ID
  dist: number;                        // Distance to query
  metadata?: Record<string, any>;      // Vector metadata
  vector?: number[];                   // Vector data (if requested)
}
```

### DatabaseStats

```typescript
interface DatabaseStats {
  database: PartitionedDBStats;         // Database-level statistics
  search: UnifiedSearchPartitionedStats; // Search statistics
  searchCache: {                       // Cache statistics
    size: number;
    capacity: number;
    hits: number;
    misses: number;
    hitRate: number;
  };
  performance: {                       // Performance metrics
    queries: number;
    avgSearchTimeMs: number;
    cacheHitRate: number;
    concurrentSearches: number;
  };
  system?: ISystem;                    // System information
  memoryUsage: NodeJS.MemoryUsage;     // Memory usage
  state: {                             // Database state
    isReady: boolean;
    isClosed: boolean;
    status: string;
  };
  options: DatabaseOptions;            // Current configuration
}
```

## Events

The Database class emits various events for monitoring and integration:

### Database Events

- `ready`: Database initialization completed
- `close`: Database is closing
- `error`: Error occurred during operation
- `search:start`: Search operation started
- `search:complete`: Search operation completed
- `search:error`: Search operation failed
- `index:progress`: Index building progress
- `index:complete`: Index building completed
- `index:error`: Index building failed
- `save:complete`: Database save completed
- `partition:created`: New partition created
- `partition:loaded`: Partition loaded
- `partition:unloaded`: Partition unloaded

### Event Usage

```typescript
db.on('ready', () => {
  console.log('Database is ready!');
});

db.on('search:complete', (data) => {
  console.log(`Search completed in ${data.totalTime}ms`);
});

db.on('error', (error) => {
  console.error('Database error:', error);
});
```

## REST API

NBase provides a REST API server for HTTP-based access to all database operations.

### Server Configuration

```typescript
const { createServer } = require('@n2flowjs/nbase');

const server = createServer({
  port: 1307,
  host: 'localhost',
  rateLimit: {
    enable: true,
    maxRequestsPerMinute: 1000
  },
  database: {
    vectorSize: 1536,
    partitioning: { partitionsDir: './data/partitions' }
  }
});
```

### API Endpoints

#### Vectors
- `POST /vectors` - Add a vector
- `GET /vectors/:id` - Get a vector
- `DELETE /vectors/:id` - Delete a vector
- `PUT /vectors/:id/metadata` - Update metadata

#### Search
- `POST /search` - Similarity search
- `POST /search/batch` - Batch search
- `POST /search/metadata` - Search with metadata filtering
- `POST /search/relationships` - Find vector relationships
- `POST /search/communities` - Find vector communities

#### Database Management
- `GET /health` - Health check
- `GET /stats` - Database statistics
- `POST /indexes/build` - Build search indexes
- `POST /save` - Save database state

#### Example API Usage

```bash
# Add a vector
curl -X POST http://localhost:1307/vectors \
  -H "Content-Type: application/json" \
  -d '{
    "id": "doc1",
    "vector": [0.1, 0.2, 0.3],
    "metadata": { "title": "Example Document" }
  }'

# Search for similar vectors
curl -X POST http://localhost:1307/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": [0.1, 0.2, 0.3],
    "k": 5,
    "includeMetadata": true
  }'
```

## Error Handling

NBase provides comprehensive error handling with detailed error messages:

```typescript
try {
  await db.addVector('doc1', [0.1, 0.2, 0.3]);
} catch (error) {
  if (error.code === 'VECTOR_EXISTS') {
    console.log('Vector already exists');
  } else if (error.code === 'INVALID_DIMENSION') {
    console.log('Vector dimension mismatch');
  } else {
    console.error('Unexpected error:', error);
  }
}
```

## Best Practices

### Performance Optimization

1. **Choose appropriate indexes**: Use HNSW for high-accuracy searches, LSH for speed
2. **Configure partitions**: Set partition capacity based on your data size
3. **Enable caching**: Use search result caching for repeated queries
4. **Monitor performance**: Enable monitoring to track system performance
5. **Batch operations**: Use bulk operations for multiple vectors

### Memory Management

1. **Limit active partitions**: Configure `maxActivePartitions` to control memory usage
2. **Enable compression**: Use vector compression for large datasets
3. **Regular cleanup**: Save and close database when not in use
4. **Monitor memory usage**: Track memory usage through monitoring system

### Search Optimization

1. **Tune HNSW parameters**: Adjust `efSearch` based on accuracy vs speed requirements
2. **Use appropriate distance metrics**: Choose cosine for text embeddings, euclidean for spatial data
3. **Filter early**: Use metadata filters to reduce search space
4. **Batch searches**: Process multiple queries together for better performance

This API reference covers the core functionality of NBase. For more detailed examples and advanced usage patterns, refer to the examples directory and test files in the repository.
 