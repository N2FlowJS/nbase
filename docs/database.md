# Database Reference

## Overview

The `Database` class is the main interface for NBase, providing high-level access to vector storage, search, and management capabilities. It wraps the underlying `PartitionedVectorDB` and provides additional features like caching, monitoring, and automatic resource management.

## Constructor

```typescript
const db = new Database(options: DatabaseOptions);
```

### Options

```typescript
interface DatabaseOptions {
  // Vector configuration
  vectorSize?: number;              // Default: 1536
  
  // Caching and performance
  cacheSize?: number;              // Default: 1000
  maxConcurrentSearches?: number;  // Default: CPU cores - 1
  
  // Clustering configuration
  clustering?: {
    clusterSize: number;           // Default: 100
    newClusterThresholdFactor: number;  // Default: 1.5
    useCompression: boolean;       // Default: true
  };
  
  // Partitioning configuration
  partitioning?: {
    partitionsDir: string;         // Default: './database/partitions'
    partitionCapacity: number;     // Default: 100000
    autoLoadPartitions: boolean;   // Default: true
    maxActivePartitions: number;   // Default: 3
  };
  
  // Indexing configuration  
  indexing?: {
    buildOnStart: boolean;         // Default: true
    autoRebuildThreshold: number;  // Default: 500
    hnsw: {
      M: number;                   // Default: 16
      efConstruction: number;      // Default: 200 
      efSearch: number;           // Default: 100
    }
  };
  
  // Persistence options
  persistence?: {
    dbPath?: string;              // Default: './database'
    saveIntervalMs?: number;      // Default: 300000 (5 minutes)
  };
  
  // Monitoring options
  monitoring?: {
    enable: boolean;              // Default: false
    intervalMs: number;           // Default: 60000
    logToConsole: boolean;        // Default: false
  };
}
```

## Core Methods

### Vector Operations

#### Adding Vectors

```typescript
// Add single vector
const result = await db.addVector(
  id: string | number | undefined,  // Optional ID
  vector: number[] | Float32Array,  // Vector data
  metadata?: Record<string, any>    // Optional metadata
): Promise<{
  partitionId: string;
  vectorId: string | number;
}>;

// Bulk add vectors
const result = await db.bulkAdd(
  vectors: Array<{
    id?: string | number;
    vector: number[] | Float32Array;
    metadata?: Record<string, any>;
  }>
): Promise<{
  count: number;
  partitionIds: string[];
}>;
```

#### Searching

```typescript
const results = await db.search(
  query: number[] | Float32Array,
  options?: {
    k?: number;                    // Number of results (default: 10)
    filter?: (id: string | number, metadata?: Record<string, any>) => boolean;
    includeMetadata?: boolean;     // Include metadata in results
    useHNSW?: boolean;            // Use HNSW index
    efSearch?: number;            // HNSW search parameter
    distanceMetric?: 'cosine' | 'euclidean';
    partitionIds?: string[];      // Specific partitions to search
    skipCache?: boolean;          // Bypass result caching
  }
): Promise<Array<{
  id: string | number;
  score: number;
  distance: number;
  metadata?: Record<string, any>;
}>>;
```

### Metadata Management

```typescript
// Add/update metadata
await db.addMetadata(id, metadata);
await db.updateMetadata(id, metadata);

// Get metadata
const metadata = await db.getMetadata(id);

// Search by metadata
const results = await db.getMetadataWithField(
  criteria: string | string[] | Record<string, any>,
  values?: any | any[],
  options?: { limit: number }
);
```

## Advanced Features

```typescript
// Extract relationships between vectors
const relationships = await db.extractRelationships(
  threshold: number,
  options?: {
    metric?: 'cosine' | 'euclidean';
    partitionIds?: string[];
    includeMetadata?: boolean;
  }
);

// Find vector communities
const communities = await db.extractCommunities(
  threshold: number,
  options?: {
    metric?: 'cosine' | 'euclidean';
    partitionIds?: string[];
    includeMetadata?: boolean;
  }
);
```

## Database Management

### Initialization and Status

```typescript
// Check if database is ready
const isReady = db.IsReady();

// Wait for initialization
await db.initializationPromise;
```

### Index Management

```typescript
// Build indices
await db.buildIndexes(
  partitionId?: string,
  options?: {
    force?: boolean;
    dimensionAware?: boolean;
    progressCallback?: (progress: number) => void;
  }
);
```

### State Management

```typescript
// Save current state
await db.save();

// Close database
await db.close();

// Get statistics
const stats = await db.getStats();
```

## Events

The Database class emits various events that can be listened to:

```typescript
// Lifecycle events
db.on('initializing', () => {});
db.on('ready', () => {});
db.on('close', () => {});

// Operation events
db.on('vector:add', (data) => {});
db.on('vectors:bulkAdd', (data) => {});
db.on('search:complete', (data) => {});
db.on('search:error', (data) => {});

// Background task events
db.on('save:complete', (data) => {});
db.on('index:progress', (data) => {});

// Error events
db.on('error', (data) => {});
db.on('warn', (data) => {});
```

## Performance Monitoring

```typescript
const stats = await db.getStats();
console.log(stats);
/*
{
  state: { isReady, isClosed, status },
  database: { vectors, partitions, indices },
  search: { calls, avgTime, methodCounts },
  searchCache: { size, hits, misses, hitRate },
  performance: { queries, avgSearchTimeMs },
  system: { cpuUsage, memoryUsage },
  options: { current configuration }
}
*/
```

## Best Practices

1. **Initialization**
   - Always wait for database initialization before performing operations
   - Use `await db.initializationPromise` or listen for 'ready' event

2. **Vector Management**
   - Use bulkAdd for adding multiple vectors
   - Keep vector dimensions consistent within partitions
   - Include relevant metadata for better filtering

3. **Search Optimization**
   - Enable HNSW indexing for large datasets
   - Use specific partitionIds when possible
   - Implement efficient filter functions
   - Utilize metadata search for non-vector queries

4. **Resource Management**
   - Monitor memory usage via getStats()
   - Close database properly when done
   - Configure appropriate partition sizes
   - Adjust maxActivePartitions based on memory
