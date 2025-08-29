# NBase Vector Database Architecture

## Overview

NBase implements a sophisticated multi-layered vector database architecture designed for high-performance similarity search across massive datasets. The system combines in-memory operations, disk-based persistence, advanced indexing algorithms, and intelligent partitioning to deliver optimal performance for various use cases.

## Architecture Layers

### Layer 1: Core Vector Database (VectorDB)

The foundational layer providing basic vector storage and operations.

#### Features
- **In-Memory Storage**: Fast access using Float32Array buffers
- **Basic Metadata**: Key-value metadata storage per vector
- **Distance Metrics**: Multiple distance calculations (Cosine, Euclidean, Manhattan)
- **Simple Operations**: Add, search, update, delete vectors

#### Usage
```typescript
import { VectorDB } from '@n2flowjs/nbase';

const db = new VectorDB({
  vectorSize: 128,
  distanceMetric: 'cosine',
  enableMetadata: true
});

// Add vectors
await db.addVector('user-1', userEmbedding, {
  name: 'John Doe',
  age: 30,
  interests: ['AI', 'ML']
});

// Search
const results = await db.search(queryEmbedding, {
  k: 10,
  includeMetadata: true
});
```

#### Performance Characteristics
- **Memory Usage**: O(n × d) where n = vectors, d = dimensions
- **Search Time**: O(n) - linear scan
- **Insert Time**: O(1) amortized
- **Best For**: Small datasets (< 10K vectors), development, testing

### Layer 2: Clustered Vector Database (ClusteredVectorDB)

Enhanced implementation with K-means clustering for improved search efficiency.

#### Features
- **K-means Clustering**: Automatic vector grouping for faster search
- **Dynamic Clusters**: Adaptive cluster creation and management
- **Cluster-Based Search**: Search nearest clusters first, then refine
- **Memory Optimization**: Reduced memory footprint through clustering

#### Configuration
```typescript
const db = new ClusteredVectorDB({
  vectorSize: 128,
  clusterSize: 100,                    // Vectors per cluster
  maxClusters: 50,                     // Maximum clusters
  newClusterThresholdFactor: 1.5,      // Cluster split threshold
  distanceMetric: 'cosine',
  rebuildClustersInterval: 3600000     // Rebuild every hour
});
```

#### How Clustering Works

1. **Initial Clustering**: Vectors are grouped using K-means algorithm
2. **Search Optimization**: Query searches nearest clusters first
3. **Dynamic Updates**: Clusters adapt as new vectors are added
4. **Memory Efficiency**: Only active clusters kept in memory

#### Performance Characteristics
- **Memory Usage**: O(n × d + k × d) where k = clusters
- **Search Time**: O(k + n/k) - sub-linear with good clustering
- **Cluster Build Time**: O(n × k × d × iterations)
- **Best For**: Medium datasets (10K - 1M vectors), balanced performance

### Layer 3: Partitioned Vector Database (PartitionedVectorDB)

Production-grade implementation with intelligent partitioning and advanced indexing.

#### Features
- **Automatic Partitioning**: Intelligent data distribution across partitions
- **Disk-Based Storage**: Persistent storage with memory caching
- **HNSW Indexing**: Graph-based approximate nearest neighbor search
- **Memory Management**: LRU caching and partition unloading
- **Concurrent Operations**: Multi-threaded processing support

#### Configuration
```typescript
const db = new PartitionedVectorDB({
  partitionsDir: './data/partitions',
  partitionCapacity: 50000,           // Vectors per partition
  maxActivePartitions: 3,             // Memory limit
  vectorSize: 128,
  distanceMetric: 'cosine',

  // Indexing
  indexing: {
    enableHNSW: true,
    hnswOptions: {
      M: 16,                         // Connections per node
      efConstruction: 200,           // Build quality
      efSearch: 100                  // Search quality
    }
  },

  // Compression
  compression: {
    enabled: true,
    algorithm: 'product_quantization',
    compressionRatio: 0.5
  },

  // Caching
  cacheSize: 1000,
  enableMetadataCache: true
});
```

## Advanced Features

### Intelligent Partitioning

#### Automatic Partition Creation
```typescript
// Partitions created automatically when capacity reached
await db.addVector('vec-1', vector1);  // Creates partition-001
// ... add more vectors
await db.addVector('vec-50001', vector2);  // Creates partition-002
```

#### Partition Management
```typescript
// Monitor partition status
const stats = await db.getStats();
console.log('Active Partitions:', stats.partitions.active);
console.log('Total Partitions:', stats.partitions.total);

// Manual partition operations
await db.unloadPartition('partition-001');  // Free memory
await db.loadPartition('partition-002');    // Load to memory
```

#### Partition Distribution Strategies
- **Round Robin**: Even distribution across partitions
- **Hash-based**: Consistent partitioning by vector ID
- **Range-based**: Partitioning by vector properties
- **Custom**: User-defined partitioning logic

### Advanced Indexing

#### HNSW Index Management
```typescript
// Build indexes for specific partitions
await db.buildIndexes({
  partitions: ['partition-001', 'partition-002'],
  indexType: 'hnsw',
  options: {
    M: 32,
    efConstruction: 300
  }
});

// Index statistics
const indexStats = await db.getIndexStats('partition-001');
console.log('HNSW Nodes:', indexStats.nodeCount);
console.log('Memory Usage:', indexStats.memoryUsage);
```

#### Index Persistence
```typescript
// Save indexes to disk
await db.saveIndexes();

// Load indexes on startup
await db.loadIndexes();

// Selective loading
await db.loadIndex('partition-001', 'hnsw');
```

### Compression Techniques

#### Product Quantization
```typescript
const db = new PartitionedVectorDB({
  compression: {
    enabled: true,
    algorithm: 'product_quantization',
    numSubquantizers: 8,
    numCentroids: 256,
    compressionRatio: 0.5
  }
});
```

#### K-means Compression
```typescript
const db = new PartitionedVectorDB({
  compression: {
    enabled: true,
    algorithm: 'kmeans',
    numClusters: 1000,
    maxIterations: 50
  }
});
```

### Memory Management

#### LRU Caching
```typescript
// Configure cache
const db = new PartitionedVectorDB({
  cacheSize: 5000,              // Max cached vectors
  cacheTTL: 3600000,            // 1 hour TTL
  enableMetadataCache: true,
  enableVectorCache: true
});

// Cache statistics
const cacheStats = await db.getCacheStats();
console.log('Cache Hit Rate:', cacheStats.hitRate);
console.log('Cache Size:', cacheStats.size);
```

#### Memory Monitoring
```typescript
// Real-time memory monitoring
db.on('memory:warning', (stats) => {
  console.log('Memory usage high:', stats.usage);
});

db.on('partition:unloaded', (partitionId) => {
  console.log('Partition unloaded:', partitionId);
});

// Manual memory management
await db.forceGC();  // Force garbage collection
await db.unloadInactivePartitions();
```

## Performance Comparison

| Feature | VectorDB | ClusteredVectorDB | PartitionedVectorDB |
|---------|----------|-------------------|-------------------|
| **Storage** | Memory-only | Memory-only | Disk + Memory |
| **Scalability** | < 10K vectors | 10K - 1M vectors | 1M+ vectors |
| **Indexing** | None | K-means clusters | HNSW + Clusters |
| **Persistence** | Manual | Manual | Automatic |
| **Memory Usage** | High | Medium | Controlled |
| **Search Speed** | Slow | Medium | Fast |
| **Setup Complexity** | Low | Medium | High |
| **Production Ready** | No | Partial | Yes |

## Performance Optimization

### Search Optimization

#### Algorithm Selection Guide
```typescript
function chooseSearchAlgorithm(datasetSize: number, accuracy: number) {
  if (datasetSize < 10000) {
    return { algorithm: 'flat', accuracy: '100%' };
  }

  if (datasetSize < 1000000) {
    return { algorithm: 'clustered', accuracy: '90-95%' };
  }

  if (accuracy > 0.95) {
    return { algorithm: 'partitioned-hnsw', accuracy: '95-99%' };
  }

  return { algorithm: 'partitioned-lsh', accuracy: '80-95%' };
}
```

#### Query Optimization
```typescript
// Optimized search configuration
const results = await db.search(queryVector, {
  k: 20,
  algorithm: 'hnsw',
  efSearch: 200,                    // Higher = more accurate
  partitions: ['partition-001'],    // Search specific partitions
  filter: (id, metadata) => metadata.category === 'relevant',
  includeMetadata: true,
  includeVectors: false,            // Save memory
  timeout: 5000                     // Prevent long searches
});
```

### Memory Optimization

#### Partition Sizing
```typescript
// Optimal partition sizing based on available memory
function calculatePartitionSize(availableMemoryMB: number, vectorSize: number) {
  const vectorMemoryMB = (vectorSize * 4) / (1024 * 1024); // Float32 = 4 bytes
  const overheadFactor = 1.5; // Account for metadata and indexes
  const maxVectorsPerPartition = Math.floor(
    (availableMemoryMB / vectorMemoryMB) * overheadFactor
  );

  return Math.min(maxVectorsPerPartition, 100000); // Cap at 100K
}
```

#### Compression Tuning
```typescript
// Balance compression vs accuracy
const compressionConfigs = {
  highAccuracy: {
    algorithm: 'product_quantization',
    compressionRatio: 0.8,
    numSubquantizers: 16
  },
  balanced: {
    algorithm: 'product_quantization',
    compressionRatio: 0.6,
    numSubquantizers: 8
  },
  highCompression: {
    algorithm: 'kmeans',
    compressionRatio: 0.3,
    numClusters: 500
  }
};
```

## Monitoring and Observability

### Database Statistics
```typescript
const stats = await db.getStats();

console.log('Database Overview:');
console.log(`- Total Vectors: ${stats.vectors.total}`);
console.log(`- Active Partitions: ${stats.partitions.active}`);
console.log(`- Memory Usage: ${stats.memory.usage}`);
console.log(`- Cache Hit Rate: ${stats.cache.hitRate}%`);
console.log(`- Average Search Time: ${stats.performance.avgSearchTime}ms`);
```

### Performance Metrics
```typescript
// Enable detailed monitoring
const db = new PartitionedVectorDB({
  monitoring: {
    enable: true,
    metricsInterval: 5000,
    trackSearchMetrics: true,
    trackMemoryMetrics: true
  }
});

// Monitor events
db.on('search:completed', (event) => {
  console.log(`Search took ${event.duration}ms, found ${event.resultCount} results`);
});

db.on('partition:loaded', (event) => {
  console.log(`Partition ${event.partitionId} loaded in ${event.duration}ms`);
});
```

## Best Practices

### Data Organization

#### Partition Strategy
```typescript
// Group related data in partitions
const partitionStrategies = {
  temporal: (vector, metadata) => {
    const date = new Date(metadata.timestamp);
    return `year-${date.getFullYear()}`;
  },

  categorical: (vector, metadata) => {
    return `category-${metadata.category}`;
  },

  hash: (vector, id) => {
    const hash = simpleHash(id);
    return `partition-${hash % 10}`;
  }
};
```

#### Metadata Optimization
```typescript
// Efficient metadata structure
const optimizedMetadata = {
  // Use primitive types
  id: 'user-123',
  category: 'premium',
  score: 0.95,
  tags: ['verified', 'active'],

  // Avoid large objects
  // profile: { ... } // Don't store large nested objects

  // Use references instead
  profileId: 'profile-456'
};
```

### Maintenance Operations

#### Regular Maintenance
```typescript
// Weekly maintenance routine
async function weeklyMaintenance(db: PartitionedVectorDB) {
  // Rebuild indexes for better performance
  await db.rebuildIndexes({
    partitions: 'all',
    optimizeFor: 'search-speed'
  });

  // Clean up old partitions
  await db.cleanupOldPartitions(30); // 30 days

  // Optimize compression
  await db.recompressPartitions({
    targetRatio: 0.6,
    algorithm: 'product_quantization'
  });

  // Update statistics
  await db.updateStats();
}
```

#### Backup and Recovery
```typescript
// Comprehensive backup
async function createBackup(db: PartitionedVectorDB, backupPath: string) {
  // Create backup directory
  await fs.mkdir(backupPath, { recursive: true });

  // Backup configuration
  const config = await db.exportConfig();
  await fs.writeFile(`${backupPath}/config.json`, JSON.stringify(config));

  // Backup partitions
  await db.backupPartitions(backupPath, {
    includeIndexes: true,
    compress: true,
    parallel: 4
  });

  // Verify backup
  const isValid = await db.verifyBackup(backupPath);
  return isValid;
}
```

## Migration Guide

### Migrating from VectorDB to ClusteredVectorDB
```typescript
// Export from VectorDB
const vectorData = await oldDB.exportAll();

// Create new ClusteredVectorDB
const newDB = new ClusteredVectorDB({
  vectorSize: 128,
  clusterSize: 100
});

// Import data
await newDB.importBulk(vectorData);
```

### Migrating to PartitionedVectorDB
```typescript
// For large-scale migration
const migration = new DatabaseMigration({
  sourceDB: oldDB,
  targetDB: newDB,
  batchSize: 10000,
  parallelWorkers: 4
});

await migration.run({
  onProgress: (progress) => console.log(`${progress}% complete`),
  validateAfterMigration: true
});
```

This comprehensive architecture provides the foundation for building high-performance, scalable vector search applications with NBase.
