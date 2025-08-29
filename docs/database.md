# Database Architecture

## Overview

NBase implements a sophisticated multi-layered database architecture designed for high-performance vector similarity search at scale. The architecture combines partitioning, clustering, indexing, and compression techniques to handle millions of high-dimensional vectors efficiently.

## Architecture Layers

### 1. Database Layer (High-Level Interface)

The `Database` class provides the primary user interface, offering:

- **Unified API**: Single entry point for all operations
- **Resource Management**: Automatic lifecycle management
- **Caching**: LRU cache for search results
- **Monitoring**: Performance metrics and system monitoring
- **Concurrency Control**: Managed concurrent operations
- **Event System**: Comprehensive event-driven architecture

```typescript
const db = new Database({
  vectorSize: 1536,
  cacheSize: 1000,
  maxConcurrentSearches: 4,
  // ... other options
});
```

### 2. Partitioning Layer (Horizontal Scaling)

The `PartitionedVectorDB` manages data distribution across multiple partitions:

- **Automatic Partitioning**: Creates new partitions when capacity is reached
- **LRU Cache**: Manages memory usage by unloading inactive partitions
- **Parallel Processing**: Distributes search queries across partitions
- **Load Balancing**: Evenly distributes vectors across partitions
- **Persistence**: Saves partition metadata and configurations

```typescript
// Partitioning automatically handles data distribution
await db.bulkAdd(vectors); // Vectors distributed across partitions
const results = await db.search(query); // Search across all partitions
```

### 3. Clustering Layer (Data Organization)

The `ClusteredVectorDB` organizes vectors within each partition:

- **K-means Clustering**: Groups similar vectors for faster search
- **Dynamic Clusters**: Creates new clusters as data grows
- **Memory Efficiency**: Reduces search space through clustering
- **Adaptive Thresholds**: Automatically adjusts clustering parameters

### 4. Storage Layer (Core Operations)

The `VectorDB` handles fundamental vector operations:

- **Vector Storage**: Efficient storage of high-dimensional vectors
- **Metadata Management**: Key-value metadata storage
- **Basic Search**: Exact nearest neighbor search
- **Memory Management**: Optimized memory usage

## Data Flow Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   REST API      │────│   Database       │────│ PartitionedDB   │
│   (Express)     │    │   (High-level)   │    │ (Scaling)       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   UnifiedSearch │────│  ClusteredDB     │────│   VectorDB      │
│   (Algorithms)  │    │   (Clustering)   │    │   (Storage)     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Search Data Flow

1. **Query Reception**: Query enters through Database or REST API
2. **Cache Check**: Check LRU cache for recent identical queries
3. **Partition Distribution**: Query distributed to relevant partitions
4. **Index Search**: Each partition uses appropriate index (HNSW/LSH)
5. **Cluster Filtering**: Clusters filter irrelevant vectors
6. **Result Aggregation**: Combine and rank results from all partitions
7. **Reranking**: Apply diversity or quality reranking if requested
8. **Response**: Return final results to client

## Storage Organization

### Directory Structure

```
database/
├── partitions/
│   ├── p-1749282892289/
│   │   ├── p-1749282892289.config.json
│   │   ├── data/
│   │   │   ├── cluster.json
│   │   │   ├── meta.json
│   │   │   └── vec.bin
│   │   └── hnsw/
│   │       └── hnsw_index.json
│   └── p-1749282892290/
│       └── ...
├── config.json
└── metadata.json
```

### File Formats

- **config.json**: Partition configuration and metadata
- **cluster.json**: K-means clustering data and centroids
- **meta.json**: Vector metadata storage
- **vec.bin**: Binary vector data storage
- **hnsw_index.json**: HNSW graph structure and parameters

## Indexing System

### Available Indexes

#### HNSW (Hierarchical Navigable Small World)
- **Use Case**: High-accuracy similarity search
- **Complexity**: O(log n) search time
- **Parameters**:
  - `M`: Maximum connections per node (default: 16)
  - `efConstruction`: Index build quality (default: 200)
  - `efSearch`: Search accuracy vs speed (default: 100)

#### LSH (Locality-Sensitive Hashing)
- **Use Case**: Ultra-fast approximate search
- **Complexity**: O(1) average case
- **Parameters**:
  - `numberOfHashes`: Hash functions (default: 10)
  - `numberOfBuckets`: Hash buckets (default: 100)

#### Flat Index
- **Use Case**: Exact search for small datasets
- **Complexity**: O(n) search time
- **Parameters**: None required

### Index Management

```typescript
// Build indexes
await db.buildIndexes({
  progressCallback: (progress) => console.log(`${progress}% complete`)
});

// Save/load indexes
await db.saveHNSWIndices();
await db.loadHNSWIndices();
```

## Clustering Strategy

### K-means Clustering

- **Purpose**: Group similar vectors to reduce search space
- **Algorithm**: Lloyd's algorithm with optimizations
- **Parameters**:
  - `clusterSize`: Target vectors per cluster (default: 1000)
  - `maxIterations`: Maximum K-means iterations (default: 50)
  - `distanceMetric`: Clustering distance function

### Dynamic Clustering

- **Auto-creation**: New clusters created when thresholds exceeded
- **Load balancing**: Evenly distribute vectors across clusters
- **Adaptive sizing**: Adjust cluster sizes based on data distribution

## Compression Techniques

### Product Quantization (PQ)

- **Purpose**: Reduce memory usage and improve search speed
- **Method**: Split vectors into subvectors, quantize each
- **Parameters**:
  - `subvectorSize`: Dimension of each subvector
  - `numClusters`: Number of centroids per subquantizer

### Usage

```typescript
const db = new Database({
  clustering: {
    useCompression: true,
    compression: {
      algorithm: 'pq',
      subvectorSize: 8,
      numClusters: 256
    }
  }
});
```

## Memory Management

### LRU Caching Strategy

#### Search Result Cache
- **Purpose**: Cache frequent search results
- **Implementation**: LRU cache with configurable size
- **Eviction**: Least recently used results evicted first

#### Partition Cache
- **Purpose**: Manage memory usage for large datasets
- **Implementation**: LRU cache for active partitions
- **Policy**: Unload least recently used partitions

### Memory Configuration

```typescript
const db = new Database({
  cacheSize: 1000,              // Search result cache
  partitioning: {
    maxActivePartitions: 3,     // Memory limit for partitions
    partitionCapacity: 50000    // Vectors per partition
  }
});
```

## Persistence and Recovery

### Automatic Persistence

- **Configuration Saving**: Partition configs saved automatically
- **Index Persistence**: Search indexes saved to disk
- **Metadata Storage**: Vector metadata persisted
- **Recovery**: Automatic recovery on restart

### Backup Strategy

```typescript
// Manual backup
await db.save();

// Automatic backup configuration
const db = new Database({
  persistence: {
    autoSave: true,
    saveIntervalMs: 300000  // 5 minutes
  }
});
```

## Performance Characteristics

### Scalability Metrics

| Dataset Size | Partitions | Memory Usage | Search Time |
|-------------|------------|--------------|-------------|
| < 10K | 1 | Low | < 10ms |
| 10K - 100K | 2-5 | Medium | 10-50ms |
| 100K - 1M | 5-20 | High | 50-200ms |
| > 1M | 20+ | Very High | 200ms+ |

### Optimization Strategies

#### For Speed
- Use LSH indexing
- Enable result caching
- Reduce `efSearch` parameter
- Use compression

#### For Accuracy
- Use HNSW indexing
- Increase `efSearch` parameter
- Disable compression
- Apply reranking

#### For Memory Efficiency
- Reduce partition capacity
- Enable compression
- Limit active partitions
- Use smaller cache sizes

## Monitoring and Observability

### Metrics Collection

```typescript
const db = new Database({
  monitoring: {
    enable: true,
    intervalMs: 5000,
    enableSystemMetrics: true,
    enableSearchMetrics: true,
    enableDatabaseMetrics: true
  }
});

// Get metrics
const stats = await db.getStats();
console.log('Performance metrics:', stats.performance);
```

### Available Metrics

- **System Metrics**: CPU, memory, disk usage
- **Search Metrics**: Query count, response times, cache hit rates
- **Database Metrics**: Vector count, partition status, index status
- **Performance Metrics**: Throughput, latency percentiles

## Configuration Best Practices

### Development Configuration

```typescript
const devConfig = {
  vectorSize: 1536,
  partitioning: {
    partitionCapacity: 10000,
    maxActivePartitions: 2
  },
  indexing: {
    buildOnStart: true,
    hnswOptions: { M: 8, efConstruction: 100 }
  },
  monitoring: { enable: true }
};
```

### Production Configuration

```typescript
const prodConfig = {
  vectorSize: 1536,
  partitioning: {
    partitionCapacity: 100000,
    maxActivePartitions: 5,
    autoCreatePartitions: true
  },
  indexing: {
    buildOnStart: true,
    autoSave: true,
    hnswOptions: { M: 16, efConstruction: 200, efSearch: 150 }
  },
  clustering: {
    clusterSize: 2000,
    useCompression: true
  },
  cacheSize: 5000,
  persistence: {
    autoSave: true,
    saveIntervalMs: 300000
  },
  monitoring: {
    enable: true,
    enableSystemMetrics: true,
    enableSearchMetrics: true
  }
};
```

## Troubleshooting

### Common Issues

#### High Memory Usage
- Reduce `maxActivePartitions`
- Enable compression
- Decrease cache size
- Monitor partition sizes

#### Slow Search Performance
- Check index status
- Adjust `efSearch` parameter
- Enable result caching
- Consider LSH for speed-critical applications

#### Index Building Issues
- Ensure sufficient memory
- Check disk space
- Monitor build progress
- Consider building indexes incrementally

### Performance Tuning

1. **Monitor System Resources**: Use built-in monitoring
2. **Profile Search Queries**: Identify slow queries
3. **Adjust Index Parameters**: Balance accuracy vs speed
4. **Optimize Configuration**: Tune based on workload
5. **Scale Horizontally**: Add more partitions as needed

This architecture provides a solid foundation for high-performance vector similarity search while maintaining flexibility and ease of use.
 