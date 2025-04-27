# Vector Databases

## Overview

Vector databases in NBase are designed to efficiently store and search high-dimensional vectors. The system supports multiple vector database implementations optimized for different use cases.

## Available Implementations

### 1. VectorDB

Basic vector database implementation with core functionality.

```typescript
import { VectorDB } from '@n2flowjs/nbase';

const db = new VectorDB({
  vectorSize: 1536,
  useCompression: true
});
```

Key features:
- In-memory vector storage
- Basic metadata support
- Simple vector operations

### 2. ClusteredVectorDB

Enhanced implementation with clustering capabilities.

```typescript
import { ClusteredVectorDB } from '@n2flowjs/nbase';

const db = new ClusteredVectorDB({
  vectorSize: 1536,
  clusterSize: 100,
  newClusterThresholdFactor: 1.5
});
```

Features:
- K-means clustering
- Optimized cluster-based search
- Dynamic cluster management

### 3. PartitionedVectorDB 

Production-grade implementation with advanced features.

```typescript
import { PartitionedVectorDB } from '@n2flowjs/nbase';

const db = new PartitionedVectorDB({
  partitionsDir: './partitions',
  partitionCapacity: 100000,
  maxActivePartitions: 3
});
```

Features:
- Disk-based storage
- Automatic partitioning
- Memory management
- HNSW indexing support

## Comparison

| Feature | VectorDB | ClusteredVectorDB | PartitionedVectorDB |
|---------|----------|-------------------|-------------------|
| Storage | Memory | Memory | Disk + Memory |
| Scale | Small | Medium | Large |
| Indexing | Basic | Clusters | HNSW + Clusters |
| Persistence | Manual | Manual | Automatic |
| Memory Usage | High | Medium | Controlled |
| Query Speed | Fast | Medium | Optimized |
| Best For | Testing | Medium Data | Production |

## Performance Characteristics

### Memory Usage

```plaintext
VectorDB: O(n * d)
ClusteredVectorDB: O(n * d + k * d)
PartitionedVectorDB: O(p * m * d)

Where:
n = number of vectors
d = vector dimensions
k = number of clusters
p = active partitions
m = partition capacity
```

### Search Complexity

| Implementation | Brute Force | With Index |
|----------------|-------------|------------|
| VectorDB | O(n) | O(n) |
| ClusteredVectorDB | O(k + n/k) | O(k + log(n/k)) |
| PartitionedVectorDB | O(p * m) | O(p * log m) |

## Implementation Details

### VectorDB

Basic vector storage using Float32Arrays:

```typescript
class VectorDB {
  private vectors: Map<IDType, Float32Array>;
  private metadata: Map<IDType, Record<string, any>>;
  
  findNearest(query: Vector, k: number): SearchResult[] {
    // Direct distance calculation
    return [...this.vectors.entries()]
      .map(([id, vec]) => ({
        id,
        distance: calculateDistance(query, vec)
      }))
      .sort((a, b) => a.distance - b.distance)
      .slice(0, k);
  }
}
```

### ClusteredVectorDB

Clustering-based implementation:

```typescript
class ClusteredVectorDB extends VectorDB {
  private clusters: Map<number, Cluster>;
  
  findNearest(query: Vector, k: number): SearchResult[] {
    // Find nearest clusters first
    const nearestClusters = this.findNearestClusters(query);
    
    // Search within closest clusters
    return this.searchInClusters(query, nearestClusters, k);
  }
}
```

### PartitionedVectorDB

Production implementation with partitioning:

```typescript
class PartitionedVectorDB {
  private partitions: Map<string, VectorPartition>;
  private activePartitions: Set<string>;
  
  async findNearest(query: Vector, k: number): Promise<SearchResult[]> {
    // Search across active partitions
    const results = await Promise.all(
      Array.from(this.activePartitions).map(pid => 
        this.partitions.get(pid)?.searchHNSW(query, k)
      )
    );
    
    // Merge and sort results
    return mergeSearchResults(results, k);
  }
}
```

## Best Practices

1. **Choosing an Implementation**
   - Small datasets (<100K vectors): Use VectorDB
   - Medium datasets with clusters: Use ClusteredVectorDB
   - Large production datasets: Use PartitionedVectorDB

2. **Memory Management**
   - Monitor memory usage with getStats()
   - Adjust partition sizes based on available RAM
   - Use vector compression for large datasets

3. **Search Optimization**
   - Enable HNSW indexing for large partitions
   - Tune cluster sizes for balanced performance
   - Use appropriate distance metrics for your data

4. **Data Organization**
   - Group similar vectors in same partition
   - Keep partition sizes balanced
   - Use meaningful partition IDs

5. **Monitoring**
   - Track partition statistics
   - Monitor search latencies
   - Watch memory consumption
