# Search Techniques in NBase

## Overview

NBase implements multiple state-of-the-art search algorithms optimized for different use cases, data scales, and performance requirements. This guide explains the available search techniques, their characteristics, and how to choose the right approach for your application.

## Search Algorithm Comparison

| Algorithm | Time Complexity | Accuracy | Memory Usage | Best For | Index Build Time |
|-----------|----------------|----------|--------------|----------|------------------|
| **Flat (Brute Force)** | O(n) | 100% | Low | Small datasets (< 10K) | None |
| **HNSW** | O(log n) | 95-99% | Medium | Large datasets, high accuracy | Medium |
| **LSH** | O(1) | 80-95% | Low | Speed-critical, approximate | Fast |
| **KNN** | O(n) | 100% | Medium | Exact search, custom metrics | None |
| **Hybrid** | Varies | Configurable | High | Complex multi-modal search | Varies |

## Flat Search (Brute Force)

### Overview
The most basic search method that compares the query vector against every vector in the dataset.

### When to Use
- Small datasets (< 10,000 vectors)
- 100% accuracy required
- No indexing overhead desired
- Development and testing

### Usage

```typescript
// Basic flat search
const results = await db.search(queryVector, {
  k: 10,
  useHNSW: false,  // Explicitly disable HNSW
  distanceMetric: 'cosine'
});

// Flat search with filtering
const results = await db.search(queryVector, {
  k: 5,
  useHNSW: false,
  filter: (id, metadata) => metadata.category === 'important',
  includeMetadata: true
});
```

### Performance Characteristics
- **Search Time**: Linear with dataset size
- **Memory Usage**: Minimal (just stores vectors)
- **Accuracy**: Perfect (100%)
- **Scalability**: Poor for large datasets

## HNSW (Hierarchical Navigable Small World)

### Overview
Graph-based approximate nearest neighbor search using hierarchical navigable small world graphs. Provides excellent balance between speed and accuracy.

### How It Works

HNSW builds a multi-layer graph structure:
- **Top layers**: Sparse, long-range connections for fast navigation
- **Bottom layer**: Dense connections for accurate local search
- **Search process**: Start from top layer, navigate down refining results

```
Layer 2:   A ──────► B
           │        │
Layer 1:   A ───► B ───► C
           │      │      │
Layer 0:   A ──► B ──► C ──► D ──► E
```

### Configuration Parameters

```typescript
const hnswConfig = {
  M: 16,                    // Max connections per node (4-64, default: 16)
  efConstruction: 200,      // Index build quality (100-800, default: 200)
  efSearch: 100,           // Search accuracy vs speed (32-800, default: 100)
  maxLevel: 10,            // Maximum graph layers
  distanceMetric: 'cosine' // Distance function
};
```

### Usage

```typescript
// High-accuracy HNSW search
const results = await db.search(queryVector, {
  k: 10,
  useHNSW: true,
  efSearch: 200,  // Higher = more accurate but slower
  includeMetadata: true
});

// Fast HNSW search
const results = await db.search(queryVector, {
  k: 10,
  useHNSW: true,
  efSearch: 50,   // Lower = faster but less accurate
  distanceMetric: 'euclidean'
});

// Balanced configuration
const results = await db.search(queryVector, {
  k: 10,
  useHNSW: true,
  efSearch: 100,  // Default balance
  partitionIds: ['p1', 'p2']  // Search specific partitions
});
```

### Performance Tuning

#### For Maximum Accuracy
```typescript
const highAccuracyConfig = {
  M: 32,               // More connections
  efConstruction: 400, // Higher build quality
  efSearch: 300        // Higher search quality
};
```

#### For Maximum Speed
```typescript
const highSpeedConfig = {
  M: 8,                // Fewer connections
  efConstruction: 100, // Faster building
  efSearch: 32         // Faster search
};
```

#### For Memory Efficiency
```typescript
const memoryEfficientConfig = {
  M: 12,               // Moderate connections
  efConstruction: 150, // Balanced build
  efSearch: 64         // Balanced search
};
```

### Index Building

```typescript
// Build HNSW index with progress tracking
await db.buildIndexes({
  indexType: 'hnsw',
  progressCallback: (progress) => {
    console.log(`HNSW build: ${progress}% complete`);
  },
  hnswOptions: {
    M: 16,
    efConstruction: 200,
    dimensionAware: true
  }
});

// Save/load HNSW indexes
await db.saveHNSWIndices();
await db.loadHNSWIndices('partition-001');
```

## LSH (Locality-Sensitive Hashing)

### Overview
Ultra-fast approximate search using hash functions that preserve similarity. Excellent for speed-critical applications.

### How It Works

1. **Hash Functions**: Multiple hash functions map similar vectors to same buckets
2. **Indexing**: Vectors stored in hash buckets
3. **Search**: Query hashed, candidates retrieved from matching buckets
4. **Refinement**: Top candidates re-ranked for accuracy

### Configuration

```typescript
const lshConfig = {
  numberOfHashes: 10,       // Hash functions (5-20)
  numberOfBuckets: 100,     // Hash buckets (50-1000)
  allowMismatchedDimensions: false
};
```

### Usage

```typescript
// Fast LSH search
const results = await db.search(queryVector, {
  k: 10,
  useHNSW: false,  // Use LSH instead of HNSW
  distanceMetric: 'cosine'
});

// LSH with custom parameters
const results = await db.search(queryVector, {
  k: 20,
  useHNSW: false,
  lshOptions: {
    numberOfHashes: 15,
    numberOfBuckets: 200
  }
});
```

### Performance Characteristics
- **Search Time**: Near constant O(1)
- **Memory Usage**: Low (hash tables)
- **Accuracy**: 80-95% (depends on hash quality)
- **Index Build**: Fast

## KNN (K-Nearest Neighbors)

### Overview
Exact nearest neighbor search with multi-threading and advanced optimization techniques.

### Features
- 100% accuracy guaranteed
- Multi-threaded execution
- Spatial partitioning
- Vectorized calculations
- Result caching
- Early stopping optimization

### Configuration

```typescript
const knnConfig = {
  metric: 'cosine',           // Distance metric
  useMultithreading: true,    // Enable parallel processing
  useHeap: true,             // Use heap for efficiency
  batchSize: 1000,           // Processing batch size
  earlyStoppingThreshold: 0.8, // Stop early when threshold reached
  maxThreads: 4,             // Maximum threads
  cacheResults: true,        // Cache results
  blockSize: 1024           // Memory block size
};
```

### Usage

```typescript
// Exact KNN search
const results = await db.findNearest(queryVector, 10, {
  distanceMetric: 'cosine',
  useMultithreading: true,
  earlyStoppingThreshold: 0.9
});

// KNN with spatial partitioning
const results = await db.findNearest(queryVector, 5, {
  spatialPartitioning: true,
  partitionCount: 8,
  distanceMetric: 'euclidean'
});
```

## Hybrid Search

### Overview
Intelligent combination of multiple search algorithms for optimal results. Useful for multi-modal search and complex ranking requirements.

### How It Works

1. **Parallel Execution**: Run multiple search algorithms simultaneously
2. **Result Combination**: Merge and deduplicate results
3. **Weighted Ranking**: Apply custom weights to different algorithms
4. **Reranking**: Apply diversity or quality reranking

### Usage

```typescript
// Multi-algorithm hybrid search
const results = await db.search(queryVector, {
  k: 10,
  searchMethod: 'hybrid',
  methods: ['hnsw', 'lsh', 'flat'],
  weights: {
    hnsw: 0.5,   // 50% weight
    lsh: 0.3,    // 30% weight
    flat: 0.2    // 20% weight
  },
  useParallelExecution: true
});

// Hybrid with reranking
const results = await db.search(queryVector, {
  k: 15,
  searchMethod: 'hybrid',
  methods: ['hnsw'],
  rerank: true,
  rerankingMethod: 'diversity',  // diversity or standard
  rerankLambda: 0.5             // Reranking strength
});
```

### Advanced Hybrid Configuration

```typescript
const hybridConfig = {
  // Algorithm selection
  methods: ['hnsw', 'lsh'],
  useParallelExecution: true,

  // Weight configuration
  weights: {
    hnsw: 0.6,
    lsh: 0.4
  },

  // Individual algorithm options
  hnswOptions: {
    efSearch: 150
  },
  lshOptions: {
    numberOfHashes: 12
  },

  // Reranking
  rerank: true,
  rerankingMethod: 'diversity',
  rerankLambda: 0.3,

  // Result processing
  maxCandidates: 100,  // Max candidates before reranking
  diversityFactor: 0.1  // Diversity strength
};
```

## Batch Search

### Overview
Efficient processing of multiple search queries in parallel. Optimizes resource usage and reduces latency for bulk operations.

### Usage

```typescript
// Prepare batch queries
const queries = [
  {
    query: vector1,
    k: 5,
    options: {
      includeMetadata: true,
      distanceMetric: 'cosine',
      useHNSW: true
    }
  },
  {
    query: vector2,
    k: 10,
    options: {
      includeMetadata: true,
      filter: (id, meta) => meta.category === 'science'
    }
  },
  {
    query: vector3,
    k: 3,
    options: {
      distanceMetric: 'euclidean',
      efSearch: 200
    }
  }
];

// Execute batch search
const batchResults = await db.batchSearch(queries, {
  maxBatchSize: 50,           // Process in batches
  prioritizeOrder: true,      // Maintain input order
  groupSimilarQueries: true,  // Optimize similar queries
  defaultSearchTimeout: 5000, // Timeout per query (ms)
  maxWorkers: 4               // Parallel workers
});

console.log(`Processed ${batchResults.length} query sets`);
// Output: Array of result arrays, one per query
```

### Batch Search Optimization

```typescript
// Optimize for different scenarios

// High-throughput batch processing
const highThroughputConfig = {
  maxBatchSize: 100,
  maxWorkers: 8,
  prioritizeOrder: false,  // Speed over order
  groupSimilarQueries: true
};

// Low-latency batch processing
const lowLatencyConfig = {
  maxBatchSize: 10,
  maxWorkers: 2,
  prioritizeOrder: true,   // Maintain order
  defaultSearchTimeout: 1000
};

// Memory-efficient batch processing
const memoryEfficientConfig = {
  maxBatchSize: 25,
  maxWorkers: 4,
  groupSimilarQueries: false  // Reduce memory usage
};
```

## Distance Metrics

### Available Metrics

| Metric | Formula | Use Case | Range |
|--------|---------|----------|-------|
| **Cosine** | `1 - (A·B)/(‖A‖‖B‖)` | Text embeddings, direction | [0, 2] |
| **Euclidean** | `√∑(Ai-Bi)²` | Spatial data, coordinates | [0, ∞) |
| **Manhattan** | `∑│Ai-Bi│` | Grid-based data | [0, ∞) |
| **Dot Product** | `A·B` | Raw similarity scores | (-∞, ∞) |

### Metric Selection Guide

```typescript
// Text embeddings (OpenAI, BERT, etc.)
const textSearch = await db.search(queryVector, {
  k: 10,
  distanceMetric: 'cosine',  // Best for normalized embeddings
  useHNSW: true
});

// Spatial coordinates
const spatialSearch = await db.search(queryVector, {
  k: 5,
  distanceMetric: 'euclidean',  // Best for coordinates
  useHNSW: true
});

// High-dimensional scientific data
const scientificSearch = await db.search(queryVector, {
  k: 8,
  distanceMetric: 'euclidean',
  useHNSW: true,
  efSearch: 300  // Higher accuracy for scientific data
});
```

## Reranking Techniques

### Overview
Improve search result quality through post-processing reranking algorithms.

### Available Methods

#### Diversity Reranking
```typescript
const results = await db.search(queryVector, {
  k: 20,
  rerank: true,
  rerankingMethod: 'diversity',
  rerankLambda: 0.5  // Balance between relevance and diversity
});
```

#### Standard Reranking
```typescript
const results = await db.search(queryVector, {
  k: 15,
  rerank: true,
  rerankingMethod: 'standard',
  rerankLambda: 0.3
});
```

#### Weighted Reranking
```typescript
const results = await db.search(queryVector, {
  k: 10,
  rerank: true,
  rerankingMethod: 'weighted',
  weights: {
    relevance: 0.7,
    recency: 0.2,
    popularity: 0.1
  }
});
```

## Performance Optimization

### Algorithm Selection Guide

```typescript
function chooseSearchAlgorithm(datasetSize, accuracy, speed) {
  if (datasetSize < 10000) {
    return { algorithm: 'flat', accuracy: '100%', speed: 'slow' };
  }

  if (speed === 'critical') {
    return { algorithm: 'lsh', accuracy: '80-95%', speed: 'fastest' };
  }

  if (accuracy === 'maximum') {
    return { algorithm: 'hnsw', accuracy: '95-99%', speed: 'medium' };
  }

  if (accuracy === 'exact') {
    return { algorithm: 'knn', accuracy: '100%', speed: 'slow' };
  }

  // Default balanced approach
  return { algorithm: 'hnsw', accuracy: '95-99%', speed: 'medium' };
}
```

### Memory vs Speed Trade-offs

```typescript
// Memory-optimized configuration
const memoryOptimized = {
  algorithm: 'hnsw',
  hnswOptions: {
    M: 8,              // Fewer connections = less memory
    efConstruction: 100, // Faster building = less memory
    efSearch: 64       // Lower search quality = faster
  },
  cacheSize: 500       // Smaller cache
};

// Speed-optimized configuration
const speedOptimized = {
  algorithm: 'lsh',
  lshOptions: {
    numberOfHashes: 8,   // Fewer hashes = faster
    numberOfBuckets: 50  // Fewer buckets = faster
  },
  cacheSize: 2000       // Larger cache for repeated queries
};

// Accuracy-optimized configuration
const accuracyOptimized = {
  algorithm: 'hnsw',
  hnswOptions: {
    M: 32,               // More connections = better accuracy
    efConstruction: 400, // Higher build quality = better accuracy
    efSearch: 300        // Higher search quality = better accuracy
  },
  cacheSize: 1000
};
```

## Monitoring and Profiling

### Search Performance Monitoring

```typescript
// Enable detailed search monitoring
const db = new Database({
  monitoring: {
    enable: true,
    enableSearchMetrics: true,
    intervalMs: 5000
  }
});

// Monitor search events
db.on('search:complete', (event) => {
  console.log(`Search completed: ${event.totalTime}ms, ${event.resultCount} results`);
  console.log(`Method used: ${event.options.useHNSW ? 'HNSW' : 'Flat'}`);
});

// Get performance statistics
const stats = await db.getStats();
console.log('Search Performance:');
console.log(`Average search time: ${stats.search.avgTime}ms`);
console.log(`Cache hit rate: ${stats.searchCache.hitRate}%`);
console.log(`Queries per minute: ${stats.search.queriesPerMinute}`);
```

### Profiling Different Algorithms

```typescript
async function profileSearchAlgorithms(query, k = 10) {
  const algorithms = [
    { name: 'Flat', options: { useHNSW: false } },
    { name: 'HNSW-Fast', options: { useHNSW: true, efSearch: 50 } },
    { name: 'HNSW-Balanced', options: { useHNSW: true, efSearch: 100 } },
    { name: 'HNSW-Accurate', options: { useHNSW: true, efSearch: 200 } },
    { name: 'LSH', options: { useHNSW: false } }
  ];

  const results = {};

  for (const algo of algorithms) {
    const start = Date.now();
    const searchResults = await db.search(query, { k, ...algo.options });
    const time = Date.now() - start;

    results[algo.name] = {
      time,
      resultCount: searchResults.length,
      firstResult: searchResults[0]
    };
  }

  return results;
}

// Usage
const profile = await profileSearchAlgorithms(queryVector);
console.table(profile);
```

This comprehensive guide covers all major search techniques in NBase. Choose the appropriate algorithm based on your dataset size, accuracy requirements, and performance needs. For most applications, HNSW provides the best balance between speed and accuracy.
