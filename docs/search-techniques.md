# Search Techniques

## Overview

NBase implements multiple search algorithms optimized for different use cases and data scales. Understanding these techniques helps in choosing the right approach for your needs.

## Available Search Methods

### 1. Standard Search (Brute Force)

Basic exhaustive search across all vectors.

```typescript
const results = await db.search(queryVector, {
  k: 10,
  useHNSW: false
});
```

Characteristics:
- 100% accuracy
- Linear time complexity O(n)
- Best for small datasets (<10K vectors)
- No indexing overhead

### 2. HNSW Search

Hierarchical Navigable Small World graphs for approximate nearest neighbor search.

```typescript
const results = await db.search(queryVector, {
  k: 10,
  useHNSW: true,
  efSearch: 100
});
```

Characteristics:
- Logarithmic search time O(log n)
- High accuracy (95-99%)
- Memory efficient
- Best for large datasets

Parameters:
- `M`: Max connections per node (default: 16)
- `efConstruction`: Build quality (default: 200)
- `efSearch`: Search quality (default: 100)

### 3. Hybrid Search

Combines multiple search methods for optimal results.

```typescript
const results = await db.search(queryVector, {
  k: 10,
  method: 'hybrid',
  weights: {
    semantic: 0.7,
    text: 0.3
  }
});
```

Features:
- Multi-modal search
- Weighted combinations
- Flexible algorithm selection

## Performance Comparison

| Method | Time Complexity | Memory Usage | Accuracy | Best For |
|--------|----------------|--------------|----------|-----------|
| Standard | O(n) | Low | 100% | Small datasets |
| HNSW | O(log n) | Medium | 95-99% | Large datasets |
| Hybrid | Varies | High | Configurable | Complex queries |

## Implementation Details

### HNSW Index Structure

```plaintext
Layer 2:   0 -----> 1
           |        |
Layer 1:   0 ---> 1 ---> 2
           |      |      |
Layer 0:   0 --> 1 --> 2 --> 3 --> 4
```

### Search Process

1. **Standard Search**
```typescript
function standardSearch(query: Vector, vectors: Vector[], k: number) {
  return vectors
    .map((vec, id) => ({
      id,
      distance: calculateDistance(query, vec)
    }))
    .sort((a, b) => a.distance - b.distance)
    .slice(0, k);
}
```

2. **HNSW Search**
```typescript
function hnswSearch(query: Vector, entryPoint: number, k: number) {
  let current = entryPoint;
  
  // Search from top layer down
  for (let layer = maxLayer; layer >= 0; layer--) {
    current = greedySearchLayer(query, current, layer);
  }
  
  // Final search on bottom layer
  return searchBaseLayer(query, current, k);
}
```

## Distance Metrics

### Cosine Similarity
```typescript
function cosineSimilarity(a: Vector, b: Vector): number {
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;
  
  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  
  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}
```

### Euclidean Distance
```typescript
function euclideanDistance(a: Vector, b: Vector): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const diff = a[i] - b[i];
    sum += diff * diff;
  }
  return Math.sqrt(sum);
}
```

## Search Optimization

### 1. Index Configuration

```typescript
await db.buildIndexes({
  method: 'hnsw',
  M: 16,               // More connections = better accuracy, more memory
  efConstruction: 200, // Higher = better index quality, slower build
  progressCallback: (progress) => console.log(`${progress}%`)
});
```

### 2. Query Optimization

```typescript
const results = await db.search(queryVector, {
  k: 10,
  efSearch: 100,        // Higher = better recall, slower search
  useHNSW: true,
  filter: (id, meta) => meta.score > 0.5,
  includeMetadata: true
});
```

### 3. Batch Processing

```typescript
const queries = [
  { vector: [...], k: 5 },
  { vector: [...], k: 10 }
];

const results = await db.batchSearch(queries, {
  maxBatchSize: 100,
  maxWorkers: 4
});
```

## Best Practices

1. **Index Selection**
   - Small datasets (<10K vectors): Use standard search
   - Medium/Large datasets: Use HNSW
   - Complex queries: Use hybrid search

2. **Performance Tuning**
   - Adjust efSearch based on accuracy needs
   - Monitor search latencies
   - Use batch operations for multiple queries
   - Enable vector compression for large datasets

3. **Memory Management**
   - Monitor index memory usage
   - Use appropriate M values for HNSW
   - Consider partition size limits

4. **Distance Metrics**
   - Use cosine similarity for normalized vectors
   - Use euclidean distance for absolute measures
   - Consider domain-specific metrics
