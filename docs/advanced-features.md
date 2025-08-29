# Advanced Features

NBase provides powerful advanced features for complex vector operations, large-scale processing, and specialized search algorithms. This guide covers relationship extraction, community detection, hybrid search, batch processing, and performance optimization techniques.

## Vector Relationships

### Extracting Relationships

Find relationships between vectors based on distance thresholds:

```javascript
// Extract relationships with distance threshold
const relationships = await db.extractRelationships(0.3, {
  metric: 'cosine',
  includeMetadata: true,
  partitionIds: ['p1', 'p2']  // Limit to specific partitions
});

console.log('Relationships found:', relationships.length);
// Output: [
//   {
//     vector1: { id: 'doc1', partitionId: 'p1', metadata: {...} },
//     vector2: { id: 'doc2', partitionId: 'p1', metadata: {...} },
//     distance: 0.25
//   },
//   ...
// ]
```

### Relationship Analysis

```javascript
// Find strongly related documents
const strongRelationships = relationships.filter(rel => rel.distance < 0.2);

// Group relationships by partition
const relationshipsByPartition = relationships.reduce((acc, rel) => {
  const pid = rel.vector1.partitionId;
  if (!acc[pid]) acc[pid] = [];
  acc[pid].push(rel);
  return acc;
}, {});
```

## Community Detection

### Finding Communities

Extract clusters (communities) of closely related vectors:

```javascript
// Detect communities with distance threshold
const communities = await db.extractCommunities(0.5, {
  metric: 'euclidean',
  includeMetadata: true,
  partitionIds: ['p1', 'p2', 'p3']
});

console.log(`Found ${communities.length} communities`);
// Output: [
//   [
//     { id: 'doc1', partitionId: 'p1', metadata: {...} },
//     { id: 'doc2', partitionId: 'p1', metadata: {...} },
//     { id: 'doc5', partitionId: 'p2', metadata: {...} }
//   ],
//   [
//     { id: 'doc3', partitionId: 'p1', metadata: {...} },
//     { id: 'doc4', partitionId: 'p1', metadata: {...} }
//   ],
//   ...
// ]
```

### Community Analysis

```javascript
// Analyze community characteristics
communities.forEach((community, index) => {
  const avgMetadata = analyzeCommunityMetadata(community);
  console.log(`Community ${index}: ${community.length} vectors, avg score: ${avgMetadata.avgScore}`);
});

// Find largest communities
const largestCommunities = communities
  .sort((a, b) => b.length - a.length)
  .slice(0, 5);
```

## Hybrid Search

### Multi-Modal Search

Combine different search methods for improved results:

```javascript
// Hybrid search with multiple algorithms
const results = await db.search(queryVector, {
  k: 10,
  searchMethod: 'hybrid',
  methods: ['hnsw', 'lsh'],  // Combine HNSW and LSH
  weights: {
    hnsw: 0.7,
    lsh: 0.3
  },
  useParallelExecution: true
});
```

### Custom Scoring

```javascript
// Implement custom hybrid scoring
const hnswResults = await db.search(queryVector, {
  k: 20,
  useHNSW: true,
  includeMetadata: true
});

const lshResults = await db.search(queryVector, {
  k: 20,
  useHNSW: false,  // Use flat/LSH
  includeMetadata: true
});

// Combine and rerank results
const combinedResults = combineResults(hnswResults, lshResults, {
  hnswWeight: 0.6,
  lshWeight: 0.4,
  diversityFactor: 0.1
});
```

## Batch Processing

### Batch Search

Process multiple search queries efficiently:

```javascript
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
      filter: (id, metadata) => metadata.category === 'science'
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
  maxBatchSize: 50,        // Process in batches of 50
  prioritizeOrder: true,   // Maintain input order
  groupSimilarQueries: true, // Optimize similar queries
  defaultSearchTimeout: 5000 // 5 second timeout per query
});

console.log(`Processed ${batchResults.length} query sets`);
```

### Batch Operations

```javascript
// Bulk vector operations
const vectors = Array.from({ length: 1000 }, (_, i) => ({
  id: `vec-${i}`,
  vector: generateRandomVector(1536),
  metadata: { batch: 'test-batch', index: i }
}));

// Add in bulk
const addResult = await db.bulkAdd(vectors);
console.log(`Added ${addResult.count} vectors to partitions: ${addResult.partitionIds.join(', ')}`);

// Bulk metadata updates
const updates = vectors.map(v => ({
  id: v.id,
  metadata: { ...v.metadata, processed: true, timestamp: Date.now() }
}));

for (const update of updates) {
  await db.updateMetadata(update.id, update.metadata);
}
```

## Advanced Filtering

### Metadata Filtering

```javascript
// Simple metadata filter
const results = await db.search(queryVector, {
  k: 10,
  includeMetadata: true,
  filter: (id, metadata) => {
    return metadata.category === 'technology' &&
           metadata.rating >= 4.5 &&
           metadata.published > '2023-01-01';
  }
});
```

### Complex Filter Objects

```javascript
// Advanced filter configuration
const complexFilter = {
  category: { $in: ['article', 'blog', 'research'] },
  rating: { $gte: 4.0 },
  tags: { $regex: '^tech.*' },
  published: { $gt: '2023-06-01' },
  author: { $ne: 'Anonymous' },
  views: { $between: [1000, 10000] }
};

const results = await db.search(queryVector, {
  k: 20,
  filters: complexFilter,
  includeMetadata: true
});
```

### Custom Filter Functions

```javascript
// Custom filter with complex logic
const customFilter = (id, metadata) => {
  // Time-based filtering
  const age = Date.now() - new Date(metadata.published).getTime();
  const isRecent = age < 365 * 24 * 60 * 60 * 1000; // 1 year

  // Content-based filtering
  const hasRequiredTags = metadata.tags?.some(tag =>
    ['machine-learning', 'ai', 'neural-networks'].includes(tag)
  );

  // Quality filtering
  const isHighQuality = metadata.rating > 4.0 && metadata.views > 500;

  return isRecent && hasRequiredTags && isHighQuality;
};

const results = await db.search(queryVector, {
  k: 15,
  filter: customFilter,
  includeMetadata: true
});
```

## Index Management

### Advanced Index Building

```javascript
// Build indexes with progress tracking
await db.buildIndexes({
  progressCallback: (progress) => {
    console.log(`Index building: ${progress}% complete`);
  },
  dimensionAware: true,
  force: true  // Rebuild even if exists
});

// Build specific index types
await db.buildIndexes({
  indexType: 'hnsw',
  hnswOptions: {
    M: 32,                    // More connections for better accuracy
    efConstruction: 400,      // Higher build quality
    dimensionAware: true
  }
});
```

### Index Optimization

```javascript
// Optimize indexes for specific workloads
const optimizationConfig = {
  // For high-accuracy search
  highAccuracy: {
    M: 24,
    efConstruction: 300,
    efSearch: 200
  },

  // For fast search
  highSpeed: {
    M: 12,
    efConstruction: 100,
    efSearch: 50
  },

  // For memory efficiency
  memoryOptimized: {
    M: 8,
    efConstruction: 50,
    efSearch: 32
  }
};

// Apply optimization
await db.buildIndexes({
  ...optimizationConfig.highAccuracy,
  progressCallback: console.log
});
```

### Index Statistics and Monitoring

```javascript
// Get detailed index statistics
const stats = await db.getStats();
const indexStats = stats.database.indices;

console.log('Index Statistics:');
console.log(`HNSW Loaded: ${indexStats.hnswLoadedCount}/${indexStats.totalPartitions}`);
console.log('HNSW Details:');
Object.entries(indexStats.hnswStats).forEach(([partitionId, hnswStat]) => {
  console.log(`  ${partitionId}: ${hnswStat.totalNodes} nodes, ${hnswStat.levels} levels`);
});
```

## Performance Monitoring

### Comprehensive Monitoring Setup

```typescript
const db = new Database({
  monitoring: {
    enable: true,
    intervalMs: 5000,          // Collect every 5 seconds
    logToConsole: true,
    enableSystemMetrics: true, // CPU, memory, disk
    enableSearchMetrics: true, // Query performance
    enableDatabaseMetrics: true, // Vector/index stats
    enableCacheMetrics: true   // Cache hit rates
  }
});
```

### Real-time Performance Tracking

```javascript
// Monitor search performance
db.on('search:complete', (event) => {
  console.log(`Search completed in ${event.totalTime}ms`);
  console.log(`Results: ${event.resultCount}, K: ${event.kRequested}`);
});

// Monitor system resources
setInterval(async () => {
  const metrics = await db.getStats();

  console.log('System Metrics:');
  console.log(`CPU: ${(metrics.system.cpuUsage * 100).toFixed(1)}%`);
  console.log(`Memory: ${(metrics.system.memoryUsage * 100).toFixed(1)}%`);
  console.log(`Search Cache: ${metrics.searchCache.hitRate.toFixed(1)}% hit rate`);

  console.log('Database Metrics:');
  console.log(`Total Vectors: ${metrics.database.vectors.totalConfigured}`);
  console.log(`Active Partitions: ${metrics.database.partitions.loadedCount}`);
}, 10000);
```

### Performance Profiling

```typescript
// Profile search operations
async function profileSearch(query, options, iterations = 100) {
  const times = [];

  for (let i = 0; i < iterations; i++) {
    const start = Date.now();
    await db.search(query, options);
    times.push(Date.now() - start);
  }

  const avg = times.reduce((a, b) => a + b) / times.length;
  const p95 = times.sort((a, b) => a - b)[Math.floor(times.length * 0.95)];

  return { avg, p95, min: Math.min(...times), max: Math.max(...times) };
}

// Profile different configurations
const configs = [
  { useHNSW: true, efSearch: 100 },
  { useHNSW: true, efSearch: 200 },
  { useHNSW: false }  // Flat search
];

for (const config of configs) {
  const profile = await profileSearch(queryVector, { k: 10, ...config });
  console.log(`${JSON.stringify(config)}: ${profile.avg.toFixed(1)}ms avg, ${profile.p95}ms p95`);
}
```

## Custom Distance Metrics

### Implementing Custom Metrics

```typescript
// Custom distance function
function customDistance(a, b) {
  // Implement your custom distance calculation
  // Example: Weighted cosine with custom weighting
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  const weights = [1.0, 0.8, 0.6, ...]; // Custom weights per dimension

  for (let i = 0; i < a.length; i++) {
    const weight = weights[i] || 1.0;
    dotProduct += (a[i] * b[i]) * weight;
    normA += (a[i] * a[i]) * weight;
    normB += (b[i] * b[i]) * weight;
  }

  return 1 - (dotProduct / (Math.sqrt(normA) * Math.sqrt(normB)));
}

// Use custom metric (requires custom implementation)
const results = await db.search(queryVector, {
  k: 10,
  distanceMetric: customDistance
});
```

## Memory Optimization

### Large Dataset Handling

```typescript
// Optimize for large datasets
const db = new Database({
  partitioning: {
    partitionCapacity: 50000,      // Smaller partitions
    maxActivePartitions: 2,        // Limit memory usage
    autoCreatePartitions: true
  },
  clustering: {
    useCompression: true,          // Enable compression
    clusterSize: 2000             // Smaller clusters
  },
  cacheSize: 1000,                // Smaller cache
  indexing: {
    hnswOptions: {
      M: 12,                      // Fewer connections
      efConstruction: 100         // Faster building
    }
  }
});
```

### Memory Usage Monitoring

```typescript
// Monitor memory usage
function monitorMemory() {
  const usage = process.memoryUsage();
  console.log(`RSS: ${(usage.rss / 1024 / 1024).toFixed(1)}MB`);
  console.log(`Heap Used: ${(usage.heapUsed / 1024 / 1024).toFixed(1)}MB`);
  console.log(`Heap Total: ${(usage.heapTotal / 1024 / 1024).toFixed(1)}MB`);
  console.log(`External: ${(usage.external / 1024 / 1024).toFixed(1)}MB`);
}

// Monitor every minute
setInterval(monitorMemory, 60000);
```

## Advanced Persistence

### Custom Backup Strategy

```typescript
// Implement custom backup with metadata
async function createIntelligentBackup(backupPath, options = {}) {
  const {
    includeMetadata = true,
    compressVectors = true,
    maxPartitionSize = 100000,
    parallelPartitions = 2
  } = options;

  // Get database state
  const stats = await db.getStats();

  // Create backup directory
  const backupDir = path.join(backupPath, `backup-${Date.now()}`);
  await fs.mkdir(backupDir, { recursive: true });

  // Backup partitions in parallel
  const partitionPromises = stats.database.partitions.configs
    .filter(p => p.active)
    .slice(0, parallelPartitions)
    .map(async (partition) => {
      const partitionBackup = await backupPartition(partition.id, backupDir, {
        includeMetadata,
        compressVectors
      });
      return partitionBackup;
    });

  const partitionBackups = await Promise.all(partitionPromises);

  // Create backup manifest
  const manifest = {
    timestamp: new Date().toISOString(),
    version: '1.0',
    database: {
      vectorCount: stats.database.vectors.totalConfigured,
      partitionCount: stats.database.partitions.totalConfigured
    },
    partitions: partitionBackups,
    options: { includeMetadata, compressVectors, maxPartitionSize }
  };

  await fs.writeFile(
    path.join(backupDir, 'manifest.json'),
    JSON.stringify(manifest, null, 2)
  );

  return backupDir;
}
```

## Integration Patterns

### Microservices Integration

```javascript
// Vector service client
class VectorServiceClient {
  constructor(baseURL) {
    this.baseURL = baseURL;
  }

  async search(query, options = {}) {
    const response = await fetch(`${this.baseURL}/search`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, ...options })
    });
    return response.json();
  }

  async addVectors(vectors) {
    const response = await fetch(`${this.baseURL}/vectors`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ vectors })
    });
    return response.json();
  }
}

// Usage
const vectorClient = new VectorServiceClient('http://localhost:1307');
const results = await vectorClient.search(queryVector, { k: 10 });
```

### Streaming and Pagination

```javascript
// Implement streaming search for large result sets
async function* streamingSearch(query, options = {}) {
  const batchSize = options.batchSize || 100;
  let offset = 0;

  while (true) {
    const batch = await db.search(query, {
      ...options,
      k: batchSize,
      offset: offset
    });

    if (batch.length === 0) break;

    yield batch;
    offset += batch.length;

    // Prevent infinite loops
    if (offset > options.maxResults) break;
  }
}

// Usage
for await (const batch of streamingSearch(queryVector, {
  k: 1000,
  maxResults: 10000
})) {
  console.log(`Processing ${batch.length} results...`);
  // Process batch
}
```

These advanced features provide powerful capabilities for complex vector operations, large-scale processing, and specialized search requirements. Experiment with different configurations to optimize performance for your specific use case.
