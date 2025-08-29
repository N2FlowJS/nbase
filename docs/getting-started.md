# Getting Started with NBase

Welcome to NBase! This guide will help you get started with the high-performance vector database for efficient similarity search and neural search applications.

## Installation

Install NBase using npm:

```bash
npm install @n2flowjs/nbase
```

## Quick Start

### Basic Example

Here's a simple example to get you started:

```javascript
const { Database } = require('@n2flowjs/nbase');

// Initialize database with basic configuration
const db = new Database({
  vectorSize: 1536,  // Dimension size for OpenAI text-embedding-ada-002
  partitioning: {
    partitionsDir: './data/partitions',
    partitionCapacity: 10000
  },
  indexing: {
    buildOnStart: true,
    hnswOptions: {
      M: 16,  // HNSW connections per node
      efConstruction: 200
    }
  },
  cacheSize: 1000
});

// Wait for database to be ready
await db.ready();

// Add vectors with metadata
await db.addVector('doc1', [0.1, 0.2, 0.3, ...], {
  title: 'Introduction to Machine Learning',
  category: 'education',
  author: 'John Doe',
  published: '2024-01-15'
});

await db.addVector('doc2', [0.4, 0.5, 0.6, ...], {
  title: 'Advanced Neural Networks',
  category: 'research',
  author: 'Jane Smith',
  published: '2024-02-20'
});

// Search for similar vectors
const results = await db.search([0.15, 0.25, 0.35, ...], {
  k: 5,  // Return top 5 results
  includeMetadata: true,
  distanceMetric: 'cosine',
  useHNSW: true  // Use HNSW index for faster search
});

console.log('Search results:', results);
// Output:
// [
//   {
//     id: 'doc1',
//     dist: 0.12,
//     metadata: {
//       title: 'Introduction to Machine Learning',
//       category: 'education',
//       author: 'John Doe',
//       published: '2024-01-15'
//     }
//   },
//   {
//     id: 'doc2',
//     dist: 0.45,
//     metadata: {
//       title: 'Advanced Neural Networks',
//       category: 'research',
//       author: 'Jane Smith',
//       published: '2024-02-20'
//     }
//   }
// ]
```

## Configuration

NBase offers extensive configuration options to optimize performance for your specific use case.

### Database Configuration

```typescript
const db = new Database({
  // Vector dimensions
  vectorSize: 1536,

  // Search result caching
  cacheSize: 1000,
  maxConcurrentSearches: 4,

  // Partitioning for scalability
  partitioning: {
    partitionsDir: './data/partitions',     // Storage directory
    partitionCapacity: 10000,               // Vectors per partition
    autoCreatePartitions: true,             // Auto-create new partitions
    maxActivePartitions: 3,                 // Max partitions in memory
    autoLoadPartitions: true                // Load partitions on startup
  },

  // Indexing for performance
  indexing: {
    buildOnStart: true,                     // Build indexes on startup
    autoSave: true,                         // Auto-save indexes
    hnswOptions: {
      M: 16,                                // HNSW connections per node
      efConstruction: 200,                  // Index build quality
      efSearch: 100                         // Search quality vs speed
    }
  },

  // Clustering for organization
  clustering: {
    clusterSize: 1000,                      // Target cluster size
    distanceMetric: 'cosine',               // Clustering metric
    useCompression: true,                   // Vector compression
    kmeansMaxIterations: 50                 // K-means iterations
  },

  // Persistence
  persistence: {
    dbPath: './data',                       // Database path
    autoSave: true,                         // Auto-save database
    saveIntervalMs: 300000                  // Save every 5 minutes
  },

  // Performance monitoring
  monitoring: {
    enable: true,
    intervalMs: 5000,                       // Collect metrics every 5s
    enableSystemMetrics: true,
    enableSearchMetrics: true,
    enableDatabaseMetrics: true
  }
});
```

### Configuration Guide

#### When to Use Different Settings

| Use Case | Recommended Configuration |
|----------|---------------------------|
| **Small Dataset (< 10K vectors)** | Single partition, no clustering, basic indexing |
| **Medium Dataset (10K - 100K)** | Multiple partitions, light clustering, HNSW indexing |
| **Large Dataset (> 100K)** | Many partitions, aggressive clustering, optimized HNSW |
| **High Accuracy Required** | Higher `efSearch`, more connections (`M`) |
| **Speed Critical** | Lower `efSearch`, LSH indexing, result caching |
| **Memory Limited** | Smaller partitions, compression enabled, LRU caching |

## Core Operations

### Vector Management

#### Adding Vectors

```javascript
// Add single vector
await db.addVector('unique-id', [0.1, 0.2, 0.3, ...], {
  title: 'Document Title',
  category: 'research',
  tags: ['machine-learning', 'neural-networks']
});

// Add multiple vectors efficiently
const vectors = [
  {
    id: 'doc1',
    vector: [0.1, 0.2, 0.3, ...],
    metadata: { title: 'Paper 1', author: 'Alice' }
  },
  {
    id: 'doc2',
    vector: [0.4, 0.5, 0.6, ...],
    metadata: { title: 'Paper 2', author: 'Bob' }
  }
];

const result = await db.bulkAdd(vectors);
console.log(`Added ${result.count} vectors to partitions: ${result.partitionIds.join(', ')}`);
```

#### Retrieving Vectors

```javascript
// Get vector data
const vectorData = await db.getVector('doc1');
if (vectorData) {
  console.log('Vector:', vectorData.vector);
  console.log('Metadata:', vectorData.metadata);
}

// Get only metadata
const metadata = await db.getMetadata('doc1');
console.log('Metadata:', metadata);
```

#### Updating and Deleting

```javascript
// Update metadata
await db.updateMetadata('doc1', {
  ...metadata,
  lastModified: new Date().toISOString(),
  version: 2
});

// Delete vector
const deleted = await db.deleteVector('doc1');
console.log(deleted ? 'Vector deleted' : 'Vector not found');
```

### Search Operations

#### Basic Similarity Search

```javascript
// Simple search
const results = await db.search(queryVector, {
  k: 10  // Return top 10 results
});

// Search with metadata
const results = await db.search(queryVector, {
  k: 5,
  includeMetadata: true,
  distanceMetric: 'cosine'
});
```

#### Advanced Search with Filters

```javascript
// Filter by metadata
const results = await db.search(queryVector, {
  k: 10,
  includeMetadata: true,
  filter: (id, metadata) => {
    return metadata.category === 'research' &&
           metadata.published > '2023-01-01' &&
           metadata.author !== 'Anonymous';
  }
});

// Search with different algorithms
const hnswResults = await db.search(queryVector, {
  k: 10,
  useHNSW: true,
  efSearch: 200  // Higher = more accurate but slower
});
```

#### Batch Search

```javascript
// Search multiple queries efficiently
const queries = [
  { query: vector1, k: 5, options: { includeMetadata: true } },
  { query: vector2, k: 3, options: { distanceMetric: 'euclidean' } }
];

const batchResults = await db.batchSearch(queries, {
  prioritizeOrder: true,  // Maintain input order in results
  maxBatchSize: 10        // Process in batches of 10
});
```

### Index Management

#### Building Indexes

```javascript
// Build all indexes
await db.buildIndexes({
  progressCallback: (progress) => {
    console.log(`Index building: ${progress}% complete`);
  }
});

// Build specific index type
await db.buildIndexes({
  indexType: 'hnsw',
  dimensionAware: true
});
```

#### Saving and Loading Indexes

```javascript
// Save indexes to disk
await db.saveHNSWIndices();

// Load indexes from disk
await db.loadHNSWIndices();

// Save/load specific partition
await db.saveHNSWIndices('partition-001');
await db.loadHNSWIndices('partition-001');
```

### Database Management

#### Persistence

```javascript
// Manual save
await db.save();

// Get database statistics
const stats = await db.getStats();
console.log('Database stats:', {
  totalVectors: stats.database.vectors.totalConfigured,
  partitions: stats.database.partitions.totalConfigured,
  memoryUsage: stats.memoryUsage,
  searchCache: stats.searchCache
});
```

#### Cleanup

```javascript
// Close database and free resources
await db.close();
```

## Working with Embeddings

### OpenAI Embeddings

```javascript
const { Configuration, OpenAIApi } = require('openai');

// Initialize OpenAI
const openai = new OpenAIApi(new Configuration({
  apiKey: process.env.OPENAI_API_KEY
}));

// Generate embeddings
async function generateEmbedding(text) {
  const response = await openai.createEmbedding({
    model: 'text-embedding-ada-002',
    input: text
  });
  return response.data.data[0].embedding;
}

// Store document with embedding
const text = 'Your document text here...';
const embedding = await generateEmbedding(text);

await db.addVector(`doc-${Date.now()}`, embedding, {
  text: text,
  model: 'text-embedding-ada-002',
  created: new Date().toISOString()
});
```

### Custom Embeddings

```javascript
// Example with TensorFlow.js or other embedding models
async function generateCustomEmbedding(text) {
  // Your embedding generation logic here
  // Return array of numbers with consistent dimensions
  return [0.1, 0.2, 0.3, ...]; // 1536 dimensions
}

// Batch process documents
const documents = [
  { id: 'doc1', text: 'First document...' },
  { id: 'doc2', text: 'Second document...' }
];

for (const doc of documents) {
  const embedding = await generateCustomEmbedding(doc.text);
  await db.addVector(doc.id, embedding, {
    text: doc.text,
    processed: new Date().toISOString()
  });
}
```

## REST API Server

NBase includes a built-in REST API server for easy integration:

```javascript
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
    partitioning: { partitionsDir: './data/partitions' },
    indexing: { buildOnStart: true }
  }
});

// Server starts automatically
console.log('NBase server running on http://localhost:1307');
```

### API Endpoints

```bash
# Add a vector
curl -X POST http://localhost:1307/vectors \
  -H "Content-Type: application/json" \
  -d '{
    "id": "doc1",
    "vector": [0.1, 0.2, 0.3],
    "metadata": { "title": "Example" }
  }'

# Search
curl -X POST http://localhost:1307/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": [0.1, 0.2, 0.3],
    "k": 5,
    "includeMetadata": true
  }'

# Get vector
curl http://localhost:1307/vectors/doc1

# Health check
curl http://localhost:1307/health
```

## Error Handling

NBase provides comprehensive error handling:

```javascript
try {
  await db.addVector('doc1', vector);
} catch (error) {
  switch (error.code) {
    case 'VECTOR_EXISTS':
      console.log('Vector ID already exists');
      break;
    case 'INVALID_DIMENSION':
      console.log('Vector dimension does not match database configuration');
      break;
    case 'PARTITION_FULL':
      console.log('Active partition is full, new partition will be created');
      break;
    default:
      console.error('Unexpected error:', error.message);
  }
}

// Handle search errors
db.on('search:error', (error) => {
  console.error('Search failed:', error);
});
```

## Performance Tips

### Optimization Strategies

1. **Choose the Right Index**: Use HNSW for accuracy, LSH for speed
2. **Tune Parameters**: Adjust `efSearch` based on your accuracy vs speed needs
3. **Use Caching**: Enable result caching for repeated queries
4. **Batch Operations**: Use bulk operations for multiple vectors
5. **Monitor Performance**: Enable monitoring to identify bottlenecks

### Memory Management

```javascript
// Configure for memory efficiency
const db = new Database({
  partitioning: {
    maxActivePartitions: 2,  // Limit memory usage
    partitionCapacity: 50000 // Smaller partitions
  },
  clustering: {
    useCompression: true     // Enable compression
  },
  cacheSize: 500            // Smaller cache
});
```

### Search Optimization

```javascript
// Fast approximate search
const results = await db.search(query, {
  k: 10,
  useHNSW: false,  // Use flat search for speed
  skipCache: true   // Skip cache for one-off queries
});

// High-accuracy search
const results = await db.search(query, {
  k: 10,
  useHNSW: true,
  efSearch: 300,    // Higher accuracy
  rerank: true      // Apply reranking
});
```

## Next Steps

- Explore the [API Reference](api-reference.md) for detailed method documentation
- Check out [Advanced Features](advanced-features.md) for clustering, compression, and more
- Review [Benchmarks](benchmarks.md) to understand performance characteristics
- Look at example projects in the repository

## Support

- 📖 [Documentation](https://github.com/N2FlowJS/nbase/tree/main/docs)
- 🐛 [Issues](https://github.com/N2FlowJS/nbase/issues)
- 💬 [Discussions](https://github.com/N2FlowJS/nbase/discussions)

Happy coding with NBase! 🚀
