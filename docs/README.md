# 🧠 NBase - Neural Vector Database

[![Made with Love](https://img.shields.io/badge/Made%20with-💖-pink.svg)](https://github.com/N2FlowJS/nbase)
[![GitHub stars](https://img.shields.io/github/stars/N2FlowJS/nbase)](https://github.com/N2FlowJS/nbase/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/N2FlowJS/nbase)](https://github.com/N2FlowJS/nbase/network/members)
[![GitHub issues](https://img.shields.io/github/issues/N2FlowJS/nbase)](https://github.com/N2FlowJS/nbase/issues)
[![GitHub license](https://img.shields.io/github/license/N2FlowJS/nbase)](https://github.com/N2FlowJS/nbase/blob/main/LICENSE)

```bash
╔═══════════════════════════════════════╗
║  _   _       ____                     ║
║ | \ | |     | __ )  __ _ ___  ___     ║
║ |  \| |_____| |_ \ / _` / __|/ _ \    ║
║ | |\  |_____| |_) | (_| \__ \  __/    ║
║ |_| \_|     |____/ \__,_|___/\___|    ║
║                                       ║
╚═══════════════════════════════════════╝
```

> 🚀 NBase is a high-performance vector database for efficient similarity search, designed for machine learning embeddings and neural search applications.

## ✨ Key Features

### 🌟 Core Capabilities
- 📦 **Enterprise-Grade Storage**
  - Store and manage millions of high-dimensional vectors
  - Built for production workloads
  - Automatic backup and recovery

### 🔍 Advanced Search Technologies
- 🎯 **State-of-the-Art Algorithms**
  - 🕸️ HNSW (Hierarchical Navigable Small World)
    - Logarithmic search time complexity
    - Optimized graph structure
  - 🎲 LSH (Locality-Sensitive Hashing)
    - Ultra-fast similarity search
    - Configurable hash functions
  - 📊 Smart Partitioning
    - Distributed search capabilities
    - Automatic load balancing

### 💪 Technical Excellence
- 📐 **Flexible Dimensionality**
  - Support for any vector dimension
  - Dynamic dimension handling
- 🗜️ **Intelligent Compression**
  - Advanced vector compression
  - Minimal quality loss

## Installation

[View on GitHub](https://github.com/N2FlowJS/nbase)

```bash
npm i @n2flowjs/nbase
```

## Build & Test

Clone the repository and install dependencies:

```bash
git clone https://github.com/N2FlowJS/nbase.git
cd nbase
npm install
```

Build the project:

```bash
npm run build
```

Run tests:

```bash
npm test
```

## Documentation

- [API Reference](https://github.com/N2FlowJS/nbase/blob/main/docs/api.md)
- [Benchmarks Guide](https://github.com/N2FlowJS/nbase/blob/main/docs/benchmarks.md)
- [Examples](https://github.com/N2FlowJS/nbase/tree/main/examples)
- [Changelog](https://github.com/N2FlowJS/nbase/blob/main/CHANGELOG.md)
- [Contributing Guide](https://github.com/N2FlowJS/nbase/blob/main/CONTRIBUTING.md)

## Community & Support

- [Discussions](https://github.com/N2FlowJS/nbase/discussions)
- [Issues](https://github.com/N2FlowJS/nbase/issues)
- [Pull Requests](https://github.com/N2FlowJS/nbase/pulls)

## Links

- [GitHub Repository](https://github.com/N2FlowJS/nbase)
- [NPM Package](https://www.npmjs.com/package/@n2flowjs/nbase)

## Quick Start

```javascript
const { Database } = require('@n2flowjs/nbase');

// Initialize the database
const db = new Database({
  vectorSize: 1536,  // OpenAI's text-embedding-ada-002 size
  indexing: {
    buildOnStart: true
  }
});

// Add vectors
await db.addVector('doc1', [0.1, 0.2, ...], { title: 'Document 1' });
await db.addVector('doc2', [0.3, 0.4, ...], { title: 'Document 2' });

// Search for similar vectors
const results = await db.search([0.15, 0.25, ...], {
  k: 5,
  includeMetadata: true,
  useHNSW: true
});

console.log(results);
// [
//   { id: 'doc1', dist: 0.12, metadata: { title: 'Document 1' } },
//   { id: 'doc2', dist: 0.45, metadata: { title: 'Document 2' } },
//   ...
// ]
```

## API Documentation

### Database

The main interface for interacting with NBase.

```typescript
const db = new Database(options);
```

#### Options

- `vectorSize`: Default size of vectors (default: 1536)
- `clustering`: Options for vector clustering
- `partitioning`: Options for database partitioning
- `indexing`: Options for index creation (HNSW, LSH)
- `persistence`: Options for saving/loading the database
- `monitoring`: Options for performance monitoring

#### Methods

- `addVector(id, vector, metadata?)`: Add a vector to the database
- `bulkAdd(vectors)`: Add multiple vectors in one operation
- `findNearest(query, k, options)`: Find k nearest neighbors
- `search(query, options)`: Alias for findNearest
- `deleteVector(id)`: Delete a vector
- `getVector(id)`: Retrieve a vector
- `getMetadata(id)`: Retrieve metadata for a vector
- `updateMetadata(id, data)`: Update metadata for a vector
- `extractRelationships(threshold, options)`: Find relationships between vectors within partitions
- `buildIndexes()`: Build search indexes
- `save()`: Save the database to disk
- `close()`: Close the database and release resources

### Search Options

```typescript
const results = await db.search(queryVector, {
  k: 10,                   // Number of results to return
  filter: (id) => true,    // Function to filter results
  includeMetadata: true,   // Include metadata in results
  distanceMetric: 'cosine', // Distance metric to use
  useHNSW: true,           // Use HNSW index for search
  rerank: false,           // Rerank results for diversity
  rerankingMethod: 'diversity', // Method for reranking
  partitionIds: ['p1', 'p2'], // Specific partitions to search
  efSearch: 100,           // HNSW search parameter
});
```

## Performance Optimization

For best performance:

1. **Choose the right index**: HNSW provides the best search performance for most use cases
2. **Adjust efSearch**: Higher values improve recall at the cost of speed
3. **Use partitioning**: For large datasets, enable partitioning to reduce memory usage
4. **Filter wisely**: Complex filters may slow down search
5. **Dimension reduction**: Consider reducing vector dimensions if possible

## REST API

NBase includes a built-in HTTP server:

```javascript
const { Server } = require('@n2flowjs/nbase');
const server = new Server({ port: 1307 });
server.start();
```

### Endpoints

- `POST /vectors`: Add a vector
- `GET /vectors/:id`: Get a vector
- `DELETE /vectors/:id`: Delete a vector
- `POST /search`: Search for similar vectors
- `GET /health`: Check server health
- `POST /search/metadata`: Search with metadata filtering
- `POST /search/relationships`: Extract relationships between vectors
- `POST /search/communities`: Finds communities (clusters) of vectors based on a distance threshold across loaded partitions.

## Advanced Usage

For more advanced usage examples, check the [examples directory](https://github.com/N2FlowJS/nbase/tree/main/examples) in the repository.

## Performance Benchmarks
Comprehensive benchmark results and analysis can be found in the [Benchmarks Guide](https://github.com/N2FlowJS/nbase/blob/main/docs/benchmarks.md).

Key performance highlights:
- HNSW Search: Up to 5.86x faster than standard search
- Bulk Operations: 320x more efficient than single operations
- Scale tested: 50,000+ vectors across multiple partitions

Benchmark scenarios:
| Operation Type              | Time (ms) | Speedup Factor |
|----------------------------|----------:|---------------:|
| Standard Search            |     37.01 |          1.00x |
| HNSW Search               |     39.12 |          0.95x |
| HNSW Search (After Reload)|      4.24 |          8.73x |

## Contributing

Contributions are welcome! Please feel free to submit a [Pull Request](https://github.com/N2FlowJS/nbase/pulls) or open an [issue](https://github.com/N2FlowJS/nbase/issues).

## License

This project is licensed under the MIT License - see the LICENSE file for details.

