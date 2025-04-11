# NBase - Neural Vector Database

NBase is a high-performance vector database for efficient similarity search, designed for machine learning embeddings and neural search applications.

## Features

- **Scalable Vector Storage**: Store and manage millions of high-dimensional vectors
- **Optimized Search Algorithms**: Fast approximate nearest neighbor search
  - HNSW (Hierarchical Navigable Small World) graphs for logarithmic search time
  - LSH (Locality-Sensitive Hashing) for fast similarity search
  - Partitioned search for large-scale databases
- **Multi-dimensional Support**: Handles vectors of different dimensions
- **Vector Compression**: Reduces memory usage while maintaining search quality
- **Rich Query Options**: Filter, rerank, and customize search parameters
- **Persistence**: Save and load your vector database to/from disk
- **REST API**: Simple HTTP interface for adding vectors and searching

## Installation

```bash
npm i @n2flowjs/nbase
```

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

## Advanced Usage

For more advanced usage examples, check the examples directory in the repository.

## Performance Benchmarks
Benchmarks comparing NBase with other vector databases can be found in the `test/benchmarks` directory.
---
| v0.1.3                      | Time (ms) | Speedup Factor  |
|-----------------------------|----------:|----------------:|
| Standard Search             | 37.01     | 1.00x           |
| HNSW Search                 | 39.12     | 0.95x           |
| HNSW Search (After Reload)  | 4.24      | 8.73x           | 
---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

