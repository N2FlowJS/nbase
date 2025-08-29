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

> 🚀 **NBase** is a high-performance, scalable vector database designed for efficient similarity search and neural search applications. Built with TypeScript, it provides enterprise-grade storage for high-dimensional vectors with advanced indexing algorithms and distributed architecture.

## ✨ Key Features

### 🌟 Core Capabilities
- **📦 Enterprise-Grade Storage**: Store and manage millions of high-dimensional vectors with production-ready reliability
- **🔄 Auto-Scaling**: Dynamic partitioning system that automatically creates new partitions as data grows
- **💾 Persistent Storage**: Automatic saving and loading of database state, partitions, and indexes
- **🔧 Flexible Configuration**: Comprehensive configuration system for clustering, indexing, partitioning, and performance tuning

### 🔍 Advanced Search Technologies
- **🎯 State-of-the-Art Algorithms**:
  - **🕸️ HNSW (Hierarchical Navigable Small World)**: Graph-based indexing providing logarithmic search complexity with optimized memory usage
  - **🎲 LSH (Locality-Sensitive Hashing)**: Ultra-fast approximate similarity search with configurable hash functions
  - **📊 KNN (K-Nearest Neighbors)**: Exact nearest neighbor search with multi-threading and caching support
  - **🔄 Hybrid Search**: Intelligent combination of multiple search methods for optimal performance
  - **📦 Batch Search**: Efficient processing of multiple search queries with parallel execution

### 💪 Technical Excellence
- **📐 Multi-Dimensional Support**: Handles vectors of any dimension with dynamic dimension management
- **🗜️ Intelligent Compression**: Advanced vector compression using Product Quantization and K-means clustering
- **📊 Smart Clustering**: K-means clustering for organizing vectors and reducing search space
- **🔄 Real-time Indexing**: Background index building and maintenance with progress tracking
- **📈 Performance Monitoring**: Comprehensive metrics collection and system monitoring
- **🔒 Production Ready**: Built-in security, rate limiting, and error handling

## Installation

```bash
npm install @n2flowjs/nbase
```

## Quick Start

### Basic Usage

```javascript
const { Database } = require('@n2flowjs/nbase');

// Initialize database with configuration
const db = new Database({
  vectorSize: 1536,  // Dimension size (e.g., OpenAI's text-embedding-ada-002)
  partitioning: {
    partitionsDir: './data/partitions',
    partitionCapacity: 10000,  // Vectors per partition
    autoCreatePartitions: true
  },
  indexing: {
    buildOnStart: true,
    hnswOptions: {
      M: 16,  // HNSW parameter: connections per node
      efConstruction: 200  // HNSW parameter: construction candidate list size
    }
  },
  clustering: {
    clusterSize: 1000,  // Target cluster size
    distanceMetric: 'cosine'
  },
  cacheSize: 1000,  // Search result cache size
  monitoring: {
    enable: true,
    enableSystemMetrics: true,
    enableSearchMetrics: true
  }
});

// Wait for database to be ready
await db.ready();

// Add vectors with metadata
await db.addVector('doc1', [0.1, 0.2, ...], {
  title: 'Document 1',
  category: 'science',
  timestamp: Date.now()
});

await db.addVector('doc2', [0.3, 0.4, ...], {
  title: 'Document 2',
  category: 'technology',
  timestamp: Date.now()
});

// Search for similar vectors
const results = await db.search([0.15, 0.25, ...], {
  k: 5,  // Number of results
  includeMetadata: true,
  distanceMetric: 'cosine',
  useHNSW: true,  // Use HNSW index for faster search
  rerank: true,   // Apply reranking for diversity
  filter: (id, metadata) => metadata.category === 'science'  // Filter results
});

console.log('Search results:', results);
// Output:
// [
//   {
//     id: 'doc1',
//     dist: 0.12,
//     metadata: { title: 'Document 1', category: 'science', timestamp: 1234567890 }
//   },
//   ...
// ]
```

### REST API Server

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

// Server starts automatically and provides REST endpoints
console.log('NBase server running on http://localhost:1307');
```

## Architecture Overview

NBase follows a layered architecture designed for scalability and performance:

### Core Components

1. **Database**: High-level interface providing unified API for all operations
2. **PartitionedVectorDB**: Manages multiple partitions for horizontal scaling
3. **ClusteredVectorDB**: Handles vector clustering within each partition
4. **VectorDB**: Core vector storage and basic operations
5. **UnifiedSearch**: Orchestrates search across different algorithms and partitions

### Data Flow

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   REST API      │────│   Database       │────│ PartitionedDB   │
│   (Express)     │    │   (High-level)   │    │ (Scaling)       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   UnifiedSearch │────│  ClusteredDB     │────│   VectorDB      │
│   (Algorithms)  │    │   (Clustering)   │    │   (Storage)     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Configuration Options

### Database Configuration

```typescript
interface DatabaseOptions {
  // Vector dimensions
  vectorSize?: number;

  // Partitioning settings
  partitioning: {
    partitionsDir: string;           // Directory for partition storage
    partitionCapacity: number;       // Max vectors per partition
    autoCreatePartitions: boolean;   // Auto-create new partitions
    maxActivePartitions: number;     // Max partitions in memory
    autoLoadPartitions: boolean;     // Load partitions on startup
  };

  // Indexing configuration
  indexing: {
    buildOnStart: boolean;           // Build indexes on startup
    autoSave: boolean;               // Auto-save indexes
    hnswOptions?: {
      M: number;                     // HNSW connections per node
      efConstruction: number;        // Construction candidate list size
      efSearch: number;              // Search candidate list size
    };
    lshOptions?: {
      numberOfHashes: number;        // LSH hash functions
      numberOfBuckets: number;       // LSH hash buckets
    };
  };

  // Clustering settings
  clustering: {
    clusterSize: number;             // Target cluster size
    distanceMetric: 'euclidean' | 'cosine';
    useCompression: boolean;         // Enable vector compression
    kmeansMaxIterations: number;     // K-means iterations
  };

  // Performance settings
  cacheSize: number;                 // Search cache size
  maxConcurrentSearches: number;     // Concurrent search limit

  // Persistence
  persistence: {
    dbPath: string;                  // Database storage path
    autoSave: boolean;               // Auto-save database
    saveIntervalMs: number;          // Save interval
  };

  // Monitoring
  monitoring: {
    enable: boolean;                 // Enable monitoring
    intervalMs: number;              // Metrics collection interval
    enableSystemMetrics: boolean;    // System metrics
    enableSearchMetrics: boolean;    // Search metrics
    enableDatabaseMetrics: boolean;  // Database metrics
  };
}
```

## API Reference

### Database Methods

#### Vector Operations

```typescript
// Add single vector
await db.addVector(id: string | number, vector: number[], metadata?: object): Promise<void>

// Add multiple vectors
await db.bulkAdd(vectors: VectorData[]): Promise<{ count: number }>

// Delete vector
await db.deleteVector(id: string | number): Promise<boolean>

// Get vector
await db.getVector(id: string | number): Promise<{ vector: number[], metadata?: object } | null>

// Update metadata
await db.updateMetadata(id: string | number, metadata: object): Promise<void>
```

#### Search Operations

```typescript
// Basic similarity search
await db.search(query: number[], options?: SearchOptions): Promise<SearchResult[]>

// Advanced search with options
await db.findNearest(query: number[], k?: number, options?: UnifiedSearchOptions): Promise<SearchResult[]>

// Batch search
await db.batchSearch(queries: BatchQuery[], options?: BatchSearchOptions): Promise<SearchResult[][]>
```

#### Search Options

```typescript
interface SearchOptions {
  k?: number;                    // Number of results (default: 10)
  includeMetadata?: boolean;     // Include metadata in results
  includeVectors?: boolean;      // Include vectors in results
  distanceMetric?: 'euclidean' | 'cosine';  // Distance metric
  filter?: (id: string | number, metadata?: object) => boolean;  // Result filter
  useHNSW?: boolean;             // Use HNSW index
  efSearch?: number;             // HNSW search parameter
  rerank?: boolean;              // Apply reranking
  rerankingMethod?: 'diversity' | 'standard';  // Reranking method
  partitionIds?: string[];       // Limit to specific partitions
  skipCache?: boolean;           // Skip result cache
}
```

#### Management Operations

```typescript
// Build search indexes
await db.buildIndexes(): Promise<void>

// Get database statistics
await db.getStats(): Promise<DatabaseStats>

// Save database state
await db.save(): Promise<void>

// Close database
await db.close(): Promise<void>
```

### REST API Endpoints

The REST API provides HTTP access to all database operations:

#### Vectors
- `POST /vectors` - Add a vector
- `GET /vectors/:id` - Get a vector
- `DELETE /vectors/:id` - Delete a vector
- `PUT /vectors/:id/metadata` - Update metadata

#### Search
- `POST /search` - Similarity search
- `POST /search/batch` - Batch search
- `POST /search/metadata` - Search with metadata filtering
- `POST /search/relationships` - Find vector relationships
- `POST /search/communities` - Find vector communities

#### Database Management
- `GET /health` - Health check
- `GET /stats` - Database statistics
- `POST /indexes/build` - Build search indexes
- `POST /save` - Save database state

#### Example API Usage

```bash
# Add a vector
curl -X POST http://localhost:1307/vectors \
  -H "Content-Type: application/json" \
  -d '{
    "id": "doc1",
    "vector": [0.1, 0.2, 0.3, ...],
    "metadata": { "title": "Example Document" }
  }'

# Search for similar vectors
curl -X POST http://localhost:1307/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": [0.15, 0.25, 0.35, ...],
    "k": 5,
    "includeMetadata": true,
    "distanceMetric": "cosine"
  }'
```

## Advanced Features

### Partitioning Strategy

NBase uses intelligent partitioning to handle large datasets:

- **Automatic Partition Creation**: New partitions are created when active partition reaches capacity
- **LRU Cache**: Recently used partitions stay in memory, older ones are unloaded
- **Parallel Processing**: Search queries can be executed across multiple partitions simultaneously
- **Load Balancing**: Vectors are distributed evenly across partitions

### Indexing System

Multiple indexing algorithms for different use cases:

- **HNSW**: Best for accuracy and moderate-speed searches
- **LSH**: Best for ultra-fast approximate searches
- **Flat**: Exact search for small datasets or high-accuracy requirements

### Clustering and Compression

- **K-means Clustering**: Groups similar vectors to reduce search space
- **Product Quantization**: Compresses vectors to save memory and improve search speed
- **Dynamic Compression**: Automatically adjusts compression based on data characteristics

### Monitoring and Profiling

Comprehensive monitoring system:

```typescript
// Enable monitoring
const db = new Database({
  monitoring: {
    enable: true,
    intervalMs: 5000,  // Collect metrics every 5 seconds
    enableSystemMetrics: true,
    enableSearchMetrics: true,
    enableDatabaseMetrics: true
  }
});

// Get metrics
const metrics = await db.getStats();
console.log('Database metrics:', metrics);
```

## Performance Optimization

### Index Selection Guide

| Use Case | Recommended Index | Configuration |
|----------|------------------|---------------|
| High Accuracy | HNSW | `M: 16-32, efConstruction: 200-500` |
| Fast Search | LSH | `numberOfHashes: 10-20` |
| Exact Search | Flat | No additional config needed |
| Balanced | Hybrid | Combine HNSW + LSH |

### Memory Management

- **Partition LRU Cache**: Controls memory usage by limiting active partitions
- **Search Result Cache**: Caches frequent search results
- **Vector Compression**: Reduces memory footprint
- **Background Cleanup**: Automatic cleanup of unused resources

### Search Performance Tuning

```typescript
// Optimize for speed
const results = await db.search(query, {
  k: 10,
  useHNSW: true,
  efSearch: 100,  // Lower for faster search, higher for better accuracy
  skipCache: false  // Use cache for repeated queries
});

// Optimize for accuracy
const results = await db.search(query, {
  k: 10,
  useHNSW: true,
  efSearch: 400,  // Higher efSearch for better recall
  rerank: true,   // Apply reranking for diversity
  distanceMetric: 'cosine'
});
```

## Benchmarks

Recent benchmark results show excellent performance:

| Operation | Dataset Size | Time | Notes |
|-----------|-------------|------|-------|
| Single Search (HNSW) | 100K vectors | ~5ms | 1536 dimensions |
| Bulk Add | 10K vectors | ~200ms | With indexing |
| Batch Search | 100 queries | ~150ms | Parallel execution |
| Index Build | 50K vectors | ~3s | HNSW construction |

*Benchmarks performed on Intel i7-9700K, 32GB RAM, SSD storage*

## Development

### Building from Source

```bash
# Clone repository
git clone https://github.com/N2FlowJS/nbase.git
cd nbase

# Install dependencies
npm install

# Build project
npm run build

# Run tests
npm test

# Start development server
npm run dev
```

### Project Structure

```
nbase/
├── src/
│   ├── database/          # High-level Database class
│   ├── vector/            # Vector database implementations
│   │   ├── vector_db.ts          # Basic vector DB
│   │   ├── clustered_vector_db.ts # With clustering
│   │   └── partitioned_vector_db.ts # With partitioning
│   ├── ann/               # Approximate nearest neighbor algorithms
│   │   ├── hnsw.ts        # HNSW implementation
│   │   └── lsh.ts         # LSH implementation
│   ├── search/            # Search engines
│   │   ├── unified_search.ts     # Unified search interface
│   │   ├── knn_search.ts         # KNN search
│   │   └── hybrid_search.ts      # Hybrid search
│   ├── compression/       # Vector compression
│   │   ├── product_quantization.ts
│   │   └── kmeans.ts
│   ├── server/            # REST API server
│   ├── utils/             # Utilities
│   └── types.ts           # TypeScript definitions
├── test/                  # Test suites
├── docs/                  # Documentation
└── package.json
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

### Code Style

- Use TypeScript for all new code
- Follow existing code style and patterns
- Add JSDoc comments for public APIs
- Include unit tests for new features
- Update documentation for API changes

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- 📖 [Documentation](https://github.com/N2FlowJS/nbase/tree/main/docs)
- 🐛 [Issues](https://github.com/N2FlowJS/nbase/issues)
- 💬 [Discussions](https://github.com/N2FlowJS/nbase/discussions)
- 📧 [Email](mailto:support@n2flowjs.com)

## Roadmap

### Upcoming Features

- [ ] **Distributed Deployment**: Multi-node clustering support
- [ ] **Advanced Compression**: More compression algorithms
- [ ] **Query Optimization**: Automatic query planning
- [ ] **Backup & Recovery**: Enhanced backup strategies
- [ ] **GraphQL API**: Alternative API interface
- [ ] **Plugin System**: Extensible architecture

### Version History

- **v0.1.9** (Current): Performance improvements, monitoring enhancements
- **v0.1.8**: REST API stabilization, clustering improvements
- **v0.1.7**: Hybrid search implementation, batch operations
- **v0.1.6**: HNSW optimization, compression features
- **v0.1.5**: Partitioning system, persistence layer
- **v0.1.0**: Initial release with core functionality

---

**Made with ❤️ by the N2FlowJS Team**
