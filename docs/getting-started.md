# Getting Started with NBase

## Installation

```bash
npm i @n2flowjs/nbase
```

## Basic Usage

```javascript
const { Database } = require('@n2flowjs/nbase');

// Initialize the database with default settings
const db = new Database({
  vectorSize: 1536,  // Default size for OpenAI embeddings
  indexing: {
    buildOnStart: true
  }
});

// Add vectors with metadata
await db.addVector('doc1', [0.1, 0.2, ...], {
  title: 'Document 1',
  category: 'article'
});

// Search for similar vectors
const results = await db.search([0.15, 0.25, ...], {
  k: 5,
  includeMetadata: true
});
```

## Configuration Options

### Basic Configuration
```javascript
const options = {
  vectorSize: 1536,            // Default vector dimensions
  cacheSize: 1000,            // Size of LRU cache for search results
  maxConcurrentSearches: 4,   // Max parallel searches
}
```

### Indexing Options
```javascript
const options = {
  indexing: {
    buildOnStart: true,       // Build index when DB starts
    autoRebuildThreshold: 500,// Rebuild after N additions
    hnsw: {
      M: 16,                 // Max connections per node
      efConstruction: 200,   // Index build quality
      efSearch: 100         // Search quality
    }
  }
}
```

### Partitioning Options
```javascript 
const options = {
  partitioning: {
    partitionsDir: './database/partitions',
    partitionCapacity: 100000,
    autoLoadPartitions: true,
    maxActivePartitions: 3
  }
}
```

## Basic Operations

### Adding Vectors

```javascript
// Add single vector
await db.addVector('doc1', vector, metadata);

// Bulk add vectors
await db.bulkAdd([
  { id: 'doc1', vector: [...], metadata: {...} },
  { id: 'doc2', vector: [...], metadata: {...} }
]);
```

### Searching

```javascript
// Basic search
const results = await db.search(queryVector, { k: 5 });

// Advanced search with filters
const results = await db.search(queryVector, {
  k: 10,
  filter: (id, metadata) => metadata.category === 'article',
  includeMetadata: true,
  useHNSW: true,
  efSearch: 100
});
```

### Managing Metadata

```javascript
// Update metadata
await db.updateMetadata('doc1', {
  title: 'Updated Title',
  tags: ['tag1', 'tag2']
});

// Get metadata
const metadata = await db.getMetadata('doc1');
```

## Next Steps

- Check out [Advanced Features](./advanced-features.md) for more complex usage
- See [Optimization Guide](./optimization.md) for performance tuning
- Review [API Reference](./api-reference.md) for detailed documentation
