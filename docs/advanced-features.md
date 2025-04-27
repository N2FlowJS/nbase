# Advanced Features

## Vector Relationships

Extract relationships between vectors based on distance:

```javascript
const relationships = await db.extractRelationships(0.3, {
  metric: 'cosine',
  includeMetadata: true
});
```

## Community Detection

Find clusters of related vectors:

```javascript
const communities = await db.extractCommunities(0.5, {
  metric: 'euclidean',
  includeMetadata: true
});
```

## Custom Search Methods

### Hybrid Search

```javascript
const results = await db.search(queryVector, {
  k: 10,
  method: 'hybrid',
  weights: {
    text: 0.7,
    semantic: 0.3
  }
});
```

### Batch Search

```javascript
const queries = [
  { vector: [...], k: 5 },
  { vector: [...], k: 10 }
];

const results = await db.batchSearch(queries, {
  maxBatchSize: 100,
  maxWorkers: 4
});
```

## Advanced Filtering

```javascript
const filter = [
  { field: 'category', operator: '$in', value: ['article', 'blog'] },
  { field: 'rating', operator: '$gte', value: 4.5 },
  { field: 'tags', operator: '$regex', value: '^tech.*' }
];

const results = await db.search(queryVector, {
  k: 10,
  filters: filter
});
```

## Index Management

### Manual Index Building

```javascript
await db.buildIndexes({
  method: 'hnsw',
  efConstruction: 200,
  M: 16
});
```

### Index Statistics

```javascript
const stats = await db.getStats();
console.log(stats.indices);
```

## Monitoring & Performance

### Enable Monitoring

```javascript
const db = new Database({
  monitoring: {
    enable: true,
    interval: 60000,
    logToConsole: true
  }
});
```

### Performance Metrics

```javascript
db.on('metrics', (snapshot) => {
  console.log('Search QPS:', snapshot.metrics.search.queriesPerMinute);
  console.log('Cache Hit Rate:', snapshot.metrics.cache.hitRate);
});
```

## Data Persistence

### Auto-save Configuration

```javascript
const db = new Database({
  persistence: {
    saveIntervalMs: 300000, // Save every 5 minutes
    dbPath: './database',
    useCompression: true
  }
});
```

### Manual Save/Load

```javascript
// Save current state
await db.save();

// Load from saved state
await db.load('./database/backup.db');
```

## Event Handling

```javascript
// Monitor vector additions
db.on('vector:add', (data) => {
  console.log(`Added vector ${data.id} with ${data.dimensions} dimensions`);
});

// Monitor search operations
db.on('search:complete', (data) => {
  console.log(`Search completed in ${data.duration}ms`);
});
```

## Best Practices

1. Use bulk operations for better performance
2. Enable monitoring in production
3. Configure auto-save for data safety
4. Use appropriate partition sizes
5. Optimize index parameters for your use case
