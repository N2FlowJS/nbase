# NBase Migration Guide

## Overview

This guide covers migrating between different versions of NBase, upgrading from other vector databases, and transitioning between different database implementations within NBase.

## Version Migration

### Migrating from NBase 0.x to 1.x

#### Breaking API Changes

##### Database Initialization
```typescript
// Before (0.x)
import { VectorDB } from '@n2flowjs/nbase';

const db = new VectorDB({
  dimensions: 1536,
  useIndex: true
});

// After (1.x)
import { Database } from '@n2flowjs/nbase';

const db = new Database({
  vectorSize: 1536,
  indexing: {
    buildOnStart: true,
    indexType: 'hnsw'
  }
});
```

##### Search API Changes
```typescript
// Before (0.x)
const results = await db.search(queryVector, 10);

// After (1.x)
const results = await db.search(queryVector, {
  k: 10,
  useHNSW: true,
  distanceMetric: 'cosine'
});
```

##### Configuration Changes
```typescript
// Before (0.x)
const config = {
  dimensions: 1536,
  useIndex: true,
  indexType: 'hnsw'
};

// After (1.x)
const config = {
  vectorSize: 1536,
  indexing: {
    buildOnStart: true,
    indexType: 'hnsw',
    hnswOptions: {
      M: 16,
      efConstruction: 200
    }
  },
  compression: {
    enabled: false
  }
};
```

#### Data Migration Process

##### Step 1: Export Data from 0.x
```typescript
import { VectorDB } from '@n2flowjs/nbase-0.x';

async function exportOldData() {
  const oldDb = new VectorDB({
    dimensions: 1536
  });

  // Load your existing data
  await oldDb.load('path/to/old/data');

  // Export in migration format
  const exportData = await oldDb.export({
    includeMetadata: true,
    includeIndices: false,  // Indices will be rebuilt
    format: 'migration'
  });

  // Save to file
  await fs.writeFile('migration-data.json', JSON.stringify(exportData));
}
```

##### Step 2: Import Data to 1.x
```typescript
import { Database } from '@n2flowjs/nbase';
import { readFileSync } from 'fs';

async function importNewData() {
  // Load migration data
  const migrationData = JSON.parse(readFileSync('migration-data.json'));

  // Create new database
  const newDb = new Database({
    vectorSize: 1536,
    partitionCapacity: 50000,
    indexing: {
      buildOnStart: false  // We'll build after import
    }
  });

  // Import data in batches
  const batchSize = 1000;
  for (let i = 0; i < migrationData.vectors.length; i += batchSize) {
    const batch = migrationData.vectors.slice(i, i + batchSize);
    await newDb.bulkAdd(batch.map(item => ({
      id: item.id,
      vector: new Float32Array(item.vector),
      metadata: item.metadata
    })));
  }

  // Build indices after import
  await newDb.buildIndexes({
    indexType: 'hnsw',
    progressCallback: (progress) => {
      console.log(`Building index: ${progress}%`);
    }
  });

  // Save the migrated database
  await newDb.save();
}
```

##### Step 3: Update Client Code
```typescript
// Update search calls
// Old
const results = await db.search(queryVector, 10);

// New
const results = await db.search(queryVector, {
  k: 10,
  useHNSW: true,
  includeMetadata: true
});

// Update result handling
results.forEach(result => {
  console.log(`ID: ${result.id}, Score: ${result.score}`);
  if (result.metadata) {
    console.log(`Metadata:`, result.metadata);
  }
});
```

### Migrating from Other Vector Databases

#### From Pinecone
```typescript
import { PineconeClient } from '@pinecone-database/pinecone';
import { Database } from '@n2flowjs/nbase';

async function migrateFromPinecone() {
  // Connect to Pinecone
  const pinecone = new PineconeClient();
  const index = pinecone.Index('your-index');

  // Create NBase database
  const nbaseDb = new Database({
    vectorSize: 1536,  // Match your Pinecone dimension
    partitionCapacity: 100000
  });

  // Fetch all vectors from Pinecone
  const pineconeData = await index.query({
    vector: new Array(1536).fill(0),  // Dummy query
    topK: 100000,  // Adjust based on your data size
    includeMetadata: true,
    includeValues: true
  });

  // Convert and import to NBase
  const vectors = pineconeData.matches.map(match => ({
    id: match.id,
    vector: new Float32Array(match.values),
    metadata: match.metadata
  }));

  await nbaseDb.bulkAdd(vectors);

  // Build HNSW index
  await nbaseDb.buildIndexes({
    indexType: 'hnsw'
  });
}
```

#### From Weaviate
```typescript
import weaviate from 'weaviate-ts-client';
import { Database } from '@n2flowjs/nbase';

async function migrateFromWeaviate() {
  // Connect to Weaviate
  const client = weaviate.client({
    scheme: 'http',
    host: 'localhost:8080'
  });

  // Create NBase database
  const nbaseDb = new Database({
    vectorSize: 1536,
    partitionCapacity: 50000
  });

  // Query all objects from Weaviate
  const result = await client.graphql
    .get()
    .withClassName('YourClass')
    .withFields(['_additional { vector }', 'property1', 'property2'])
    .withLimit(10000)
    .do();

  // Convert and import
  const vectors = result.data.Get.YourClass.map(item => ({
    id: item._additional.id,
    vector: new Float32Array(item._additional.vector),
    metadata: {
      property1: item.property1,
      property2: item.property2
    }
  }));

  await nbaseDb.bulkAdd(vectors);
}
```

#### From Qdrant
```typescript
import { QdrantClient } from '@qdrant/js-client-rest';
import { Database } from '@n2flowjs/nbase';

async function migrateFromQdrant() {
  const qdrant = new QdrantClient({ url: 'http://localhost:6333' });
  const nbaseDb = new Database({ vectorSize: 1536 });

  // Scroll through all points
  let offset = null;
  const vectors = [];

  do {
    const response = await qdrant.scroll('your-collection', {
      offset,
      limit: 1000,
      with_payload: true,
      with_vectors: true
    });

    vectors.push(...response.points.map(point => ({
      id: point.id.toString(),
      vector: new Float32Array(point.vector),
      metadata: point.payload
    })));

    offset = response.next_page_offset;
  } while (offset);

  // Import to NBase
  await nbaseDb.bulkAdd(vectors);
}
```

## Implementation Migration

### Migrating from VectorDB to ClusteredVectorDB

```typescript
import { VectorDB, ClusteredVectorDB } from '@n2flowjs/nbase';

async function migrateToClustered() {
  // Export from VectorDB
  const oldDb = new VectorDB({ vectorSize: 128 });
  const allData = await oldDb.exportAll();

  // Create ClusteredVectorDB
  const newDb = new ClusteredVectorDB({
    vectorSize: 128,
    clusterSize: 100,
    maxClusters: 20
  });

  // Import data (clustering happens automatically)
  for (const [id, data] of allData) {
    await newDb.addVector(id, data.vector, data.metadata);
  }

  // The database will automatically create clusters
  console.log(`Created ${newDb.getClusterCount()} clusters`);
}
```

### Migrating to PartitionedVectorDB

```typescript
import { ClusteredVectorDB, PartitionedVectorDB } from '@n2flowjs/nbase';

async function migrateToPartitioned() {
  const oldDb = new ClusteredVectorDB({ vectorSize: 128 });
  const newDb = new PartitionedVectorDB({
    vectorSize: 128,
    partitionCapacity: 10000,
    maxActivePartitions: 3
  });

  // Export all data
  const allVectors = await oldDb.exportAllVectors();

  // Import in batches
  const batchSize = 5000;
  for (let i = 0; i < allVectors.length; i += batchSize) {
    const batch = allVectors.slice(i, i + batchSize);
    await newDb.bulkAdd(batch);
  }

  // Build HNSW indexes
  await newDb.buildIndexes({
    indexType: 'hnsw',
    partitions: 'all'
  });
}
```

## Data Format Migration

### Converting Between Distance Metrics

```typescript
// Convert vectors from one metric to another
function convertDistanceMetric(vectors: Float32Array[], fromMetric: string, toMetric: string) {
  const conversions = {
    'euclidean->cosine': (vec: Float32Array) => {
      const norm = Math.sqrt(vec.reduce((sum, val) => sum + val * val, 0));
      return new Float32Array(vec.map(val => val / norm));
    },
    'cosine->euclidean': (vec: Float32Array) => {
      // Cosine vectors are already normalized
      return vec;
    }
  };

  const key = `${fromMetric}->${toMetric}`;
  const converter = conversions[key];

  if (!converter) {
    throw new Error(`Unsupported conversion: ${key}`);
  }

  return vectors.map(converter);
}
```

### Metadata Schema Migration

```typescript
// Migrate metadata schema
function migrateMetadataSchema(oldMetadata: any): any {
  return {
    // Rename fields
    title: oldMetadata.name,
    category: oldMetadata.type,

    // Transform values
    tags: Array.isArray(oldMetadata.tags)
      ? oldMetadata.tags
      : oldMetadata.tags?.split(',') || [],

    // Add new fields with defaults
    version: '1.0',
    migrated_at: new Date().toISOString(),

    // Preserve unknown fields
    ...Object.fromEntries(
      Object.entries(oldMetadata).filter(([key]) =>
        !['name', 'type', 'tags'].includes(key)
      )
    )
  };
}
```

## Performance Migration

### Optimizing After Migration

```typescript
async function optimizeAfterMigration(db: Database) {
  // Rebuild indexes with optimal settings
  await db.buildIndexes({
    indexType: 'hnsw',
    hnswOptions: {
      M: 32,              // More connections for better accuracy
      efConstruction: 400 // Higher build quality
    }
  });

  // Enable compression if needed
  await db.enableCompression({
    algorithm: 'product_quantization',
    compressionRatio: 0.6
  });

  // Warm up the cache
  const stats = await db.getStats();
  console.log('Migration complete. Database stats:', stats);
}
```

### Benchmarking Migration Results

```typescript
async function benchmarkMigration(oldDb: any, newDb: Database) {
  const testQueries = generateTestQueries(100);

  console.log('Benchmarking migration results...');

  // Benchmark old database
  const oldResults = await benchmarkQueries(oldDb, testQueries);

  // Benchmark new database
  const newResults = await benchmarkQueries(newDb, testQueries);

  // Compare results
  console.log('Performance comparison:');
  console.log(`Old DB: ${oldResults.avgTime}ms avg`);
  console.log(`New DB: ${newResults.avgTime}ms avg`);
  console.log(`Improvement: ${((oldResults.avgTime - newResults.avgTime) / oldResults.avgTime * 100).toFixed(1)}%`);
}
```

## Rollback Strategy

### Creating Migration Backups

```typescript
async function createMigrationBackup(db: Database) {
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
  const backupPath = `./backups/migration-${timestamp}`;

  // Create backup directory
  await fs.mkdir(backupPath, { recursive: true });

  // Export configuration
  const config = await db.exportConfig();
  await fs.writeFile(`${backupPath}/config.json`, JSON.stringify(config));

  // Export data
  const data = await db.export({
    includeIndices: true,
    compress: true
  });
  await fs.writeFile(`${backupPath}/data.json`, JSON.stringify(data));

  // Create rollback script
  const rollbackScript = `
    const { Database } = require('@n2flowjs/nbase');
    const fs = require('fs');

    async function rollback() {
      const config = JSON.parse(fs.readFileSync('./config.json'));
      const data = JSON.parse(fs.readFileSync('./data.json'));

      const db = new Database(config);
      await db.import(data);
      await db.save();

      console.log('Rollback complete');
    }

    rollback();
  `;

  await fs.writeFile(`${backupPath}/rollback.js`, rollbackScript);

  return backupPath;
}
```

## Best Practices

### Pre-Migration Checklist
- [ ] Create full backup of existing data
- [ ] Test migration on subset of data first
- [ ] Verify hardware requirements for new version
- [ ] Update client applications to new API
- [ ] Plan downtime for production migration
- [ ] Prepare rollback strategy

### Migration Execution
- [ ] Run migration in staging environment first
- [ ] Use batch processing for large datasets
- [ ] Monitor memory usage during migration
- [ ] Validate data integrity after migration
- [ ] Rebuild indexes with optimal settings
- [ ] Update monitoring and alerting

### Post-Migration Validation
- [ ] Compare search results between old and new versions
- [ ] Benchmark performance improvements
- [ ] Verify all client applications work correctly
- [ ] Update documentation and runbooks
- [ ] Train team on new features and APIs

## Troubleshooting

### Common Migration Issues

#### Memory Issues During Import
```typescript
// Use streaming import for large datasets
const stream = fs.createReadStream('large-dataset.json');
const rl = readline.createInterface({ input: stream });

let batch = [];
for await (const line of rl) {
  const item = JSON.parse(line);
  batch.push(item);

  if (batch.length >= 1000) {
    await db.bulkAdd(batch);
    batch = [];
  }
}
```

#### Index Building Failures
```typescript
// Handle index building errors gracefully
try {
  await db.buildIndexes({
    indexType: 'hnsw',
    onError: (error, partitionId) => {
      console.error(`Index build failed for ${partitionId}:`, error);
      // Continue with other partitions
    }
  });
} catch (error) {
  console.error('Index building failed:', error);
  // Fallback to flat search
}
```

#### Data Validation Errors
```typescript
// Validate data before import
function validateVectorData(data: any[]): boolean {
  return data.every(item => {
    return item.id &&
           item.vector &&
           item.vector.length > 0 &&
           item.vector.every((v: number) => !isNaN(v));
  });
}

// Use validation during import
if (!validateVectorData(importData)) {
  throw new Error('Invalid data format detected');
}
```

This comprehensive migration guide ensures smooth transitions between NBase versions and from other vector databases while maintaining data integrity and performance.
