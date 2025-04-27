# Migration Guide

## Version Migration

### 0.x to 1.0

#### Breaking Changes
```typescript
// Old API
const db = new VectorDB();

// New API
const db = new Database({
  vectorSize: 1536
});
```

#### Database Configuration
```typescript
// Before 1.0
const config = {
  dimensions: 1536,
  useIndex: true
};

// After 1.0
const config = {
  vectorSize: 1536,
  indexing: {
    buildOnStart: true
  }
};
```

#### Search API Changes
```typescript
// Before 1.0
await db.search(vector, 10);

// After 1.0
await db.search(vector, {
  k: 10,
  useHNSW: true
});
```

### Data Migration

#### Export Format
```typescript
interface ExportFormat {
  version: string;
  vectors: {
    id: string;
    vector: number[];
    metadata?: Record<string, any>;
  }[];
  indices?: {
    type: string;
    data: any;
  }[];
}
```

#### Migration Scripts
```typescript
async function migrateData() {
  // Export from old version
  const oldData = await oldDb.export();
  
  // Import to new version
  const newDb = new Database(config);
  await newDb.import(oldData);
}
```

## Best Practices

### Before Migration
1. Backup existing data
2. Run test migrations
3. Verify performance

### During Migration
1. Use batch operations
2. Monitor memory usage
3. Handle errors gracefully

### After Migration
1. Verify data integrity
2. Rebuild indices
3. Update client code
