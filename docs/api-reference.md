# API Reference

## Database Class

### Constructor

```typescript
constructor(options: DatabaseOptions)
```

#### DatabaseOptions
```typescript
interface DatabaseOptions {
  vectorSize?: number;
  cacheSize?: number;
  maxConcurrentSearches?: number;
  indexing: IndexingConfiguration;
  partitioning: PartitioningConfiguration;
  persistence: PersistenceOptions;
  monitoring?: MonitoringConfiguration;
}
```

### Core Methods

#### addVector
```typescript
async addVector(
  id: string | number,
  vector: number[] | Float32Array,
  metadata?: Record<string, any>
): Promise<void>
```

#### bulkAdd
```typescript
async bulkAdd(
  vectors: Array<{
    id: string | number;
    vector: number[] | Float32Array;
    metadata?: Record<string, any>;
  }>
): Promise<BulkAddResult>
```

#### search / findNearest
```typescript
async search(
  query: number[] | Float32Array,
  options?: SearchOptions
): Promise<SearchResult[]>
```

#### getMetadata
```typescript
async getMetadata(
  id: string | number
): Promise<Record<string, any> | null>
```

### Index Management

#### buildIndexes
```typescript
async buildIndexes(
  options?: BuildIndexOptions
): Promise<void>
```

#### saveIndex
```typescript
async saveIndex(
  filepath: string
): Promise<void>
```

### Database Management

#### save
```typescript
async save(): Promise<void>
```

#### close
```typescript
async close(): Promise<void>
```

### Advanced Methods

#### extractRelationships
```typescript
async extractRelationships(
  threshold: number,
  options?: {
    metric?: 'cosine' | 'euclidean';
    includeMetadata?: boolean;
  }
): Promise<Array<{
  vector1: { id: string | number; metadata?: any };
  vector2: { id: string | number; metadata?: any };
  distance: number;
}>>
```

#### getStats
```typescript
async getStats(): Promise<DatabaseStats>
```

### Events

The Database class extends EventEmitter and emits the following events:

```typescript
interface DatabaseEvents {
  'vector:add': { id: string | number; dimensions: number };
  'vectors:bulkAdd': { count: number; ids: Array<string | number> };
  'search:complete': { duration: number; resultCount: number };
  'search:error': { error: Error; duration: number };
  'index:build': { method: string; duration: number };
  'save:complete': { type: 'config' | 'indices' | 'all' };
  'error': { message: string; context: string; error: Error };
}
```

### Type Definitions

#### SearchOptions
```typescript
interface SearchOptions {
  k: number;
  includeMetadata?: boolean;
  includeVectors?: boolean;
  filter?: (id: string | number, metadata?: Record<string, any>) => boolean;
  filters?: FilterConfig[];
  useHNSW?: boolean;
  efSearch?: number;
  distanceMetric?: 'cosine' | 'euclidean';
  maxDistance?: number;
  partitionIds?: string[];
}
```

#### SearchResult
```typescript
interface SearchResult {
  id: string | number;
  score: number;
  distance: number;
  metadata?: Record<string, any>;
  vector?: number[] | Float32Array;
}
```

#### DatabaseStats
```typescript
interface DatabaseStats {
  vectorCount: number;
  partitionCount: number;
  memoryUsage: number;
  indices: {
    hnsw?: HNSWStats;
    lsh?: LSHStats;
  };
  search: {
    totalQueries: number;
    averageResponseTime: number;
    queriesPerSecond: number;
  };
  cache: {
    size: number;
    hitRate: number;
  };
}
```
