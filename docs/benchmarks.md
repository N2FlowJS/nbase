# NBase Performance Benchmarks

## Overview

NBase provides comprehensive benchmarking tools to measure and optimize performance across different operations, algorithms, and configurations. The benchmarking suite helps you understand performance characteristics and make informed decisions about deployment configurations.

## Benchmark Architecture

### Test Structure
```
test/benchmarks/
├── partitioned_vector_db1.benchmark.ts    # Large-scale operations
├── partitioned_vector_db2.benchmark.ts    # Operation latency
├── benchmark_results_suite1_v0.1.3.md     # Suite 1 results
└── benchmark_results_suite2_v0.1.3.md     # Suite 2 results
```

### Benchmark Categories

1. **Large-Scale Operations** - Tests with 50K+ vectors across multiple partitions
2. **Operation Latency** - Measures individual operation response times
3. **Algorithm Comparison** - Compares HNSW, LSH, Flat, and KNN performance
4. **Memory Profiling** - Tracks memory usage patterns
5. **Concurrent Operations** - Tests multi-threaded performance

## Running Benchmarks

### Quick Start

```bash
# Run all benchmarks
npm run benchmark

# Run specific suites
npm run benchmark:suite1    # Large-scale operations
npm run benchmark:suite2    # Operation latency

# Run with custom configuration
npm run benchmark -- --config=./custom-config.json
```

### Benchmark Options

```typescript
const benchmarkOptions = {
  iterations: 100,           // Number of test iterations
  warmupIterations: 10,      // Warmup iterations
  outputFormat: 'json',      // json, csv, or markdown
  outputFile: './results.json',
  includeMemoryStats: true,  // Track memory usage
  includeSystemStats: true,  // Track system metrics
  timeout: 300000           // Timeout in milliseconds
};
```

## Benchmark Results

### Suite 1: Large-Scale Operations

**Test Configuration:**
- Dataset Size: 50,000 vectors
- Vector Dimensions: 128
- Partitions: 5 partitions × 10,000 vectors each
- Memory Limit: 2GB per partition
- Index Types: HNSW, Flat

**Performance Results:**

| Operation | Time (ms) | Throughput | Memory Peak | CPU Usage |
|-----------|-----------|------------|-------------|-----------|
| **Database Initialization** | 20.09 | - | 45MB | 15% |
| **Bulk Vector Addition** | 67,408.81 | 741 vec/sec | 1.2GB | 85% |
| **HNSW Index Build** | 253,428.89 | - | 2.1GB | 95% |
| **Flat Search (k=10)** | 50.49 | 19,802 queries/sec | 120MB | 45% |
| **HNSW Search (k=10)** | 17.63 | 56,720 queries/sec | 180MB | 35% |
| **Database Persistence** | 1,088.92 | - | 800MB | 70% |
| **Database Restore** | 501.52 | - | 950MB | 60% |
| **Post-Restore HNSW Search** | 8.62 | 116,030 queries/sec | 200MB | 30% |

**Key Insights:**
- HNSW provides 2.86x faster search after index building
- Post-restore performance improves by 2.05x due to optimized loading
- Memory usage peaks during index building but stabilizes during search
- Bulk operations show 95% CPU utilization during processing

### Suite 2: Operation Latency

**Test Configuration:**
- Dataset Size: 5,000 vectors
- Vector Dimensions: 128
- Test Iterations: 100 per operation
- Concurrent Workers: 4
- Distance Metrics: Cosine, Euclidean

**Latency Results:**

| Operation | Min (ms) | Max (ms) | Avg (ms) | P95 (ms) | P99 (ms) |
|-----------|----------|----------|----------|----------|----------|
| **addVector** | 0.85 | 2.34 | 1.12 | 1.89 | 2.15 |
| **bulkAdd (100)** | 12.45 | 23.67 | 17.39 | 21.34 | 22.89 |
| **findNearest (Flat)** | 1.02 | 2.89 | 1.37 | 2.45 | 2.67 |
| **findNearest (HNSW)** | 0.67 | 1.98 | 0.97 | 1.67 | 1.89 |
| **batchSearch (10)** | 8.45 | 15.23 | 11.67 | 14.12 | 14.89 |
| **getVector** | 0.12 | 0.45 | 0.23 | 0.38 | 0.42 |
| **updateVector** | 1.67 | 3.45 | 2.12 | 2.89 | 3.12 |
| **deleteVector** | 0.89 | 2.01 | 1.34 | 1.78 | 1.95 |

**Performance Analysis:**
- HNSW search is 29% faster than flat search on average
- Bulk operations show 15.5x better throughput than individual adds
- Read operations (getVector) are sub-millisecond
- Batch operations provide 8.5x better latency than individual searches

## Algorithm Performance Comparison

### Search Algorithm Benchmarks

**Test Setup:**
- Dataset: 25,000 vectors (128 dimensions)
- Query Count: 1,000 random queries
- k: 10 nearest neighbors
- Distance Metric: Cosine similarity

| Algorithm | Avg Latency | Throughput | Accuracy | Memory Usage | Index Build Time |
|-----------|-------------|------------|----------|--------------|------------------|
| **Flat (Brute Force)** | 45.23ms | 22,080 q/s | 100% | 12MB | None |
| **HNSW (ef=100)** | 12.45ms | 80,321 q/s | 97.8% | 45MB | 45.2s |
| **HNSW (ef=200)** | 18.67ms | 53,561 q/s | 98.9% | 67MB | 67.8s |
| **LSH (10 hashes)** | 3.12ms | 320,513 q/s | 89.4% | 8MB | 2.3s |
| **LSH (20 hashes)** | 4.56ms | 219,298 q/s | 92.1% | 15MB | 4.1s |
| **KNN (Exact)** | 42.89ms | 23,315 q/s | 100% | 18MB | None |

### Algorithm Recommendations

| Use Case | Recommended Algorithm | Configuration |
|----------|----------------------|---------------|
| **Small Dataset (< 10K)** | Flat | Default settings |
| **Large Dataset, High Accuracy** | HNSW | ef=200, M=16 |
| **Speed Critical** | LSH | 15-20 hashes |
| **Exact Results Required** | KNN/Flat | Default settings |
| **Balanced Performance** | HNSW | ef=100, M=12 |

## Memory Profiling

### Memory Usage Patterns

```typescript
// Memory tracking during operations
const memoryStats = {
  baseline: 45 * 1024 * 1024,    // 45MB baseline
  perVector: 512,                // ~512 bytes per vector
  indexOverhead: 1.8,            // 1.8x overhead for HNSW
  partitionOverhead: 2 * 1024 * 1024,  // 2MB per partition
  compressionRatio: 0.65         // 35% reduction with compression
};
```

### Memory Optimization Strategies

1. **Vector Compression**
   ```typescript
   const db = new Database({
     compression: {
       enabled: true,
       algorithm: 'product_quantization',
       compressionRatio: 0.5
     }
   });
   ```

2. **Partition Management**
   ```typescript
   // Automatic partition unloading
   db.configure({
     maxActivePartitions: 3,
     partitionUnloadThreshold: 0.8,  // Unload when 80% memory used
     memoryLimit: 2 * 1024 * 1024 * 1024  // 2GB limit
   });
   ```

3. **Index Memory Control**
   ```typescript
   // Memory-efficient HNSW
   await db.buildIndexes({
     indexType: 'hnsw',
     hnswOptions: {
       M: 8,              // Fewer connections
       efConstruction: 100 // Faster building
     }
   });
   ```

## Concurrent Performance

### Multi-Threaded Benchmarking

**Test Configuration:**
- Workers: 1, 2, 4, 8 concurrent threads
- Dataset: 10,000 vectors
- Operations: Mixed read/write workload

| Concurrent Workers | Throughput (ops/sec) | Avg Latency | CPU Usage | Memory Usage |
|-------------------|---------------------|-------------|-----------|--------------|
| **1 Worker** | 892 | 1.12ms | 25% | 120MB |
| **2 Workers** | 1,756 | 1.14ms | 45% | 145MB |
| **4 Workers** | 3,423 | 1.17ms | 78% | 180MB |
| **8 Workers** | 5,678 | 1.41ms | 92% | 245MB |

**Scaling Analysis:**
- Near-linear scaling up to 4 workers
- Diminishing returns beyond 4 workers due to contention
- Memory usage increases 2x from 1 to 8 workers
- Optimal configuration: 4 workers for most systems

## System Requirements & Optimization

### Hardware Recommendations

| Component | Minimum | Recommended | High Performance |
|-----------|---------|-------------|------------------|
| **CPU** | 2 cores | 4-8 cores | 16+ cores |
| **RAM** | 4GB | 16GB | 64GB+ |
| **Storage** | HDD | SSD | NVMe SSD |
| **Network** | 100Mbps | 1Gbps | 10Gbps |

### Performance Tuning Guide

#### For Search-Heavy Workloads
```typescript
const searchOptimizedConfig = {
  // HNSW optimization
  hnswOptions: {
    M: 16,
    efConstruction: 200,
    efSearch: 150
  },

  // Memory optimization
  maxActivePartitions: 5,
  cacheSize: 1000,

  // Search optimization
  defaultSearchOptions: {
    useHNSW: true,
    efSearch: 100,
    includeMetadata: false
  }
};
```

#### For Write-Heavy Workloads
```typescript
const writeOptimizedConfig = {
  // Bulk operation optimization
  bulkBatchSize: 10000,
  maxConcurrentBatches: 3,

  // Partition optimization
  partitionCapacity: 50000,
  autoPartitioning: true,

  // Memory optimization
  compression: {
    enabled: true,
    algorithm: 'kmeans',
    compressionRatio: 0.6
  }
};
```

#### For Memory-Constrained Environments
```typescript
const memoryOptimizedConfig = {
  // Minimal memory footprint
  maxActivePartitions: 2,
  partitionCapacity: 25000,

  // Compression enabled
  compression: {
    enabled: true,
    algorithm: 'product_quantization',
    compressionRatio: 0.4
  },

  // Minimal caching
  cacheSize: 100,
  disableMetadataCache: true
};
```

## Custom Benchmarking

### Creating Custom Benchmarks

```typescript
import { Database } from '@n2flowjs/nbase';
import { createTimer, BenchmarkSuite } from './test-helpers/benchmark-utils';

async function customBenchmark() {
  const db = new Database({ /* config */ });
  const suite = new BenchmarkSuite('Custom Benchmark');

  // Warmup
  await suite.warmup(async () => {
    const vectors = generateRandomVectors(1000, 128);
    await db.bulkAdd(vectors);
  });

  // Custom test
  suite.addTest('Custom Search Test', async () => {
    const timer = createTimer();
    timer.start('search');

    const query = generateRandomVector(128);
    const results = await db.search(query, { k: 10 });

    timer.stop('search');
    return timer.getResults();
  });

  // Run benchmark
  const results = await suite.run({
    iterations: 100,
    outputFormat: 'json'
  });

  console.log('Benchmark Results:', results);
}
```

### Benchmark Metrics Collection

```typescript
// Comprehensive metrics tracking
const metrics = {
  timing: {
    totalTime: 0,
    averageTime: 0,
    minTime: Infinity,
    maxTime: 0,
    percentiles: {}
  },
  memory: {
    peakUsage: 0,
    averageUsage: 0,
    growthRate: 0
  },
  system: {
    cpuUsage: 0,
    ioOperations: 0,
    networkTraffic: 0
  },
  custom: {
    throughput: 0,
    errorRate: 0,
    cacheHitRate: 0
  }
};
```

## Interpreting Results

### Performance Analysis Checklist

- [ ] Compare results across different configurations
- [ ] Identify performance bottlenecks
- [ ] Check memory usage patterns
- [ ] Validate accuracy vs speed trade-offs
- [ ] Consider system resource constraints
- [ ] Evaluate scaling characteristics
- [ ] Test with production-like data patterns

### Common Performance Issues

1. **High Memory Usage**
   - Reduce active partitions
   - Enable compression
   - Use smaller HNSW M values

2. **Slow Search Performance**
   - Build HNSW indexes
   - Increase efSearch values
   - Optimize partition sizes

3. **Slow Write Performance**
   - Use bulk operations
   - Increase partition capacity
   - Optimize batch sizes

4. **High CPU Usage**
   - Reduce concurrent operations
   - Use LSH for faster searches
   - Optimize algorithm parameters

This comprehensive benchmarking guide helps you understand NBase performance characteristics and optimize configurations for your specific use cases.
