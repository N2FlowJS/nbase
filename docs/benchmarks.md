# NBase Benchmarking Guide

## Overview

NBase includes comprehensive benchmarking suites to measure performance across different operations and scenarios. The benchmarks are located in `test/benchmarks/` and cover two main test suites.

## Benchmark Suites

### Suite 1: Large-Scale Operations

Tests database performance with large datasets and multiple partitions.

**Configuration:**
- Total Vectors: 50,000
- Vector Dimension: 128
- Partition Capacity: 10,000
- Max Active Partitions: 3
- Bulk Add Chunk Size: 5,000

**Key Metrics:**
```
| Operation                    | Time (ms) | Average (ms/op) |
|-----------------------------|-----------:|----------------:|
| DB Initialization          |      20.09 |           20.09 |
| Bulk Add (50k vectors)     |   67408.81 |         6740.88 |
| Standard Search            |      50.49 |           50.49 |
| HNSW Index Build          |  253428.89 |        84476.30 |
| HNSW Search               |      17.63 |           17.63 |
| DB Save                   |    1088.92 |         1088.92 |
| DB Load                   |     501.52 |          501.52 |
| HNSW Search (After Load)  |       8.62 |            8.62 |
```

### Suite 2: Operation Latency

Focuses on individual operation performance and response times.

**Configuration:**
- Vector Size: 128
- Partition Capacity: 10,000
- Max Active Partitions: 3
- Test Iterations: 100
- Vector Count: 5,000

**Key Metrics:**
```
| Operation        | Average Time (ms) | Total Time (ms) |
|-----------------|------------------:|----------------:|
| addVector       |              1.12 |          111.57 |
| bulkAdd         |             17.39 |         1738.55 |
| findNearest     |              1.37 |          137.34 |
| findNearestHNSW |              0.97 |           97.05 |
```

## Performance Analysis

### Search Performance

1. **Standard vs HNSW Search**
   - HNSW search is 1.42x faster than standard search in typical scenarios
   - Performance gap increases with dataset size
   - HNSW shows dramatic improvement after database reload (5.86x faster)

2. **Memory Impact**
   - Active partitions affect search speed
   - Optimal performance with 3-5 active partitions
   - Memory usage scales linearly with vector count

### Write Performance

1. **Single vs Bulk Operations**
   - Bulk operations are more efficient per vector
   - Average bulk add: 0.0035ms per vector
   - Single add: 1.12ms per vector

2. **Partition Impact**
   - New partition creation adds overhead
   - Automatic partition management balances performance

## Running Benchmarks

Execute the benchmark suites using:

```bash
npm run benchmark
```

Or run individual suites:

```bash
npm run benchmark:suite1
npm run benchmark:suite2
```

## Benchmark Configuration

Customize benchmark parameters in the test files:

```typescript
// Suite 1 Configuration
const NUM_VECTORS = 50000;
const VECTOR_DIMENSION = 128;
const PARTITION_CAPACITY = 10000;
const MAX_ACTIVE_PARTITIONS = 3;
const BULK_ADD_CHUNK_SIZE = 5000;

// Suite 2 Configuration
const vectorSize = 128;
const partitionCapacity = 10000;
const maxActivePartitions = 3;
const numVectors = 5000;
const numIterations = 100;
```

## Performance Optimization Tips

1. **HNSW Index Usage**
   - Build HNSW indices for frequently accessed partitions
   - Consider memory trade-off vs search speed improvement
   - Recommended for datasets > 10,000 vectors

2. **Partition Management**
   - Keep active partitions count <= available CPU cores
   - Consider SSD speed for partition loading/unloading
   - Monitor partition stats via `db.getStats()`

3. **Bulk Operations**
   - Use bulkAdd for large datasets
   - Optimal chunk size: 5,000-10,000 vectors
   - Monitor memory usage during bulk operations

4. **Memory Optimization**
   - Enable vector compression for large datasets
   - Use appropriate partition sizes
   - Monitor system metrics during operations

## System Requirements

Recommended specifications for optimal performance:

- CPU: 4+ cores
- RAM: 16GB+
- Storage: SSD with 1GB+ free space
- Node.js: v16+

## Custom Benchmarking

Create custom benchmarks using the provided utilities:

```typescript
const { createTimer } = require('@n2flowjs/nbase');

const timer = createTimer();
timer.start('operation');
// ... perform operation
const result = timer.stop('operation');
console.log(`Operation took ${result.total}ms`);
```
