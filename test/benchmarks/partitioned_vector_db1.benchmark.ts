/**
 * # PartitionedVectorDB Benchmark Suite 1
 *
 * This benchmark focuses on the performance of `PartitionedVectorDB` under
 * significant load, specifically testing:
 * - Initialization time.
 * - Bulk adding a large number of vectors (`bulkAdd`) and observing partition creation/loading.
 * - Standard vector search performance (`findNearest`) across multiple partitions.
 * - HNSW index build time (`buildIndexHNSW`) for loaded partitions.
 * - HNSW search performance (`findNearestHNSW`).
 * - Database save (`save`) and close (`close`) operations.
 * - Database re-loading and subsequent search performance.
 */
import { promises as fs } from 'fs';
import path from 'path';
import { PartitionedVectorDB } from '../../src/vector/partitioned_vector_db'; // Adjust path as needed
import { VectorData } from '../../src/types'; // Adjust path as needed

// --- Benchmark Parameters ---
const BENCHMARK_DIR = path.join(__dirname, 'benchmark_db_data');
const NUM_VECTORS = 50000; // Total vectors to add
const VECTOR_DIMENSION = 128;
const PARTITION_CAPACITY = 10000;
const MAX_ACTIVE_PARTITIONS = 3;
const SEARCH_K = 10;
const BULK_ADD_CHUNK_SIZE = 5000; // How many vectors to add in each bulkAdd call

// --- Helper Functions ---

/** Generates a random vector */
function generateRandomVector(dim: number): number[] {
  const vec: number[] = [];
  for (let i = 0; i < dim; i++) {
    vec.push(Math.random() * 2 - 1); // Range [-1, 1]
  }
  return vec;
}

/** Generates vector data for bulk add */
function generateVectorData(count: number, dim: number): VectorData[] {
  const data: VectorData[] = [];
  for (let i = 0; i < count; i++) {
    data.push({
      id: `vec_${i}`,
      vector: generateRandomVector(dim),
      metadata: { index: i, timestamp: Date.now() },
    });
  }
  return data;
}

/** Cleans up the benchmark directory */
async function cleanupBenchmarkDir(): Promise<void> {
  console.log(`\nCleaning up benchmark directory: ${BENCHMARK_DIR}`);
  try {
    await fs.rm(BENCHMARK_DIR, { recursive: true, force: true });
    console.log('Cleanup complete.');
  } catch (error: any) {
    console.error('Error during cleanup:', error.message);
  }
}

// Timing utility to accurately measure and store benchmark results
const benchmarkResults: Record<string, number> = {};
function startTimer(label: string): void {
  console.time(label);
  benchmarkTimes[label] = process.hrtime.bigint();
}

function endTimer(label: string): number {
  console.timeEnd(label);
  const startTime = benchmarkTimes[label];
  const endTime = process.hrtime.bigint();
  const durationMs = Number(endTime - startTime) / 1000000;
  benchmarkResults[label] = durationMs;
  return durationMs;
}

const benchmarkTimes: Record<string, bigint> = {};

/** Main benchmark function */
async function runBenchmarks() {
  await cleanupBenchmarkDir(); // Start clean

  const startTime = Date.now();

  console.log('--- Starting PartitionedVectorDB Benchmarks ---');
  console.log(`Parameters:
    - Total Vectors: ${NUM_VECTORS}
    - Dimension: ${VECTOR_DIMENSION}
    - Partition Capacity: ${PARTITION_CAPACITY}
    - Max Active Partitions: ${MAX_ACTIVE_PARTITIONS}
    - Benchmark Directory: ${BENCHMARK_DIR}\n`);

  // --- Initialization ---
  startTimer('DB Initialization');
  const db = new PartitionedVectorDB({
    partitionsDir: BENCHMARK_DIR,
    partitionCapacity: PARTITION_CAPACITY,
    maxActivePartitions: MAX_ACTIVE_PARTITIONS,
    autoCreatePartitions: true,
    vectorSize: VECTOR_DIMENSION,
    autoLoadHNSW: false, // Don't load HNSW initially
  });
  await db.initializationPromise;
  endTimer('DB Initialization');
  console.log(`Initial partitions: ${db.getPartitionConfigs().length}`);

  // --- Bulk Add ---
  console.log(`\n--- Benchmarking Bulk Add (${NUM_VECTORS} vectors in chunks of ${BULK_ADD_CHUNK_SIZE}) ---`);
  startTimer('Total Bulk Add');
  let totalAdded = 0;
  for (let i = 0; i < NUM_VECTORS; i += BULK_ADD_CHUNK_SIZE) {
    const batchNum = i / BULK_ADD_CHUNK_SIZE + 1;
    const count = Math.min(BULK_ADD_CHUNK_SIZE, NUM_VECTORS - i);
    const vectorsToAdd = generateVectorData(count, VECTOR_DIMENSION).map((v, idx) => ({
      ...v,
      id: `vec_${i + idx}`, // Ensure unique IDs across batches
    })); // Generate fresh data for each batch

    const batchLabel = `Bulk Add Batch ${batchNum} (${count} vectors)`;
    startTimer(batchLabel);
    const result = await db.bulkAdd(vectorsToAdd);
    endTimer(batchLabel);
    totalAdded += result.count;
    console.log(` -> Added ${result.count} vectors. Partitions used: [${result.partitionIds.join(', ')}]`);
    const stats = await db.getStats();
    console.log(` -> DB Stats: ${stats.partitions.loadedCount} loaded, ${stats.vectors.totalConfigured} total vectors, Active: ${stats.partitions.activeId}`);
  }
  endTimer('Total Bulk Add');
  console.log(`Finished Bulk Add. Total vectors added: ${totalAdded}`);
  const finalStats = await db.getStats();
  console.log(`Final partition count: ${finalStats.partitions.totalConfigured}`);

  // --- Standard Search ---
  console.log(`\n--- Benchmarking Standard FindNearest (k=${SEARCH_K}) ---`);
  const queryVector = generateRandomVector(VECTOR_DIMENSION);
  startTimer('Standard FindNearest');
  const standardResults = await db.findNearest(queryVector, SEARCH_K);
  endTimer('Standard FindNearest');
  console.log(`Found ${standardResults.length} results (Standard). First result dist: ${standardResults[0]?.dist}`);

  // --- HNSW Build ---
  console.log('\n--- Benchmarking HNSW Index Build (All Loaded Partitions) ---');
  startTimer('Total HNSW Build');
  // Ensure all partitions needed are loaded (bulk add might have evicted some)
  // For simplicity, we'll just build on currently loaded ones. A real scenario might load all.
  const loadedPartitionIds = finalStats.partitions.loadedIds;
  console.log(`Building HNSW for loaded partitions: [${loadedPartitionIds.join(', ')}]`);
  await db.buildIndexHNSW(); // Build for all loaded partitions
  endTimer('Total HNSW Build');
  const statsAfterBuild = await db.getStats();
  console.log(`HNSW indices loaded: ${statsAfterBuild.indices.hnswLoadedCount}`);

  // --- HNSW Search ---
  console.log(`\n--- Benchmarking HNSW FindNearest (k=${SEARCH_K}) ---`);
  startTimer('HNSW FindNearest');
  const hnswResults = await db.findNearestHNSW(queryVector, SEARCH_K);
  endTimer('HNSW FindNearest');
  console.log(`Found ${hnswResults.length} results (HNSW). First result dist: ${hnswResults[0]?.dist}`);

  // --- Save ---
  console.log('\n--- Benchmarking Save ---');
  startTimer('DB Save');
  await db.save();
  endTimer('DB Save');

  // --- Close ---
  console.log('\n--- Benchmarking Close ---');
  startTimer('DB Close');
  await db.close();
  endTimer('DB Close');

  // --- Re-Load (Optional) ---
  console.log('\n--- Benchmarking Re-Load ---');
  startTimer('DB Re-Load');
  const db2 = new PartitionedVectorDB({
    partitionsDir: BENCHMARK_DIR,
    partitionCapacity: PARTITION_CAPACITY,
    maxActivePartitions: MAX_ACTIVE_PARTITIONS,
    autoCreatePartitions: false, // Don't create new ones on load
    vectorSize: VECTOR_DIMENSION,
    autoLoadHNSW: true, // Try loading HNSW indices on load
  });
  await db2.initializationPromise;
  endTimer('DB Re-Load');
  const loadedStats = await db2.getStats();
  console.log(`Loaded DB Stats: ${loadedStats.partitions.totalConfigured} partitions configured, ${loadedStats.partitions.loadedCount} loaded, ${loadedStats.indices.hnswLoadedCount} HNSW loaded.`);

  // --- Final Search After Re-Load (Optional) ---
  console.log(`\n--- Benchmarking HNSW FindNearest After Re-Load (k=${SEARCH_K}) ---`);
  startTimer('HNSW FindNearest After Re-Load');
  const hnswResultsAfterLoad = await db2.findNearestHNSW(queryVector, SEARCH_K);
  endTimer('HNSW FindNearest After Re-Load');
  console.log(`Found ${hnswResultsAfterLoad.length} results (HNSW after load). First result dist: ${hnswResultsAfterLoad[0]?.dist}`);

  await db2.close(); // Close the second instance

  console.log('\n--- Benchmarks Complete ---');

  // Generate markdown results
  const totalTime = Date.now() - startTime;
  await generateMarkdownReport(benchmarkResults, finalStats, totalTime);
}

// Generate markdown report
async function generateMarkdownReport(
  results: Record<string, number>,
  stats: any,
  totalTime: number
): Promise<void> {
  // Extract version from package.json
  const packageJsonPath = path.join(__dirname, '../../package.json');
  const packageJson = JSON.parse(await fs.readFile(packageJsonPath, 'utf-8'));
  const version = packageJson.version;

  const reportPath = path.join(BENCHMARK_DIR, `../benchmark_results_suite1_v${version}.md`);
  const timestamp = new Date().toISOString();

  // Calculate average times
  const averageTimes: Record<string, number> = {
    'DB Initialization': results['DB Initialization'] / 1, // One-time operation
    'Total Bulk Add': results['Total Bulk Add'] / Math.ceil(NUM_VECTORS / BULK_ADD_CHUNK_SIZE), // Average per batch
    'Standard FindNearest': results['Standard FindNearest'] / 1, // Single operation
    'Total HNSW Build': results['Total HNSW Build'] / (stats.partitions.loadedCount || 1), // Average per partition
    'HNSW FindNearest': results['HNSW FindNearest'] / 1, // Single operation
    'DB Save': results['DB Save'] / 1, // One-time operation
    'DB Close': results['DB Close'] / 1, // One-time operation
    'DB Re-Load': results['DB Re-Load'] / 1, // One-time operation
    'HNSW FindNearest After Re-Load': results['HNSW FindNearest After Re-Load'] / 1, // Single operation
  };

  let markdown = `# PartitionedVectorDB Benchmark Results - Suite 1 - v${version}\n\n`;
  markdown += `*Run at: ${timestamp}*\n\n`;

  markdown += `## Configuration\n`;
  markdown += `- Total Vectors: ${NUM_VECTORS}\n`;
  markdown += `- Vector Dimension: ${VECTOR_DIMENSION}\n`;
  markdown += `- Partition Capacity: ${PARTITION_CAPACITY}\n`;
  markdown += `- Max Active Partitions: ${MAX_ACTIVE_PARTITIONS}\n`;
  markdown += `- Search K: ${SEARCH_K}\n`;
  markdown += `- Bulk Add Chunk Size: ${BULK_ADD_CHUNK_SIZE}\n\n`;

  markdown += `## Results\n\n`;
  markdown += `| Operation | Total Time (ms) | Average Time (ms) |\n`;
  markdown += `|-----------|----------------:|------------------:|\n`;

  Object.entries(results).forEach(([operation, time]) => {
    const avgTime = averageTimes[operation] || time;
    markdown += `| ${operation} | ${time.toFixed(2)} | ${avgTime.toFixed(2)} |\n`;
  });

  // Add search performance summary
  markdown += `\n## Search Performance Summary\n\n`;
  markdown += `### Standard vs HNSW Search Comparison\n\n`;
  markdown += `| Search Method | Time (ms) | Speedup Factor |\n`;
  markdown += `|---------------|----------:|---------------:|\n`;

  const standardTime = results['Standard FindNearest'] || 0;
  const hnswTime = results['HNSW FindNearest'] || 0;
  const speedupFactor = standardTime > 0 ? standardTime / hnswTime : 0;

  markdown += `| Standard Search | ${standardTime.toFixed(2)} | 1.00x |\n`;
  markdown += `| HNSW Search | ${hnswTime.toFixed(2)} | ${speedupFactor.toFixed(2)}x |\n`;

  if (results['HNSW FindNearest After Re-Load']) {
    const hnswReloadTime = results['HNSW FindNearest After Re-Load'];
    const reloadSpeedupFactor = standardTime > 0 ? standardTime / hnswReloadTime : 0;
    markdown += `| HNSW Search (After Reload) | ${hnswReloadTime.toFixed(2)} | ${reloadSpeedupFactor.toFixed(2)}x |\n`;
  }

  markdown += `\n**Note**: ${speedupFactor > 1 ? 'HNSW search is faster' : 'Standard search is faster'} by a factor of ${Math.abs(speedupFactor - 1).toFixed(2)}x.\n`;

  markdown += `\n## Database Stats\n\n`;
  markdown += `- Total partitions: ${stats.partitions.totalConfigured}\n`;
  markdown += `- Loaded partitions: ${stats.partitions.loadedCount}\n`;
  markdown += `- Total vectors: ${stats.vectors.totalConfigured}\n`;
  markdown += `- HNSW indices: ${stats.indices.hnswLoadedCount}\n\n`;

  markdown += `## Summary\n\n`;
  markdown += `Total benchmark execution time: ${(totalTime / 1000).toFixed(2)} seconds\n`;

  try {
    await fs.writeFile(reportPath, markdown);
    console.log(`Benchmark results written to: ${reportPath}`);
  } catch (error: any) {
    console.error(`Error writing benchmark results: ${error.message}`);
  }
}

// Run the benchmarks and handle cleanup
runBenchmarks()
  .catch((err) => {
    console.error('\n--- Benchmark Failed ---');
    console.error(err);
    process.exitCode = 1;
  })
  .finally(async () => {
    // Optional: Keep the data for inspection
    // await cleanupBenchmarkDir();
    console.log('Benchmark script finished.');
  });