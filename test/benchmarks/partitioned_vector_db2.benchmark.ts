/**
 * # PartitionedVectorDB Benchmark Suite 2
 *
 * This benchmark measures the average time taken for core `PartitionedVectorDB`
 * operations over a fixed number of iterations. It focuses on:
 * - `addVector`: Adding single vectors repeatedly.
 * - `bulkAdd`: Adding batches of vectors repeatedly.
 * - `findNearest`: Standard nearest neighbor search.
 * - `findNearestHNSW`: HNSW-accelerated nearest neighbor search.
 *
 * Note: This benchmark uses a smaller dataset and focuses on operation latency
 * rather than throughput under heavy load or partition management aspects.
 */
import { PartitionedVectorDB } from '../../src/vector/partitioned_vector_db';
import { VectorData } from '../../src/types';
import path from 'path';
import fs from 'fs';

// Configuration
const vectorSize = 128;
const partitionCapacity = 10000;
const maxActivePartitions = 3;
const numVectors = 5000;
const numIterations = 100; // Number of iterations for each benchmark

// Helper function to generate random vectors
function generateRandomVector(size: number): number[] {
  return Array.from({ length: size }, () => Math.random());
}

// Helper function to generate random VectorData
function generateRandomVectorData(count: number, vectorSize: number): VectorData[] {
  return Array.from({ length: count }, (_, i) => ({
    id: `vec-${i}`,
    vector: generateRandomVector(vectorSize),
    metadata: { index: i },
  }));
}

// Setup and teardown
const dbPath = path.join(__dirname, '../database');
if (!fs.existsSync(dbPath)) {
  fs.mkdirSync(dbPath, { recursive: true });
}

let db: PartitionedVectorDB;

async function setup() {
  db = new PartitionedVectorDB({
    partitionsDir: dbPath,
    vectorSize: vectorSize,
    partitionCapacity: partitionCapacity,
    maxActivePartitions: maxActivePartitions,
    autoCreatePartitions: true,
  });
  await db.initializationPromise;
}

async function teardown() {
  await db.close();

  setTimeout(() => {
    fs.rmSync(dbPath, { recursive: true, force: true });
  }, 3000);
}

// Store benchmark results
interface BenchmarkResult {
  operation: string;
  averageTime: number;
  totalTime: number;
}

const benchmarkResults: BenchmarkResult[] = [];

async function benchmarkAddVector() {
  const vector = generateRandomVector(vectorSize);
  const startTime = performance.now();
  for (let i = 0; i < numIterations; i++) {
    await db.addVector(undefined, vector);
  }
  const endTime = performance.now();
  const totalTime = endTime - startTime;
  const averageTime = totalTime / numIterations;

  benchmarkResults.push({
    operation: 'addVector',
    averageTime,
    totalTime,
  });

  console.log(`addVector: ${averageTime} ms per iteration (Total: ${totalTime} ms)`);
}

async function benchmarkBulkAdd() {
  const vectors = generateRandomVectorData(100, vectorSize);
  const startTime = performance.now();
  for (let i = 0; i < numIterations; i++) {
    await db.bulkAdd(vectors);
  }
  const endTime = performance.now();
  const totalTime = endTime - startTime;
  const averageTime = totalTime / numIterations;

  benchmarkResults.push({
    operation: 'bulkAdd',
    averageTime,
    totalTime,
  });

  console.log(`bulkAdd: ${averageTime} ms per iteration (Total: ${totalTime} ms)`);
}

async function benchmarkFindNearest() {
  const queryVector = generateRandomVector(vectorSize);
  const startTime = performance.now();
  for (let i = 0; i < numIterations; i++) {
    await db.findNearest(queryVector, 10);
  }
  const endTime = performance.now();
  const totalTime = endTime - startTime;
  const averageTime = totalTime / numIterations;

  benchmarkResults.push({
    operation: 'findNearest',
    averageTime,
    totalTime,
  });

  console.log(`findNearest: ${averageTime} ms per iteration (Total: ${totalTime} ms)`);
}

async function benchmarkFindNearestHNSW() {
  const queryVector = generateRandomVector(vectorSize);
  const startTime = performance.now();
  for (let i = 0; i < numIterations; i++) {
    await db.findNearestHNSW(queryVector, 10);
  }
  const endTime = performance.now();
  const totalTime = endTime - startTime;
  const averageTime = totalTime / numIterations;

  benchmarkResults.push({
    operation: 'findNearestHNSW',
    averageTime,
    totalTime,
  });

  console.log(`findNearestHNSW: ${averageTime} ms per iteration (Total: ${totalTime} ms)`);
}

async function generateMarkdownReport() {
  // Extract version from package.json
  const packageJsonPath = path.join(__dirname, '../../package.json');
  const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, 'utf-8'));
  const version = packageJson.version;

  const reportPath = path.join(__dirname, `benchmark_results_suite2_v${version}.md`);
  const timestamp = new Date().toISOString();

  let markdown = `# PartitionedVectorDB Benchmark Results - Suite 2 - v${version}\n\n`;
  markdown += `*Run at: ${timestamp}*\n\n`;

  markdown += `## Configuration\n`;
  markdown += `- Vector Size: ${vectorSize}\n`;
  markdown += `- Partition Capacity: ${partitionCapacity}\n`;
  markdown += `- Max Active Partitions: ${maxActivePartitions}\n`;
  markdown += `- Vector Count: ${numVectors}\n`;
  markdown += `- Iterations per test: ${numIterations}\n\n`;

  markdown += `## Results\n\n`;
  markdown += `| Operation | Average Time (ms) | Total Time (ms) |\n`;
  markdown += `|-----------|------------------:|-----------------:|\n`;

  benchmarkResults.forEach((result) => {
    markdown += `| ${result.operation} | ${result.averageTime.toFixed(2)} | ${result.totalTime.toFixed(2)} |\n`;
  });

  // Add search performance summary
  markdown += `\n## Search Performance Summary\n\n`;

  // Find the search-related results
  const standardSearch = benchmarkResults.find((r) => r.operation === 'findNearest');
  const hnswSearch = benchmarkResults.find((r) => r.operation === 'findNearestHNSW');

  if (standardSearch && hnswSearch) {
    const speedupFactor = standardSearch.averageTime / hnswSearch.averageTime;
    const fasterMethod = speedupFactor > 1 ? 'HNSW' : 'Standard';
    const factor = speedupFactor > 1 ? speedupFactor : 1 / speedupFactor;

    markdown += `### Standard vs HNSW Search Comparison\n\n`;
    markdown += `| Search Method | Average Time (ms) | Relative Performance |\n`;
    markdown += `|---------------|------------------:|---------------------:|\n`;
    markdown += `| Standard Search | ${standardSearch.averageTime.toFixed(2)} | ${speedupFactor <= 1 ? 'Fastest (1.00x)' : `${speedupFactor.toFixed(2)}x slower`} |\n`;
    markdown += `| HNSW Search | ${hnswSearch.averageTime.toFixed(2)} | ${speedupFactor > 1 ? 'Fastest (1.00x)' : `${(1 / speedupFactor).toFixed(2)}x slower`} |\n\n`;

    markdown += `**Summary**: ${fasterMethod} search is approximately ${factor.toFixed(2)}x faster than ${fasterMethod === 'HNSW' ? 'Standard' : 'HNSW'} search in this benchmark.\n`;
  }

  try {
    fs.writeFileSync(reportPath, markdown);
    console.log(`Benchmark results written to: ${reportPath}`);
  } catch (error: any) {
    console.error(`Error writing benchmark results: ${error.message}`);
  }
}

async function runBenchmarks() {
  console.log('Setting up...');
  await setup();

  console.log('Running benchmarks...');
  await benchmarkAddVector();
  await benchmarkBulkAdd();
  await benchmarkFindNearest();
  await benchmarkFindNearestHNSW();

  // Generate markdown report
  await generateMarkdownReport();

  console.log('Tearing down...');
  await teardown();
}

runBenchmarks();
