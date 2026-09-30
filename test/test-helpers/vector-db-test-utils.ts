import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import { rimraf } from 'rimraf';

/**
 * Per-run temp directory for DB fixtures.
 *
 * This used to be `path.join(process.cwd(), 'test-data', ...)`, which meant
 * every suite shared one fixed directory inside the repository. That made the
 * location depend on the current working directory and hard-blocked
 * `mocha --parallel` (one suite's `before` would rimraf another's fixtures).
 */
export const TEST_DIR = fs.mkdtempSync(path.join(os.tmpdir(), 'nbase-test-'));
export const PARTITIONS_DIR = path.join(TEST_DIR, 'partitions');

// Test vectors
export const generateRandomVector = (size: number): number[] => {
  return Array.from({ length: size }, () => Math.random() - 0.5);
};

/**
 * Creates test vectors with stable IDs for testing
 */
export function createTestVectors(count: number, dimension: number): any[] {
  return Array.from({ length: count }).map((_, i) => ({
    id: `test-${i}`, // Generate stable string IDs
    vector: new Float32Array(Array.from({ length: dimension }).map(() => Math.random())),
    metadata: {
      testIndex: i,
      source: 'test-generation',
      created: new Date().toISOString(),
    },
  }));
}

// Setup and teardown helpers
export const setupTestDirectory = async () => {
  if (fs.existsSync(TEST_DIR)) {
    await rimraf(TEST_DIR);
  }
  fs.mkdirSync(TEST_DIR, { recursive: true });
};

export const cleanupPartitionsDir = async () => {
  await rimraf(PARTITIONS_DIR);
};

export const cleanupTestDirectory = async () => {
  await rimraf(TEST_DIR);
};
