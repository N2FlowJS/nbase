import { before, beforeEach, after, afterEach } from "mocha";
import { PartitionedVectorDB } from "../../src/vector/partitioned_vector_db";
import {
  TEST_DIR,
  PARTITIONS_DIR,
  setupTestDirectory,
  cleanupPartitionsDir,
  cleanupTestDirectory,
} from "../test-helpers/vector-db-test-utils";

export const VECTOR_SIZE = 10;

export function setupPersistenceTests() {
  let db: PartitionedVectorDB;

  before(async () => {
    await setupTestDirectory();
  });

  beforeEach(async () => {
    db = new PartitionedVectorDB({
      partitionsDir: PARTITIONS_DIR,
      vectorSize: VECTOR_SIZE,
      partitionCapacity: 1000,
    });
    await db.initializationPromise;
  });

  afterEach(async () => {
    if (db) {
      await db.close();
    }
    await cleanupPartitionsDir();
  });

  after(async () => {
    await cleanupTestDirectory();
  });

  return { getDB: () => db };
}
