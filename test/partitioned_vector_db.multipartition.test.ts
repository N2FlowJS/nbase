import { expect } from "chai";
import { describe, it, before, beforeEach, after, afterEach } from "mocha";
import { PartitionedVectorDB } from "../src/vector/partitioned_vector_db";
import {
  TEST_DIR,
  PARTITIONS_DIR,
  generateRandomVector,
  createTestVectors,
  setupTestDirectory,
  cleanupPartitionsDir,
  cleanupTestDirectory,
} from "./test-helpers/vector-db-test-utils";

describe("PartitionedVectorDB - Multiple Partitions", () => {
  const VECTOR_SIZE = 10;
  let db: PartitionedVectorDB;

  before(async () => {
    await setupTestDirectory();
  });

  beforeEach(async () => {
    db = new PartitionedVectorDB({
      partitionsDir: PARTITIONS_DIR,
      vectorSize: VECTOR_SIZE,
      partitionCapacity: 50,
      maxActivePartitions: 3,
      autoCreatePartitions: true,
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

  it("should automatically create new partitions when needed", async () => {
    // Create first partition
    await db.createPartition("auto-part-1", "Auto Partition 1", {
      setActive: true,
    });

    // Add vectors that exceed partition capacity, forcing auto-creation
    const vectors = createTestVectors(100, VECTOR_SIZE);
    const result = await db.bulkAdd(vectors);

    // Should have more than one partition
    expect(result.partitionIds.length).to.be.greaterThan(1);

    const configs = db.getPartitionConfigs();
    expect(configs.length).to.be.greaterThan(1);

    // Verify vectors are spread across partitions
    const stats = await db.getStats();
    expect(stats.vectors.totalConfigured).to.equal(100);
  });

  it("should search across multiple partitions", async () => {
    // Create two partitions with different vectors
    await db.createPartition("search-part-1", "Search Part 1", {
      setActive: true,
    });
    const vectors1 = createTestVectors(30, VECTOR_SIZE);
    await db.bulkAdd(vectors1);

    await db.createPartition("search-part-2", "Search Part 2", {
      setActive: true,
    });
    const vectors2 = createTestVectors(30, VECTOR_SIZE);
    await db.bulkAdd(vectors2);

    // Search across all partitions
    const queryVector = generateRandomVector(VECTOR_SIZE);
    const results = await db.findNearest(queryVector, 10);
    expect(results.length).to.be.greaterThan(0);

    // Search with specific partition IDs
    const specificResults = await db.findNearest(queryVector, 5, {
      partitionIds: ["search-part-1"],
    });
    console.log(`[TEST] Specific Results: ${JSON.stringify(specificResults)}`);

    expect(specificResults.length).to.greaterThan(0);
  });

  it("should handle LRU eviction correctly", async () => {
    // Create more partitions than the maxActivePartitions
    for (let i = 0; i < 5; i++) {
      await db.createPartition(`lru-part-${i}`, `LRU Part ${i}`, {
        setActive: true,
      });
      const vectors = createTestVectors(10, VECTOR_SIZE);
      await db.bulkAdd(vectors);
    }

    // Get stats to check loaded partitions
    const stats = await db.getStats();

    // Should not exceed maxActivePartitions
    expect(stats.partitions.loadedCount).to.be.at.most(3);

    // Access an older partition to bring it into LRU cache
    await db.getPartition("lru-part-0");

    // This partition should now be in the loaded partitions
    const newStats = await db.getStats();
    expect(newStats.partitions.loadedIds).to.include("lru-part-0");
  });
});
