import { expect } from "chai";
import { describe, it } from "mocha";
import { PartitionedVectorDB } from "../../src/vector/partitioned_vector_db";
import {
  PARTITIONS_DIR,
  createTestVectors,
} from "../test-helpers/vector-db-test-utils";
import { setupPersistenceTests, VECTOR_SIZE } from "./partitioned_vector_db.common_setup";

describe("PartitionedVectorDB - State Persistence", () => {
  const { getDB } = setupPersistenceTests();
  
  it("should save and load the database state", async () => {
    const db = getDB();
    
    // Create partition and add vectors with consistent IDs for testing
    await db.createPartition("save-test", "Save Test", { setActive: true });
    
    // Generate vectors with explicit IDs that can be retrieved later
    const vectors = Array.from({ length: 20 }).map((_, i) => ({
      id: `test-vector-${i}`, // Use consistent string ID format
      vector: new Float32Array(Array.from({ length: VECTOR_SIZE }).map(() => Math.random())),
      metadata: { testIndex: i, source: 'persistence-test' }
    }));
    
    // Add the vectors
    const result = await db.bulkAdd(vectors);
    expect(result.count).to.equal(20);
    console.log(`[TEST] Added ${result.count} vectors with explicit IDs`);
    
    const stats = await db.getStats();
    console.log(stats);
    
    console.log('[TEST] saving state...');
    console.log('====================================');
    // Save state
    await db.save();
    console.log('====================================');

    // Close and reopen
    await db.close();

    // Create new instance pointing to same directory
    const newDB = new PartitionedVectorDB({
      partitionsDir: PARTITIONS_DIR,
      vectorSize: VECTOR_SIZE,
      partitionCapacity: 1000,
    });
    await newDB.initializationPromise;

    // Verify state was loaded
    const configs = newDB.getPartitionConfigs();
    let vectorCount = 0;
    for (const config of configs) {
      vectorCount += config.vectorCount;
    }
    expect(configs.length).to.equal(2); // Initial + saved partition
    expect(configs[1].id).to.equal("save-test");
    
    // The vector count should match what we added
    expect(vectorCount).to.be.at.least(20, "Expected at least 20 vectors to be persisted");
    
    // Try to retrieve one of the vectors we added with a known ID
    const testVectorId = "test-vector-0";
    console.log(`[TEST] Attempting to retrieve vector with ID: ${testVectorId}`);
    const vector = await newDB.getVector(testVectorId);
    
    // This should not be null if persistence worked correctly
    expect(vector).to.not.be.null;
    
    // Also try to find similar vectors
    const similar = await newDB.findNearest(vectors[0].vector, 1);
    console.log("[TEST] Found similar vectors:", similar);
    expect(similar).to.have.lengthOf.at.least(1, "Expected to find at least one similar vector");
    
    // Clean up
    await newDB.close();
  });
});
