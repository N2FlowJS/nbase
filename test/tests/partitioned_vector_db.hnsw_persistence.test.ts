import { expect } from "chai";
import { describe, it } from "mocha";
import { PartitionedVectorDB } from "../../src";
import {
  PARTITIONS_DIR,
  createTestVectors,
} from "../../test/test-helpers/vector-db-test-utils";
import { setupPersistenceTests, VECTOR_SIZE } from "./partitioned_vector_db.common_setup";

describe("PartitionedVectorDB - HNSW Persistence", () => {
  const { getDB } = setupPersistenceTests();

  it("should save and load HNSW indices", async () => {
    const db = getDB();
    
    await db.createPartition("hnsw-test", "HNSW Test", { setActive: true });
    const vectors = createTestVectors(50, VECTOR_SIZE);
    await db.bulkAdd(vectors);

    // Build HNSW index
    await db.buildIndexHNSW("hnsw-test");

    // Save HNSW indices
    await db.saveHNSWIndices("hnsw-test");

    // Close and reopen
    await db.close();

    // Create new instance with autoLoadHNSW
    const newDB = new PartitionedVectorDB({
      partitionsDir: PARTITIONS_DIR,
      vectorSize: VECTOR_SIZE,
      autoLoadHNSW: true,
    });
    await newDB.initializationPromise;

    // Load partition and HNSW index
    await newDB.getPartition("hnsw-test");
    await newDB.loadHNSWIndices("hnsw-test");

    // Verify HNSW is loaded by checking stats
    const stats = newDB.getHNSWStats("hnsw-test");
    expect(stats).to.not.be.null;
    
    // Clean up
    await newDB.close();
  });
});
