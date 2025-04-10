import { expect } from 'chai';
import { describe, it, before, beforeEach, after, afterEach } from 'mocha';
import { PartitionedVectorDB } from '../src/vector/partitioned_vector_db';
import { 
    TEST_DIR, 
    PARTITIONS_DIR, 
    generateRandomVector,
    createTestVectors,
    setupTestDirectory, 
    cleanupPartitionsDir, 
    cleanupTestDirectory 
} from './test-helpers/vector-db-test-utils';

describe('PartitionedVectorDB - Search Operations', () => {
    const VECTOR_SIZE = 10;
    const TEST_VECTORS_COUNT = 100;
    let db: PartitionedVectorDB;

    before(async () => {
        await setupTestDirectory();
    });
    
    beforeEach(async () => {
        db = new PartitionedVectorDB({
            partitionsDir: PARTITIONS_DIR,
            vectorSize: VECTOR_SIZE,
            partitionCapacity: 500
        });
        await db.initializationPromise;
        await db.createPartition('test-partition', 'Test Partition', { setActive: true });
        
        // Add test vectors
        const vectors = createTestVectors(TEST_VECTORS_COUNT, VECTOR_SIZE);
        await db.bulkAdd(vectors);
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

    it('should find nearest vectors using standard search', async () => {
        const queryVector = generateRandomVector(VECTOR_SIZE);
        const results = await db.findNearest(queryVector, 5);
        
        expect(results).to.be.an('array');
        expect(results.length).to.be.greaterThan(0);
        expect(results[0]).to.have.property('dist');
        expect(results[0]).to.have.property('id');
    });

    it('should build HNSW index and perform approximate search', async () => {
        // Build HNSW index
        await db.buildIndexHNSW('test-partition', { dimensionAware: true });
        
        const queryVector = generateRandomVector(VECTOR_SIZE);
        const results = await db.findNearestHNSW(queryVector, 5);
        
        expect(results).to.be.an('array');
        expect(results.length).to.equal(5);
        
        // Check HNSW stats
        const hnswStats = db.getHNSWStats('test-partition');
        
        expect(hnswStats).to.not.be.null;
        expect(hnswStats?.totalNodes).to.greaterThan(0);
    });
});
