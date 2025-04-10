import { expect } from 'chai';
import { describe, it, before, beforeEach, after, afterEach } from 'mocha';
import { PartitionedVectorDB } from '../src/vector/partitioned_vector_db';
import { 
    TEST_DIR, 
    PARTITIONS_DIR, 
    setupTestDirectory, 
    cleanupPartitionsDir, 
    cleanupTestDirectory 
} from './test-helpers/vector-db-test-utils';

describe('PartitionedVectorDB - Initialization', () => {
    let db: PartitionedVectorDB;

    before(async () => {
        await setupTestDirectory();
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

    it('should initialize with default options', async () => {
        db = new PartitionedVectorDB({
            partitionsDir: PARTITIONS_DIR,
            vectorSize: 10
        });
        
        await db.initializationPromise;
        expect(db.IsReady()).to.be.true;
        
        const stats = await db.getStats();
        
        expect(stats.status).to.equal('initialized');
        
        expect(stats.partitions.totalConfigured).to.equal(1);
    });

    it('should initialize with custom options', async () => {
        db = new PartitionedVectorDB({
            partitionsDir: PARTITIONS_DIR,
            vectorSize: 10,
            partitionCapacity: 500,
            maxActivePartitions: 2,
            autoCreatePartitions: true,
            useCompression: true
        });
        
        await db.initializationPromise;
        const stats = await db.getStats();
        
        expect(stats.settings.partitionCapacity).to.equal(500);
        expect(stats.settings.maxActivePartitions).to.equal(2);
        expect(stats.settings.autoCreatePartitions).to.be.true;
        expect(stats.settings.useCompression).to.be.true;
    });
});
