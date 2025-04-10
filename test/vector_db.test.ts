import { expect } from 'chai';
import { VectorDB } from '../src';
import * as fs from 'fs';
import * as path from 'path';
import { promisify } from 'util';

const mkdtemp = promisify(fs.mkdtemp);
const rimraf = promisify(fs.rm);

describe('VectorDB', () => {
    let tempDir: string;
    
    before(async () => {
        tempDir = await mkdtemp(path.join(__dirname, 'vector_db_test_'));
    });
    
    after(async () => {
        await rimraf(tempDir, { recursive: true, force: true });
    });
    
    describe('Constructor', () => {
        it('should create a VectorDB instance with default parameters', () => {
            const db = new VectorDB();
            expect(db).to.be.instanceOf(VectorDB);
            expect(db.defaultVectorSize).to.be.null;
            expect(db.memoryStorage.size).to.equal(0);
        });
        
        it('should create a VectorDB with specified vector size', () => {
            const db = new VectorDB(128);
            expect(db.defaultVectorSize).to.equal(128);
        });
    });
    
    describe('Vector Operations', () => {
        let db: VectorDB;
        
        beforeEach(() => {
            db = new VectorDB(3);
        });
        
        it('should add a vector with auto-generated ID', () => {
            const vector = new Float32Array([0.1, 0.2, 0.3]);
            const id = db.addVector(undefined, vector);
            expect(id).to.equal(1);
            expect(db.getVector(id)).to.deep.equal(vector);
        });
        
        it('should add a vector with specified numeric ID', () => {
            const vector = new Float32Array([0.4, 0.5, 0.6]);
            const id = db.addVector(5, vector);
            expect(id).to.equal(5);
            expect(db.getVector(id)).to.deep.equal(vector);
            expect(db.getIdCounter()).to.equal(6); // Should update idCounter
        });
        
        it('should add a vector with specified string ID', () => {
            const vector = new Float32Array([0.7, 0.8, 0.9]);
            const id = db.addVector('test-id', vector);
            expect(id).to.equal('test-id');
            expect(db.getVector(id)).to.deep.equal(vector);
        });
        
        it('should overwrite existing vector with same ID', () => {
            const vector1 = new Float32Array([0.1, 0.2, 0.3]);
            const vector2 = new Float32Array([0.4, 0.5, 0.6]);
            db.addVector(1, vector1);
            db.addVector(1, vector2);
            expect(db.getVector(1)).to.deep.equal(vector2);
        });
        
        it('should check if vector exists', () => {
            db.addVector(1, [0.1, 0.2, 0.3]);
            expect(db.hasVector(1)).to.be.true;
            expect(db.hasVector(2)).to.be.false;
        });
        
        it('should delete a vector', () => {
            db.addVector(1, [0.1, 0.2, 0.3]);
            expect(db.deleteVector(1)).to.be.true;
            expect(db.hasVector(1)).to.be.false;
            expect(db.deleteVector(1)).to.be.false; // Already deleted
        });
        
        it('should get vector dimension', () => {
            db.addVector(1, [0.1, 0.2, 0.3]);
            db.addVector(2, [0.1, 0.2, 0.3, 0.4]);
            expect(db.getVectorDimension(1)).to.equal(3);
            expect(db.getVectorDimension(2)).to.equal(4);
            expect(db.getVectorDimension(3)).to.be.null;
        });
        
        it('should bulk add vectors', () => {
            const vectors = [
                { id: 1, vector: [0.1, 0.2, 0.3] },
                { id: 2, vector: [0.4, 0.5, 0.6] },
                { id: 'test', vector: [0.7, 0.8, 0.9] }
            ];
            const added = db.bulkAdd(vectors);
            expect(added).to.equal(3);
            expect(db.getVector(1)).to.deep.equal(new Float32Array([0.1, 0.2, 0.3]));
            expect(db.getVector(2)).to.deep.equal(new Float32Array([0.4, 0.5, 0.6]));
            expect(db.getVector('test')).to.deep.equal(new Float32Array([0.7, 0.8, 0.9]));
        });
    });
    
    describe('Metadata Operations', () => {
        let db: VectorDB;
        
        beforeEach(() => {
            db = new VectorDB(3);
            db.addVector(1, [0.1, 0.2, 0.3]);
        });
        
        it('should add metadata to a vector', () => {
            db.addMetadata(1, { source: 'test', tags: ['tag1', 'tag2'] });
            expect(db.getMetadata(1)).to.deep.equal({ source: 'test', tags: ['tag1', 'tag2'] });
        });
        
        it('should throw when adding metadata to non-existent vector', () => {
            expect(() => db.addMetadata(999, { source: 'test' })).to.throw();
        });
        
        it('should update metadata with object', () => {
            db.addMetadata(1, { source: 'test', count: 5 });
            const updated = db.updateMetadata(1, { count: 10, newField: 'value' });
            expect(updated).to.be.true;
            expect(db.getMetadata(1)).to.deep.equal({
                source: 'test',
                count: 10,
                newField: 'value'
            });
        });
        
        it('should update metadata with function', () => {
            db.addMetadata(1, { counter: 5 });
            const updated = db.updateMetadata(1, (current) => {
                return { counter: current ? current.counter + 1 : 1 };
            });
            expect(updated).to.be.true;
            expect(db.getMetadata(1)).to.deep.equal({ counter: 6 });
        });
        
        it('should return false when updating non-existent vector metadata', () => {
            const updated = db.updateMetadata(999, { field: 'value' });
            expect(updated).to.be.false;
        });
    });
    
    describe('Vector Search', () => {
        let db: VectorDB;
        
        beforeEach(() => {
            db = new VectorDB(3);
            db.addVector(1, [1, 0, 0], { category: 'x' });
            db.addVector(2, [0, 1, 0], { category: 'y' });
            db.addVector(3, [0, 0, 1], { category: 'z' });
            db.addVector(4, [0.7, 0.7, 0], { category: 'xy' });
        });
        
        it('should find nearest vectors using euclidean distance', () => {
            const results = db.findNearest([0.8, 0.8, 0], 2);
            expect(results.length).to.equal(2);
            expect(results[0].id).to.equal(4); // Vector 4 should be closest
            expect(results[1].id).to.be.oneOf([1, 2]); // Either vector 1 or 2
        });
        
        it('should find nearest vectors using cosine distance', () => {
            const results = db.findNearest([0.5, 0.5, 0], 2, { metric: 'cosine' });
            expect(results.length).to.equal(2);
            expect(results[0].id).to.equal(4); // Vector 4 should be closest
        });
        
        it('should apply filter when searching', () => {
            const results = db.findNearest([0.8, 0.8, 0], 3, {
                filter: (id, metadata) => metadata?.category === 'xy'
            });
            expect(results.length).to.equal(1);
            expect(results[0].id).to.equal(4);
        });
    });
    
    describe('Stats and Lifecycle', () => {
        let db: VectorDB;
        
        beforeEach(() => {
            db = new VectorDB(3);
            db.addVector(1, [1, 0, 0], { category: 'x' });
            db.addVector(2, [0, 1, 0], { category: 'y' });
            db.addVector(3, [0, 0, 1], { category: 'z' });
        });
        
        it('should return correct vector count', () => {
            expect(db.getVectorCount()).to.equal(3);
        });
        
        it('should return vector statistics', () => {
            const stats = db.getStats();
            expect(stats.vectorCount).to.equal(3);
            expect(stats.metadataCount).to.equal(3);
            expect(stats.defaultVectorSize).to.equal(3);
            expect(stats.dimensions.unique).to.equal(1);
            expect(stats.dimensions.counts).to.deep.equal({ '3': 3 });
        });
        
        it('should clear data on close', async () => {
            await db.close();
            expect(db.getVectorCount()).to.equal(0);
        });
    });
    
    describe('Persistence', () => {
        let db: VectorDB;
        let dbPath: string;
        
        beforeEach(() => {
            dbPath = path.join(tempDir, 'test_db');
            db = new VectorDB(3, dbPath);
        });
        
        afterEach(async () => {
            await db.close();
        });
        
        it('should save and load database', async () => {
            // Add vectors and metadata
            db.addVector(1, [0.1, 0.2, 0.3], { tag: 'test1' });
            db.addVector(2, [0.4, 0.5, 0.6], { tag: 'test2' });
            
            // Save to disk
            await db.save();
            
            // Create a new instance that loads from disk
            const loadedDb = new VectorDB(null, dbPath);
            
            // Wait a bit for async loading
            await new Promise(resolve => setTimeout(resolve, 100));
            
            // Verify data is loaded
            expect(loadedDb.getVectorCount()).to.equal(2);
            expect(loadedDb.getVector(1)).to.deep.equal(new Float32Array([0.1, 0.2, 0.3]));
            expect(loadedDb.getMetadata(1)).to.deep.equal({ tag: 'test1' });
        });
        
        it('should save and load with compression', async () => {
            // Create DB with compression
            const compressedDbPath = path.join(tempDir, 'compressed_db');
            const compressedDb = new VectorDB(3, compressedDbPath, { useCompression: true });
            
            // Add vectors
            compressedDb.addVector(1, [0.1, 0.2, 0.3]);
            
            // Save to disk
            await compressedDb.save();
            
            // Create a new instance with compression that loads from disk
            const loadedDb = new VectorDB(null, compressedDbPath, { useCompression: true });
            
            // Wait a bit for async loading
            await new Promise(resolve => setTimeout(resolve, 100));
            
            // Verify data is loaded
            expect(loadedDb.getVectorCount()).to.equal(1);
            expect(loadedDb.getVector(1)).to.deep.equal(new Float32Array([0.1, 0.2, 0.3]));
            
            await compressedDb.close();
        });
    });
    
    describe('Events', () => {
        it('should emit events on vector operations', (done) => {
            const db = new VectorDB(3);
            
            db.on('vector:add', (data) => {
                expect(data.id).to.equal(1);
                expect(data.dimensions).to.equal(3);
                done();
            });
            
            db.addVector(1, [0.1, 0.2, 0.3]);
        });
        
        it('should emit events on metadata operations', (done) => {
            const db = new VectorDB(3);
            db.addVector(1, [0.1, 0.2, 0.3]);
            
            db.on('metadata:add', (data) => {
                expect(data.id).to.equal(1);
                expect(data.metadata).to.deep.equal({ test: 'value' });
                done();
            });
            
            db.addMetadata(1, { test: 'value' });
        });
    });
});