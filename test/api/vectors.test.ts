import { expect } from 'chai';
import express from 'express';
import { beforeEach, describe, it } from 'mocha';
import request from 'supertest';
import { vectorRoutes } from '../../src/server/routes/vectors';
import { ApiContext } from '../../src/types';

describe('Vector Routes', () => {
  let app: express.Application;
  let mockDatabase: any;
  let mockTimer: any;
  let context: ApiContext;

  beforeEach(() => {
    // Create mock database
    mockDatabase = {
      IsReady: () => true,
      addVector: async () => ({ partitionId: 'partition1', vectorId: 'vec1' }),
      bulkAdd: async () => ({
        count: 3,
        partitionIds: ['partition1', 'partition2'],
      }),
      getVector: async (id: string | number) => {
        if (id === 'nonexistent') return null;
        return {
          partitionId: 'partition1',
          vector: new Float32Array([0.1, 0.2, 0.3]),
        };
      },
      getMetadata: async (id: string | number) => {
        if (id === 'nonexistent') return null;
        return {
          partitionId: 'partition1',
          metadata: { name: 'test', dimension: 3 },
        };
      },
      hasVector: async (id: string | number) => id !== 'nonexistent',
      updateMetadata: async (id: string | number, metadataOrFn: any) => id !== 'nonexistent',
      deleteVector: async (id: string | number) => id !== 'nonexistent',
      search: async () => [
        {
          id: 'vec1',
          score: 0.9,
          vector: new Float32Array([0.1, 0.2, 0.3]),
          metadata: { name: 'similar1' },
        },
        {
          id: 'vec2',
          score: 0.8,
          vector: new Float32Array([0.2, 0.3, 0.4]),
          metadata: { name: 'similar2' },
        },
      ],
    };

    // Create mock timer
    mockTimer = {
      start: (name: string) => {},
      stop: (name: string) => ({ total: 10 }),
    };

    // Setup context
    context = {
      database: mockDatabase,
      timer: mockTimer,
    } as ApiContext;

    // Setup express app
    app = express();
    app.use(express.json());
    app.use('/api/vectors', vectorRoutes(context));
  });

  describe('POST /', () => {
    it('should add a single vector successfully', async () => {
      const response = await request(app)
        .post('/api/vectors')
        .send({
          id: 'test-id',
          vector: [0.1, 0.2, 0.3],
          metadata: { name: 'test vector' },
        });

      expect(response.status).to.equal(201);
      expect(response.body.success).to.be.true;
      expect(response.body.count).to.equal(1);
      expect(response.body.ids).to.deep.equal(['vec1']);
    });

    it('should handle bulk vector addition successfully', async () => {
      const response = await request(app)
        .post('/api/vectors')
        .send({
          vectors: [
            {
              id: 'vec1',
              vector: [0.1, 0.2, 0.3],
              metadata: { name: 'test1' },
            },
            {
              id: 'vec2',
              vector: [0.4, 0.5, 0.6],
              metadata: { name: 'test2' },
            },
          ],
        });

      expect(response.status).to.equal(200);
      expect(response.body.success).to.be.true;
      expect(response.body.count).to.equal(3);
    });

    it('should return 400 when a single vector request is missing vector array', async () => {
      const response = await request(app)
        .post('/api/vectors')
        .send({
          id: 'test-id',
          metadata: { name: 'test vector' },
        });

      expect(response.status).to.equal(400);
      expect(response.body.success).to.be.false;
    });

    it('should return 400 when bulk vectors request has invalid format', async () => {
      const response = await request(app).post('/api/vectors').send({
        vectors: 'not-an-array',
      });

      expect(response.status).to.equal(400);
      expect(response.body.success).to.be.false;
    });

    it('should return 503 when database is not ready', async () => {
      mockDatabase.IsReady = () => false;

      const response = await request(app)
        .post('/api/vectors')
        .send({
          id: 'test-id',
          vector: [0.1, 0.2, 0.3],
        });

      expect(response.status).to.equal(503);
      expect(response.body.success).to.be.false;
    });
  });

  describe('GET /:id', () => {
    it('should get a vector by ID with vector and metadata', async () => {
      const response = await request(app).get('/api/vectors/test-id?includeVector=true');
      expect(response.status).to.equal(200);
      expect(response.body.success).to.be.true;
      expect(response.body.id).to.equal('test-id');
      expect(response.body.vector).to.deep.equal([0.10000000149011612, 0.20000000298023224, 0.30000001192092896]);
      expect(response.body.metadata).to.deep.equal({
        name: 'test',
        dimension: 3,
      });
    });

    it('should return 404 when vector does not exist', async () => {
      const response = await request(app).get('/api/vectors/nonexistent');

      expect(response.status).to.equal(404);
      expect(response.body.success).to.be.false;
    });
  });

  describe('GET /:id/exists', () => {
    it('should confirm a vector exists', async () => {
      const response = await request(app).get('/api/vectors/test-id/exists');

      expect(response.status).to.equal(200);
      expect(response.body.success).to.be.true;
      expect(response.body.exists).to.be.true;
      expect(response.body.dimension).to.equal(3);
    });

    it('should confirm a vector does not exist', async () => {
      const response = await request(app).get('/api/vectors/nonexistent/exists');

      expect(response.status).to.equal(200);
      expect(response.body.success).to.be.true;
      expect(response.body.exists).to.be.false;
    });
  });

  describe('PATCH /:id/metadata', () => {
    it('should update metadata with merge operation', async () => {
      const response = await request(app)
        .patch('/api/vectors/test-id/metadata')
        .send({
          metadata: { category: 'updated' },
          operation: 'merge',
        });

      expect(response.status).to.equal(200);
      expect(response.body.success).to.be.true;
      expect(response.body.id).to.equal('test-id');
      expect(response.body.operation).to.equal('merge');
    });

    it('should update metadata with replace operation', async () => {
      const response = await request(app)
        .patch('/api/vectors/test-id/metadata')
        .send({
          metadata: { category: 'new' },
          operation: 'replace',
        });

      expect(response.status).to.equal(200);
      expect(response.body.success).to.be.true;
      expect(response.body.operation).to.equal('replace');
    });

    it('should return 400 when metadata is missing', async () => {
      const response = await request(app).patch('/api/vectors/test-id/metadata').send({
        operation: 'merge',
      });

      expect(response.status).to.equal(400);
      expect(response.body.success).to.be.false;
    });

    it('should return 404 when vector does not exist', async () => {
      const response = await request(app)
        .patch('/api/vectors/nonexistent/metadata')
        .send({
          metadata: { category: 'new' },
        });

      expect(response.status).to.equal(404);
      expect(response.body.success).to.be.false;
    });
  });

  describe('DELETE /:id', () => {
    it('should delete a vector successfully', async () => {
      const response = await request(app).delete('/api/vectors/test-id');

      expect(response.status).to.equal(200);
      expect(response.body.success).to.be.true;
      expect(response.body.id).to.equal('test-id');
    });

    it('should return 404 when vector to delete does not exist', async () => {
      const response = await request(app).delete('/api/vectors/nonexistent');

      expect(response.status).to.equal(404);
      expect(response.body.success).to.be.false;
    });
  });

  describe('GET /:id/similar', () => {
    it('should find similar vectors', async () => {
      const response = await request(app).get('/api/vectors/test-id/similar?k=2&includeMetadata=true');

      expect(response.status).to.equal(200);
      expect(response.body.success).to.be.true;
      expect(response.body.queryId).to.equal('test-id');
      expect(response.body.results).to.be.an('array');
      expect(response.body.results.length).to.equal(2);
    });

    it('should return 404 when source vector does not exist', async () => {
      const response = await request(app).get('/api/vectors/nonexistent/similar');

      expect(response.status).to.equal(404);
      expect(response.body.success).to.be.false;
    });
  });
});
