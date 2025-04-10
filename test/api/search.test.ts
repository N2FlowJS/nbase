import { expect } from 'chai';
import express from 'express';
import { beforeEach, describe, it } from 'mocha';
import * as sinon from 'sinon';
import request from 'supertest';
import { searchRoutes } from '../../src/server/routes/search';
import { ApiContext } from '../../src/types';

describe('Search Routes', () => {
  let app: express.Application;
  let context: ApiContext;
  let mockDatabase: any;
  let mockTimer: any;
  let mockCreateFilterFunction: any;
  let originalConsoleError: any;

  beforeEach(() => {
    // Save original console.error to restore later
    originalConsoleError = console.error;
    console.error = sinon.stub();

    // Create mock database
    mockDatabase = {
      findNearest: sinon.stub().resolves([
        { id: '1', score: 0.95, metadata: { title: 'Test 1' } },
        { id: '2', score: 0.85, metadata: { title: 'Test 2' } },
      ]),
    };

    // Create mock timer
    mockTimer = {
      start: sinon.stub().returns(undefined),
      stop: sinon.stub().returns({ total: 123 }),
    };

    // Create mock filter function
    mockCreateFilterFunction = sinon.stub().returns(() => true);

    // Create mock context
    context = {
      database: mockDatabase,
      timer: mockTimer,
      createFilterFunction: mockCreateFilterFunction,
    } as any;

    // Setup express app with search routes
    app = express();
    app.use(express.json());
    app.use('/search', searchRoutes(context));
  });

  afterEach(() => {
    console.error = originalConsoleError;
    sinon.restore();
  });

  it('should return 400 when query is missing', async () => {
    const response = await request(app).post('/search').send({});

    expect(response.status).to.equal(400);
    expect(response.body.error).to.equal('Invalid request: query vector array is required');
  });

  it('should return 400 when query is not an array', async () => {
    const response = await request(app).post('/search').send({ query: 'not an array' });

    expect(response.status).to.equal(400);
    expect(response.body.error).to.equal('Invalid request: query vector array is required');
  });

  it('should return search results with default options', async () => {
    const response = await request(app)
      .post('/search')
      .send({ query: [0.1, 0.2, 0.3], k: 2 });

    expect(response.status).to.equal(200);
    expect(response.body.results).to.be.an('array').with.lengthOf(2);
    expect(response.body.count).to.equal(2);
    expect(response.body.duration).to.equal(123);
    expect(response.body.searchOptions).to.deep.include({
      k: 2,
      method: 'clustered',
      partitionsSearched: 'all',
      filtersApplied: false,
      cacheUsed: true,
    });

    expect(mockTimer.start.calledWith('search')).to.be.true;
    expect(mockTimer.stop.calledWith('search')).to.be.true;
    expect(mockDatabase.findNearest.calledOnce).to.be.true;
  });

  it('should handle search with HNSW method', async () => {
    const response = await request(app)
      .post('/search')
      .send({
        query: [0.1, 0.2, 0.3],
        k: 5,
        method: 'hnsw',
        probes: 10,
        efSearch: 100,
      });

    expect(response.status).to.equal(200);
    expect(response.body.searchOptions.method).to.equal('hnsw');

    // Verify correct options were passed to database.findNearest
    const searchOptionsArg = mockDatabase.findNearest.args[0][2];
    expect(searchOptionsArg.useHNSW).to.be.true;
    expect(searchOptionsArg.probes).to.equal(10);
    expect(searchOptionsArg.efSearch).to.equal(100);
  });

  it('should handle search with filters', async () => {
    const filters = { category: 'test' };

    const response = await request(app)
      .post('/search')
      .send({
        query: [0.1, 0.2, 0.3],
        filters: filters,
      });

    expect(response.status).to.equal(200);
    expect(response.body.searchOptions.filtersApplied).to.be.true;
    expect(mockCreateFilterFunction.calledWith(filters)).to.be.true;
  });

  it('should handle search with array filters', async () => {
    const filters = [{ field: 'category', value: 'test' }];

    const response = await request(app)
      .post('/search')
      .send({
        query: [0.1, 0.2, 0.3],
        filters: filters,
      });

    expect(response.status).to.equal(200);
    expect(response.body.searchOptions.filtersApplied).to.be.true;
    expect(mockCreateFilterFunction.calledWith(filters)).to.be.true;
  });

  it('should handle search with partitionIds', async () => {
    const partitionIds = ['partition1', 'partition2'];

    const response = await request(app)
      .post('/search')
      .send({
        query: [0.1, 0.2, 0.3],
        partitionIds: partitionIds,
      });

    expect(response.status).to.equal(200);
    expect(response.body.searchOptions.partitionsSearched).to.equal(2);
  });

  it('should handle database errors', async () => {
    const errorMessage = 'Database error';
    mockDatabase.findNearest.rejects(new Error(errorMessage));

    const response = await request(app)
      .post('/search')
      .send({ query: [0.1, 0.2, 0.3] });

    expect(response.status).to.equal(500);
    expect(response.body.error).to.equal(errorMessage);
    expect(response.body.duration).to.equal(123);
  });

  it('should include stack trace in development mode', async () => {
    process.env.NODE_ENV = 'development';
    mockDatabase.findNearest.rejects(new Error('Test error'));

    const response = await request(app)
      .post('/search')
      .send({ query: [0.1, 0.2, 0.3] });

    expect(response.status).to.equal(500);
    expect(response.body.stack).to.be.a('string');

    // Reset NODE_ENV
    delete process.env.NODE_ENV;
  });
});
