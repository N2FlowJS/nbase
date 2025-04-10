import { expect } from 'chai';
import { Request, Response } from 'express';
import { beforeEach, describe, it } from 'mocha';
import * as sinon from 'sinon';
import express from 'express';
import { searchRoutes } from '../../src/server/routes/search';
import { ApiContext } from '../../src/types';

describe('Search Metadata Endpoint', () => {
  let mockDatabase: any;
  let mockTimer: any;
  let mockCreateFilterFunction: any;
  let mockContext: ApiContext;
  let searchRouter: express.Router;
  let originalConsoleError: any;

  // Sample metadata for tests
  const sampleMetadataResults = [
    { partitionId: 'p1', vectorId: 'v1', metadata: { type: 'article', title: 'Test 1', published: true } },
    { partitionId: 'p2', vectorId: 'v2', metadata: { type: 'book', title: 'Test 2', published: false } },
    { partitionId: 'p1', vectorId: 'v3', metadata: { type: 'article', author: 'Author', published: true } },
  ];

  // Mock Express request and response objects
  const mockRequest = (body: any): Request =>
    ({
      body,
    } as unknown as Request);

  const mockResponse = (): { status: sinon.SinonStub; json: sinon.SinonStub } => {
    return {
      status: sinon.stub().returnsThis(),
      json: sinon.stub().returnsThis(),
    };
  };

  beforeEach(() => {
    // Save original console.error to restore later
    originalConsoleError = console.error;
    console.error = sinon.stub();

    // Create mock database
    mockDatabase = {
      getMetadataWithField: sinon.stub().resolves(sampleMetadataResults),
      getVector: sinon.stub().resolves({
        partitionId: 'p1',
        vector: new Float32Array([0.1, 0.2, 0.3]),
      }),
      IsReady: sinon.stub().returns(true),
    };

    // Create mock timer
    mockTimer = {
      start: sinon.stub().returns(undefined),
      stop: sinon.stub().returns({ total: 42 }),
    };

    // Create mock filter function
    mockCreateFilterFunction = sinon.stub().returns(() => true);

    // Setup the API context with mocks
    mockContext = {
      database: mockDatabase as any,
      timer: mockTimer as any,
      createFilterFunction: mockCreateFilterFunction,
    };

    // Create the router with mocks
    searchRouter = searchRoutes(mockContext);
  });

  afterEach(() => {
    console.error = originalConsoleError;
    sinon.restore();
  });

  // Find the route handler for /metadata POST
  const findMetadataRouteHandler = () => {
    const router = searchRouter as any;
    const metadataRoute = router.stack.find((layer: any) => layer.route?.path === '/metadata');
    return metadataRoute?.route?.stack[0]?.handle;
  };

  it('Should return matching metadata entries for string criteria', async () => {
    const metadataRouteHandler = findMetadataRouteHandler();
    if (!metadataRouteHandler) throw new Error('Metadata route handler not found');

    const req = mockRequest({ criteria: 'type' });
    const res = mockResponse();

    await metadataRouteHandler(req, res, sinon.stub());

    expect(mockTimer.start.calledWith('metadata_search')).to.be.true;
    expect(mockDatabase.getMetadataWithField.calledWith('type', undefined)).to.be.true;
    expect(mockTimer.stop.calledWith('metadata_search')).to.be.true;
    expect(
      res.json.calledWith({
        results: sampleMetadataResults,
        count: 3,
        duration: 42,
      })
    ).to.be.true;
  });

  it('Should return matching metadata entries for string criteria with value', async () => {
    const metadataRouteHandler = findMetadataRouteHandler();
    if (!metadataRouteHandler) throw new Error('Metadata route handler not found');

    const req = mockRequest({ criteria: 'type', values: 'article' });
    const res = mockResponse();

    await metadataRouteHandler(req, res, sinon.stub());

    expect(mockDatabase.getMetadataWithField.calledWith('type', 'article')).to.be.true;
    expect(
      res.json.calledWith({
        results: sampleMetadataResults,
        count: 3,
        duration: 42,
      })
    ).to.be.true;
  });

  it('Should return matching metadata entries for array criteria', async () => {
    const metadataRouteHandler = findMetadataRouteHandler();
    if (!metadataRouteHandler) throw new Error('Metadata route handler not found');

    const req = mockRequest({ criteria: ['type', 'published'] });
    const res = mockResponse();

    await metadataRouteHandler(req, res, sinon.stub());

    expect(mockDatabase.getMetadataWithField.calledWith(['type', 'published'], undefined)).to.be.true;
    expect(
      res.json.calledWith({
        results: sampleMetadataResults,
        count: 3,
        duration: 42,
      })
    ).to.be.true;
  });

  it('Should return matching metadata entries for array criteria with values', async () => {
    const metadataRouteHandler = findMetadataRouteHandler();
    if (!metadataRouteHandler) throw new Error('Metadata route handler not found');

    const req = mockRequest({ criteria: ['type', 'published'], values: ['article', true] });
    const res = mockResponse();

    await metadataRouteHandler(req, res, sinon.stub());

    expect(mockDatabase.getMetadataWithField.calledWith(['type', 'published'], ['article', true])).to.be.true;
    expect(
      res.json.calledWith({
        results: sampleMetadataResults,
        count: 3,
        duration: 42,
      })
    ).to.be.true;
  });

  it('Should return matching metadata entries for object criteria', async () => {
    const metadataRouteHandler = findMetadataRouteHandler();
    if (!metadataRouteHandler) throw new Error('Metadata route handler not found');

    const criteria = { type: 'article', published: true };
    const req = mockRequest({ criteria });
    const res = mockResponse();

    await metadataRouteHandler(req, res, sinon.stub());

    expect(mockDatabase.getMetadataWithField.calledWith(criteria, undefined)).to.be.true;
    expect(
      res.json.calledWith({
        results: sampleMetadataResults,
        count: 3,
        duration: 42,
      })
    ).to.be.true;
  });

  it('Should include vectors when includeVectors is true', async () => {
    const metadataRouteHandler = findMetadataRouteHandler();
    if (!metadataRouteHandler) throw new Error('Metadata route handler not found');

    const req = mockRequest({ criteria: 'type', includeVectors: true });
    const res = mockResponse();

    await metadataRouteHandler(req, res, sinon.stub());

    expect(mockDatabase.getMetadataWithField.calledWith('type', undefined)).to.be.true;
    expect(mockDatabase.getVector.callCount).to.equal(3);
    expect(res.json.calledOnce).to.be.true;
    // We can check that json was called with an object containing count and duration
    const jsonArg = res.json.firstCall.args[0];
    expect(jsonArg).to.have.property('count', 3);
    expect(jsonArg).to.have.property('duration', 42);
  });

  it('Should return 400 for invalid criteria type', async () => {
    const metadataRouteHandler = findMetadataRouteHandler();
    if (!metadataRouteHandler) throw new Error('Metadata route handler not found');

    const req = mockRequest({ criteria: 123 }); // Number is invalid criteria type
    const res = mockResponse();

    await metadataRouteHandler(req, res, sinon.stub());

    expect(mockDatabase.getMetadataWithField.called).to.be.false;
    expect(res.status.calledWith(400)).to.be.true;
    expect(res.json.calledOnce).to.be.true;
    expect(res.json.firstCall.args[0]).to.have.property('error');
  });

  it('Should return 400 for missing criteria', async () => {
    const metadataRouteHandler = findMetadataRouteHandler();
    if (!metadataRouteHandler) throw new Error('Metadata route handler not found');

    const req = mockRequest({});
    const res = mockResponse();

    await metadataRouteHandler(req, res, sinon.stub());

    expect(mockDatabase.getMetadataWithField.called).to.be.false;
    expect(res.status.calledWith(400)).to.be.true;
    expect(res.json.calledOnce).to.be.true;
    expect(res.json.firstCall.args[0]).to.have.property('error');
  });

  it('Should return 500 if database.getMetadataWithField throws error', async () => {
    const metadataRouteHandler = findMetadataRouteHandler();
    if (!metadataRouteHandler) throw new Error('Metadata route handler not found');

    const req = mockRequest({ criteria: 'type' });
    const res = mockResponse();
    const testError = new Error('Database error');

    mockDatabase.getMetadataWithField.rejects(testError);

    await metadataRouteHandler(req, res, sinon.stub());

    expect(mockTimer.start.calledWith('metadata_search')).to.be.true;
    expect(mockDatabase.getMetadataWithField.calledWith('type', undefined)).to.be.true;
    expect(mockTimer.stop.calledWith('metadata_search')).to.be.true;
    expect(res.status.calledWith(500)).to.be.true;
    expect(res.json.firstCall.args[0]).to.have.property('error', 'Database error');
    expect(res.json.firstCall.args[0]).to.have.property('duration', 42);
  });

  it('Should return empty results correctly', async () => {
    const metadataRouteHandler = findMetadataRouteHandler();
    if (!metadataRouteHandler) throw new Error('Metadata route handler not found');

    const req = mockRequest({ criteria: 'nonExistentField' });
    const res = mockResponse();

    // Reset stub behavior for this test only
    mockDatabase.getMetadataWithField.reset();
    mockDatabase.getMetadataWithField.resolves([]);

    await metadataRouteHandler(req, res, sinon.stub());

    expect(mockDatabase.getMetadataWithField.calledWith('nonExistentField', undefined)).to.be.true;
    expect(
      res.json.calledWith({
        results: [],
        count: 0,
        duration: 42,
      })
    ).to.be.true;
  });

  it('Should handle nested object criteria', async () => {
    const metadataRouteHandler = findMetadataRouteHandler();
    if (!metadataRouteHandler) throw new Error('Metadata route handler not found');

    const nestedCriteria = { 'metadata.nested.field': 'value' };
    const req = mockRequest({ criteria: nestedCriteria });
    const res = mockResponse();

    await metadataRouteHandler(req, res, sinon.stub());

    expect(mockDatabase.getMetadataWithField.calledWith(nestedCriteria, undefined)).to.be.true;
    expect(
      res.json.calledWith({
        results: sampleMetadataResults,
        count: 3,
        duration: 42,
      })
    ).to.be.true;
  });

  it('Should handle array values with mixed types', async () => {
    const metadataRouteHandler = findMetadataRouteHandler();
    if (!metadataRouteHandler) throw new Error('Metadata route handler not found');

    const req = mockRequest({ criteria: 'field', values: ['value1', 123, true] });
    const res = mockResponse();

    await metadataRouteHandler(req, res, sinon.stub());

    expect(mockDatabase.getMetadataWithField.calledWith('field', ['value1', 123, true])).to.be.true;
    expect(
      res.json.calledWith({
        results: sampleMetadataResults,
        count: 3,
        duration: 42,
      })
    ).to.be.true;
  });

  it('Should handle limit parameter when provided', async () => {
    const metadataRouteHandler = findMetadataRouteHandler();
    if (!metadataRouteHandler) throw new Error('Metadata route handler not found');

    const req = mockRequest({ criteria: 'type', limit: 1 });
    const res = mockResponse();

    // Reset stub behavior for this test only
    mockDatabase.getMetadataWithField.reset();
    mockDatabase.getMetadataWithField.resolves([sampleMetadataResults[0]]);

    await metadataRouteHandler(req, res, sinon.stub());

    // Check that the limit was passed through to the database call
    expect(mockDatabase.getMetadataWithField.calledOnce).to.be.true;
    expect(mockDatabase.getMetadataWithField.firstCall.args[0]).to.equal('type');
    expect(mockDatabase.getMetadataWithField.firstCall.args[1]).to.equal(undefined);
    expect(mockDatabase.getMetadataWithField.firstCall.args[2]).to.deep.equal({ limit: 1 });

    expect(
      res.json.calledWith({
        results: [sampleMetadataResults[0]],
        count: 1,
        duration: 42,
      })
    ).to.be.true;
  });

  it('Should handle error when getVector fails for includeVectors', async () => {
    const metadataRouteHandler = findMetadataRouteHandler();
    if (!metadataRouteHandler) throw new Error('Metadata route handler not found');

    const req = mockRequest({ criteria: 'type', includeVectors: true });
    const res = mockResponse();

    // Reset stub behavior for getVector
    mockDatabase.getVector.reset();
    mockDatabase.getVector.onFirstCall().resolves({
      partitionId: 'p1',
      vector: new Float32Array([0.1, 0.2, 0.3]),
    });
    mockDatabase.getVector.onSecondCall().rejects(new Error('Vector not found'));
    mockDatabase.getVector.onThirdCall().resolves({
      partitionId: 'p1',
      vector: new Float32Array([0.1, 0.2, 0.3]),
    });

    await metadataRouteHandler(req, res, sinon.stub());

    expect(mockDatabase.getVector.callCount).to.equal(3); // Should stop after the error
    expect(res.status.calledWith(500)).to.be.true;
    expect(res.json.calledOnce).to.be.true;
    expect(res.json.firstCall.args[0]).to.have.property('error', 'Vector not found');
    expect(res.json.firstCall.args[0]).to.have.property('duration', 42);
  });

  it('Should handle large result sets efficiently', async () => {
    const metadataRouteHandler = findMetadataRouteHandler();
    if (!metadataRouteHandler) throw new Error('Metadata route handler not found');

    const req = mockRequest({ criteria: 'type' });
    const res = mockResponse();

    // Create a large mock result set
    const largeResultSet = Array(1000)
      .fill(null)
      .map((_, i) => ({
        partitionId: `p${i % 5}`,
        vectorId: `v${i}`,
        metadata: { type: i % 2 === 0 ? 'article' : 'book', title: `Test ${i}` },
      }));

    // Reset stub behavior for this test only
    mockDatabase.getMetadataWithField.reset();
    mockDatabase.getMetadataWithField.resolves(largeResultSet);

    await metadataRouteHandler(req, res, sinon.stub());

    expect(res.json.calledOnce).to.be.true;
    const jsonArg = res.json.firstCall.args[0];
    expect(jsonArg).to.have.property('count', 1000);
    expect(jsonArg).to.have.property('duration', 42);
    expect(jsonArg.results).to.equal(largeResultSet);
  });
});
