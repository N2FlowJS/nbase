import { expect } from 'chai';
import HNSW from '../src/ann/hnsw';
import { VectorProvider } from '../src/types';

class MockVectorProvider implements VectorProvider {
  private vectors: Map<number | string, Float32Array> = new Map();

  addVector(id: number | string, vector: number[]) {
    this.vectors.set(id, new Float32Array(vector));
  }

  getVector(id: number | string) {
    return this.vectors.get(id);
  }

  getVectorIds() {
    return Array.from(this.vectors.keys());
  }
}

describe('HNSW Worker Thread Indexing', function() {
  this.timeout(10000); // Indexing can take time

  it('should build HNSW index using a worker thread', async () => {
    const provider = new MockVectorProvider();
    // Add 100 random vectors
    for (let i = 0; i < 100; i++) {
      provider.addVector(i, Array.from({ length: 128 }, () => Math.random()));
    }

    const hnsw = new HNSW(provider, { M: 16, efConstruction: 100 });
    
    let progressCalled = false;
    await hnsw.buildIndex({
      useWorker: true,
      progressCallback: () => {
        progressCalled = true;
        // console.log(`Progress: ${progress}`);
      }
    });

    expect(hnsw.getNodeCount()).to.equal(100);
    expect(progressCalled).to.be.true;

    // Test searching works on the built index
    const query = Array.from({ length: 128 }, () => Math.random());
    const results = hnsw.findNearest(new Float32Array(query), 5);
    expect(results).to.have.lengthOf(5);
  });
});
