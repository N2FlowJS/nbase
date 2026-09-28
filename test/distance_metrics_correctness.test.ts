import { expect } from 'chai';
import { cosine, dotProduct, euclidean, manhattan, squaredEuclidean } from '../src/utils/distance_metrics';

describe('Distance Metrics', () => {
  const v1 = [1, 2, 3];
  const v2 = [4, 5, 6];
  const v1f = new Float32Array([1, 2, 3]);
  const v2f = new Float32Array([4, 5, 6]);

  describe('euclidean', () => {
    it('should calculate euclidean distance correctly', () => {
      const expected = Math.sqrt(Math.pow(1-4, 2) + Math.pow(2-5, 2) + Math.pow(3-6, 2));
      expect(euclidean(v1, v2)).to.be.closeTo(expected, 1e-10);
      expect(euclidean(v1f, v2f)).to.be.closeTo(expected, 1e-10);
    });
  });

  describe('manhattan', () => {
    it('should calculate manhattan distance correctly', () => {
      const expected = Math.abs(1-4) + Math.abs(2-5) + Math.abs(3-6);
      expect(manhattan(v1, v2)).to.equal(expected);
      expect(manhattan(v1f, v2f)).to.equal(expected);
    });
  });

  describe('cosine', () => {
    it('should calculate cosine distance correctly', () => {
      const dot = 1*4 + 2*5 + 3*6;
      const normA = Math.sqrt(1*1 + 2*2 + 3*3);
      const normB = Math.sqrt(4*4 + 5*5 + 6*6);
      const expected = 1 - (dot / (normA * normB));
      expect(cosine(v1, v2)).to.be.closeTo(expected, 1e-10);
      expect(cosine(v1f, v2f)).to.be.closeTo(expected, 1e-10);
    });
  });

  describe('dotProduct', () => {
    it('should calculate dot product correctly', () => {
      const expected = 1*4 + 2*5 + 3*6;
      expect(dotProduct(v1, v2)).to.equal(expected);
      expect(dotProduct(v1f, v2f)).to.equal(expected);
    });
  });

  describe('squaredEuclidean', () => {
    it('should calculate squared euclidean distance correctly', () => {
      const expected = Math.pow(1-4, 2) + Math.pow(2-5, 2) + Math.pow(3-6, 2);
      expect(squaredEuclidean(v1, v2)).to.equal(expected);
      expect(squaredEuclidean(v1f, v2f)).to.equal(expected);
    });
  });

  describe('Loop Unrolling Correctness', () => {
    it('should handle vectors with length not multiple of unroll factor', () => {
      const va = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
      const vb = new Float32Array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1]);
      
      const expectedSq = va.reduce((acc, val, i) => acc + Math.pow(val - (vb[i] ?? 0), 2), 0);
      expect(squaredEuclidean(va, vb)).to.be.closeTo(expectedSq, 1e-10);

      const expectedDot = va.reduce((acc, val, i) => acc + val * (vb[i] ?? 0), 0);
      expect(dotProduct(va, vb)).to.be.closeTo(expectedDot, 1e-10);
    });
  });
});
