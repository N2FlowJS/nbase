import { expect } from 'chai';
import { InvalidRequestError, MAX_GRAPH_THRESHOLD, MAX_K, requireInteger, requireNumber, requireVector, requireVectorId, stripDangerousKeys } from '../../src/server/utils/validation';

/**
 * The numeric fields of the search API used to be forwarded unchecked, which
 * turned two of them into denial of service rather than bad input.
 */
describe('server input validation', () => {
  describe('requireInteger', () => {
    it('accepts values inside the range', () => {
      expect(requireInteger(10, 'k', 1, MAX_K)).to.equal(10);
      expect(requireInteger('10', 'k', 1, MAX_K)).to.equal(10);
      expect(requireInteger(undefined, 'k', 1, MAX_K)).to.equal(undefined);
      expect(requireInteger(null, 'k', 1, MAX_K)).to.equal(undefined);
    });

    it('rejects NaN, which used to disable the HNSW termination test', () => {
      // `Math.max(NaN, efSearch)` is NaN, and `results.size >= NaN` is never
      // true, so the beam never stopped: a full index scan per request.
      expect(() => requireInteger('abc', 'k', 1, MAX_K)).to.throw(InvalidRequestError);
      expect(() => requireInteger(Number.NaN, 'k', 1, MAX_K)).to.throw(InvalidRequestError);
      expect(() => requireInteger(Number.POSITIVE_INFINITY, 'k', 1, MAX_K)).to.throw(InvalidRequestError);
    });

    it('rejects out-of-range and fractional values', () => {
      expect(() => requireInteger(0, 'k', 1, MAX_K)).to.throw(InvalidRequestError);
      expect(() => requireInteger(1e9, 'k', 1, MAX_K)).to.throw(InvalidRequestError);
      expect(() => requireInteger(1.5, 'k', 1, MAX_K)).to.throw(InvalidRequestError);
    });
  });

  describe('requireNumber', () => {
    it('bounds the graph threshold that drives the O(n^2) endpoints', () => {
      expect(requireNumber(0.5, 'threshold', 0, MAX_GRAPH_THRESHOLD)).to.equal(0.5);
      // A huge threshold makes every pair related, so the result set is n^2
      // objects even inside the vector-count budget.
      expect(() => requireNumber(1e12, 'threshold', 0, MAX_GRAPH_THRESHOLD)).to.throw(InvalidRequestError);
      expect(() => requireNumber(-1, 'threshold', 0, MAX_GRAPH_THRESHOLD)).to.throw(InvalidRequestError);
    });
  });

  describe('requireVectorId', () => {
    it('accepts strings and safe integers', () => {
      expect(requireVectorId('abc')).to.equal('abc');
      expect(requireVectorId(7)).to.equal(7);
      expect(requireVectorId(undefined)).to.equal(undefined);
    });

    it('rejects ids that would corrupt the id namespace', () => {
      // An object/null id was stored as a Map key, and because
      // `typeof id === 'number'` was false the auto-id counter never advanced:
      // the next generated id overwrote it.
      expect(() => requireVectorId({ a: 1 })).to.throw(InvalidRequestError);
      expect(() => requireVectorId([])).to.throw(InvalidRequestError);
      expect(() => requireVectorId('')).to.throw(InvalidRequestError);
      expect(() => requireVectorId(1.5)).to.throw(InvalidRequestError);
    });
  });

  describe('requireVector', () => {
    it('rejects components that Float32Array would silently turn into NaN', () => {
      expect(requireVector([1, 0, 0, 0])).to.deep.equal([1, 0, 0, 0]);
      expect(() => requireVector([1, '2', 0, 0])).to.throw(InvalidRequestError);
      expect(() => requireVector([1, null, 0, 0])).to.throw(InvalidRequestError);
      expect(() => requireVector([1, Number.NaN, 0, 0])).to.throw(InvalidRequestError);
      expect(() => requireVector([])).to.throw(InvalidRequestError);
      expect(() => requireVector(new Array(9000).fill(0))).to.throw(InvalidRequestError);
    });
  });

  describe('stripDangerousKeys', () => {
    it('drops keys that must not be persisted or echoed back', () => {
      const metadata = JSON.parse('{"label":"a","__proto__":{"admin":true},"constructor":"x"}') as Record<string, unknown>;
      const cleaned = stripDangerousKeys(metadata);

      expect(Object.keys(cleaned)).to.deep.equal(['label']);
      expect(Object.getPrototypeOf(cleaned)).to.equal(Object.prototype);
      expect((cleaned as Record<string, unknown>)['label']).to.equal('a');
    });
  });
});
