import { euclidean, squaredEuclidean, dotProduct, cosine } from '../../src/utils/distance_metrics';

function naiveSquaredEuclidean(a: number[] | Float32Array, b: number[] | Float32Array): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const diff = a[i] - b[i];
    sum += diff * diff;
  }
  return sum;
}

function naiveDotProduct(a: number[] | Float32Array, b: number[] | Float32Array): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    sum += a[i] * b[i];
  }
  return sum;
}

const DIM = 1536;
const ITERATIONS = 1000000;

const v1 = new Float32Array(DIM).map(() => Math.random());
const v2 = new Float32Array(DIM).map(() => Math.random());

console.log(`Benchmarking with vector dimension: ${DIM}, iterations: ${ITERATIONS}`);

console.time('Naive Squared Euclidean');
for (let i = 0; i < ITERATIONS; i++) {
  naiveSquaredEuclidean(v1, v2);
}
console.timeEnd('Naive Squared Euclidean');

console.time('Optimized Squared Euclidean');
for (let i = 0; i < ITERATIONS; i++) {
  squaredEuclidean(v1, v2);
}
console.timeEnd('Optimized Squared Euclidean');

console.log('---');

console.time('Naive Dot Product');
for (let i = 0; i < ITERATIONS; i++) {
  naiveDotProduct(v1, v2);
}
console.timeEnd('Naive Dot Product');

console.time('Optimized Dot Product');
for (let i = 0; i < ITERATIONS; i++) {
  dotProduct(v1, v2);
}
console.timeEnd('Optimized Dot Product');
