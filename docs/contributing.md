# Contributing to NBase

## Overview

Welcome to NBase! We're excited to have you contribute to our high-performance vector database. This guide will help you get started with development, understand our processes, and make meaningful contributions to the project.

## Development Environment Setup

### Prerequisites

**Required:**
- **Node.js**: v16.0.0 or higher (v18 LTS recommended)
- **NPM**: v7.0.0 or higher (or Yarn v1.22.0+, PNPM v6.0.0+)
- **Git**: v2.25.0 or higher
- **TypeScript**: v4.5.0 or higher (installed automatically)

**Recommended:**
- **VS Code** with TypeScript and Node.js extensions
- **GitHub CLI** for streamlined workflow
- **Docker** for testing and deployment
- **Understanding of**: Vector databases, ANN algorithms, TypeScript, Node.js

### Quick Start

```bash
# Fork and clone the repository
git clone https://github.com/your-username/nbase.git
cd nbase

# Install dependencies
npm install

# Set up development environment
npm run setup:dev

# Run initial tests
npm test

# Start development server
npm run dev
```

### Development Scripts

```bash
# Core development
npm run build          # Build TypeScript to JavaScript
npm run dev            # Start development server with hot reload
npm run clean          # Clean build artifacts
npm run type-check     # Run TypeScript type checking

# Testing
npm test               # Run all tests
npm run test:unit      # Run unit tests only
npm run test:integration # Run integration tests
npm run test:watch     # Run tests in watch mode
npm run test:coverage  # Generate test coverage report

# Benchmarking
npm run benchmark      # Run all benchmarks
npm run benchmark:suite1 # Run large-scale benchmarks
npm run benchmark:suite2 # Run latency benchmarks

# Quality
npm run lint           # Run ESLint
npm run lint:fix       # Auto-fix linting issues
npm run format         # Format code with Prettier
npm run type-check     # TypeScript type checking

# Documentation
npm run docs           # Generate API documentation
npm run docs:serve     # Serve documentation locally
npm run docs:build     # Build documentation for deployment
```

## Project Structure

```
nbase/
├── src/                          # Source code
│   ├── index.ts                  # Main entry point
│   ├── types.ts                  # TypeScript type definitions
│   ├── ann/                      # Approximate Nearest Neighbor algorithms
│   │   ├── hnsw.ts              # HNSW implementation
│   │   └── lsh.ts               # LSH implementation
│   ├── compression/             # Vector compression algorithms
│   │   ├── index.ts
│   │   ├── kmeans.ts
│   │   └── product_quantization.ts
│   ├── config/                  # Configuration management
│   │   ├── default.ts
│   │   ├── factory.ts
│   │   └── index.ts
│   ├── database/                # Core database implementations
│   │   ├── database.ts         # Main database class
│   │   └── scripts/
│   │       └── service.ts
│   ├── search/                  # Search engine implementations
│   │   ├── batch_search.ts
│   │   ├── hybrid_search.ts
│   │   ├── knn_search.ts
│   │   ├── reranking.ts
│   │   └── unified_search.ts
│   ├── server/                  # REST API server
│   │   ├── index.ts
│   │   ├── middleware/
│   │   │   └── common.ts
│   │   └── routes/
│   │       ├── index.ts
│   │       ├── search.ts
│   │       └── vectors.ts
│   ├── utils/                   # Utility functions
│   │   ├── distance_metrics.ts
│   │   ├── log.ts
│   │   ├── profiling.ts
│   │   └── vector_monitoring.ts
│   └── vector/                  # Vector database implementations
│       ├── clustered_vector_db.ts
│       ├── index.ts
│       ├── partitioned_vector_db.ts
│       └── vector_db.ts
├── test/                        # Test files
│   ├── api/                     # API tests
│   ├── benchmarks/              # Performance benchmarks
│   ├── test-helpers/            # Test utilities
│   └── *.test.ts                # Unit and integration tests
├── docs/                        # Documentation
│   ├── api-reference.md
│   ├── getting-started.md
│   ├── search-techniques.md
│   └── ...
├── database/                    # Runtime data storage
│   └── partitions/              # Partitioned data
├── node_modules/                # Dependencies (generated)
├── package.json                 # Package configuration
├── tsconfig.json               # TypeScript configuration
├── eslint.config.mjs           # ESLint configuration
├── nodemon.json                # Development server config
└── typedoc.json                # API documentation config
```

## Development Workflow

### 1. Choose an Issue

- Check [GitHub Issues](https://github.com/n2flowjs/nbase/issues) for open tasks
- Look for issues labeled `good first issue` or `help wanted`
- Comment on the issue to indicate you're working on it

### 2. Create a Branch

```bash
# Create and switch to a new branch
git checkout -b feature/your-feature-name

# Or for bug fixes
git checkout -b fix/issue-description

# Or for documentation
git checkout -b docs/update-readme
```

### 3. Make Changes

```bash
# Ensure you're working with latest code
git pull origin main

# Make your changes
# ... edit files ...

# Run tests to ensure nothing breaks
npm test

# Run linting
npm run lint

# Format code
npm run format
```

### 4. Test Your Changes

```bash
# Run unit tests
npm run test:unit

# Run integration tests
npm run test:integration

# Run benchmarks to ensure performance
npm run benchmark

# Test with your specific use case
node -e "
const { Database } = require('./dist');
const db = new Database({ vectorSize: 128 });
// ... your test code ...
"
```

### 5. Commit Your Changes

```bash
# Stage your changes
git add .

# Commit with descriptive message
git commit -m "feat: add new HNSW optimization

- Implement dynamic M parameter adjustment
- Add memory usage monitoring
- Improve search accuracy by 15%

Closes #123"

# Follow conventional commit format
# Type can be: feat, fix, docs, style, refactor, test, chore
```

### 6. Create Pull Request

```bash
# Push your branch
git push origin feature/your-feature-name

# Create PR on GitHub or use GitHub CLI
gh pr create --title "Add new HNSW optimization" --body "Detailed description..."
```

## Coding Standards

### TypeScript Guidelines

```typescript
// ✅ Good: Use interfaces for public APIs
interface SearchOptions {
  readonly k: number;
  readonly includeMetadata?: boolean;
  readonly distanceMetric?: DistanceMetric;
  readonly useHNSW?: boolean;
}

// ✅ Good: Use type unions for constrained values
type DistanceMetric = 'cosine' | 'euclidean' | 'manhattan' | 'dot-product';

// ✅ Good: Document complex functions with TSDoc
/**
 * Performs approximate nearest neighbor search using HNSW algorithm.
 *
 * @param query - The query vector to search for
 * @param options - Search configuration options
 * @returns Promise resolving to search results sorted by distance
 *
 * @example
 * ```typescript
 * const results = await db.search(queryVector, {
 *   k: 10,
 *   useHNSW: true,
 *   distanceMetric: 'cosine'
 * });
 * ```
 *
 * @throws {ValidationError} When query vector dimension doesn't match database
 * @throws {IndexNotBuiltError} When HNSW index is not available
 */
async function search(query: Float32Array, options: SearchOptions): Promise<SearchResult[]> {
  // Implementation
}

// ❌ Bad: Avoid any types
function badFunction(param: any): any {
  return param;
}

// ❌ Bad: Avoid large interfaces
interface BadInterface {
  prop1: string;
  prop2: number;
  prop3: boolean;
  prop4: string[];
  // ... many more properties
}
```

### Code Style

```typescript
// ✅ Good: Use descriptive variable names
const queryVector = new Float32Array(128);
const searchResults = await database.search(queryVector, { k: 10 });

// ✅ Good: Use early returns
function validateInput(input: any): boolean {
  if (!input) return false;
  if (!Array.isArray(input.vector)) return false;
  if (input.vector.length !== this.vectorSize) return false;

  return true;
}

// ✅ Good: Handle errors appropriately
try {
  await database.addVector(id, vector);
} catch (error) {
  if (error instanceof ValidationError) {
    logger.warn('Invalid vector data:', error.message);
    throw error;
  }
  logger.error('Unexpected error:', error);
  throw new DatabaseError('Failed to add vector');
}

// ❌ Bad: Avoid magic numbers
const results = await db.search(query, { k: 10 }); // What does 10 mean?

// ✅ Good: Use named constants
const DEFAULT_SEARCH_RESULTS = 10;
const results = await db.search(query, { k: DEFAULT_SEARCH_RESULTS });
```

### File Organization

```typescript
// ✅ Good: Group related functionality
// src/search/hnsw_search.ts
export class HNSWSearch {
  // HNSW-specific search implementation
}

// src/search/lsh_search.ts
export class LSHSearch {
  // LSH-specific search implementation
}

// src/search/index.ts
export { HNSWSearch } from './hnsw_search';
export { LSHSearch } from './lsh_search';
export { UnifiedSearch } from './unified_search';

// ❌ Bad: Don't put everything in one file
// src/everything.ts - Contains HNSW, LSH, KNN, and more
```

## Testing

### Unit Tests

```typescript
// test/vector_db.test.ts
import { describe, it, expect, beforeEach } from '@jest/globals';
import { VectorDB } from '../src/vector/vector_db';

describe('VectorDB', () => {
  let db: VectorDB;

  beforeEach(() => {
    db = new VectorDB({
      vectorSize: 128,
      distanceMetric: 'cosine'
    });
  });

  describe('addVector', () => {
    it('should add a vector successfully', async () => {
      const vector = new Float32Array(128);
      const id = 'test-vector';

      await expect(db.addVector(id, vector)).resolves.toBeUndefined();
      expect(await db.getVector(id)).toEqual(vector);
    });

    it('should reject invalid vector dimensions', async () => {
      const invalidVector = new Float32Array(64); // Wrong size
      const id = 'invalid-vector';

      await expect(db.addVector(id, invalidVector))
        .rejects.toThrow('Vector dimension mismatch');
    });

    it('should handle duplicate IDs', async () => {
      const vector1 = new Float32Array(128);
      const vector2 = new Float32Array(128);
      const id = 'duplicate-id';

      await db.addVector(id, vector1);
      await expect(db.addVector(id, vector2))
        .rejects.toThrow('Vector ID already exists');
    });
  });

  describe('search', () => {
    beforeEach(async () => {
      // Setup test data
      const vectors = generateTestVectors(100);
      for (let i = 0; i < vectors.length; i++) {
        await db.addVector(`vec-${i}`, vectors[i]);
      }
    });

    it('should find nearest neighbors', async () => {
      const query = new Float32Array(128);
      const results = await db.search(query, { k: 5 });

      expect(results).toHaveLength(5);
      expect(results[0]).toHaveProperty('id');
      expect(results[0]).toHaveProperty('score');
      expect(results[0]).toHaveProperty('distance');
    });

    it('should respect k parameter', async () => {
      const query = new Float32Array(128);
      const results = await db.search(query, { k: 3 });

      expect(results).toHaveLength(3);
    });
  });
});
```

### Integration Tests

```typescript
// test/api/vectors.test.ts
import { describe, it, expect, beforeAll, afterAll } from '@jest/globals';
import { Server } from '../src/server';
import { Database } from '../src/database/database';

describe('Vectors API', () => {
  let server: Server;
  let baseUrl: string;

  beforeAll(async () => {
    server = new Server({
      port: 0, // Random port
      database: {
        vectorSize: 128
      }
    });

    await server.start();
    const address = server.getAddress();
    baseUrl = `http://localhost:${address.port}`;
  });

  afterAll(async () => {
    await server.stop();
  });

  describe('POST /api/vectors', () => {
    it('should add a vector successfully', async () => {
      const response = await fetch(`${baseUrl}/api/vectors`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          id: 'test-vector',
          vector: Array.from({ length: 128 }, () => Math.random()),
          metadata: { type: 'test' }
        })
      });

      expect(response.status).toBe(201);
      const result = await response.json();
      expect(result.success).toBe(true);
      expect(result.id).toBe('test-vector');
    });

    it('should handle validation errors', async () => {
      const response = await fetch(`${baseUrl}/api/vectors`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          id: 'invalid-vector',
          vector: [1, 2, 3], // Wrong dimension
          metadata: { type: 'test' }
        })
      });

      expect(response.status).toBe(400);
      const result = await response.json();
      expect(result.success).toBe(false);
      expect(result.error.code).toBe('VECTOR_DIMENSION_MISMATCH');
    });
  });
});
```

### Benchmark Tests

```typescript
// test/benchmarks/custom.benchmark.ts
import { Database } from '../../src/database/database';
import { createTimer } from '../test-helpers/benchmark-utils';

async function customBenchmark() {
  const db = new Database({
    vectorSize: 128,
    partitionCapacity: 10000
  });

  // Setup test data
  const testVectors = generateTestVectors(5000);
  await db.bulkAdd(testVectors.map((vec, i) => ({
    id: `vec-${i}`,
    vector: vec,
    metadata: { category: `cat-${i % 10}` }
  })));

  // Benchmark different search configurations
  const query = generateRandomVector(128);
  const configurations = [
    { name: 'Flat Search', options: { k: 10, useHNSW: false } },
    { name: 'HNSW Fast', options: { k: 10, useHNSW: true, efSearch: 32 } },
    { name: 'HNSW Balanced', options: { k: 10, useHNSW: true, efSearch: 100 } },
    { name: 'HNSW Accurate', options: { k: 10, useHNSW: true, efSearch: 200 } }
  ];

  console.log('Running custom benchmark...');
  console.table(
    await Promise.all(configurations.map(async (config) => {
      const timer = createTimer();
      timer.start();

      for (let i = 0; i < 100; i++) {
        await db.search(query, config.options);
      }

      timer.stop();
      return {
        Configuration: config.name,
        'Avg Time (ms)': (timer.total / 100).toFixed(2),
        'Total Time (ms)': timer.total.toFixed(2)
      };
    }))
  );
}
```

## Pull Request Process

### Branch Naming Convention

```bash
# Feature branches
feature/add-hnsw-optimization
feature/implement-lsh-index
feature/add-rest-api-endpoints

# Bug fix branches
fix/memory-leak-in-cache
fix/search-accuracy-issue
fix/api-validation-error

# Documentation branches
docs/update-api-reference
docs/add-contribution-guide
docs/improve-search-techniques

# Maintenance branches
chore/update-dependencies
chore/cleanup-unused-code
chore/improve-error-messages
```

### Commit Message Format

We follow [Conventional Commits](https://conventionalcommits.org/) specification:

```bash
# Format
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]

# Examples
feat(search): add LSH locality-sensitive hashing implementation

- Implement LSH index with configurable hash functions
- Add support for cosine and euclidean distance metrics
- Include performance benchmarks and documentation

Closes #123

fix(memory): resolve vector cache memory leak in PartitionedVectorDB

The cache was not properly cleaning up expired entries,
leading to unbounded memory growth over time.

fix(api): correct search result ordering in REST API

Results were being returned in arbitrary order instead of
distance-sorted order as documented.

BREAKING CHANGE: Search results now include distance field

docs(api): update search endpoint documentation

- Add examples for different search algorithms
- Document all available parameters and options
- Include error response examples

refactor(database): extract common functionality to base class

- Create AbstractDatabase base class
- Move shared methods to reduce code duplication
- Improve type safety with generics

test(search): add comprehensive HNSW search test suite

- Test various configurations and edge cases
- Include performance regression tests
- Add memory usage validation
```

### Pull Request Template

When creating a PR, please use this template:

```markdown
## Description

Brief description of the changes and the problem they solve.

## Changes Made

### Code Changes
- [ ] Added new feature/functionality
- [ ] Fixed bug/issue
- [ ] Refactored existing code
- [ ] Updated documentation
- [ ] Added tests

### Files Changed
- `src/search/hnsw.ts` - Added optimization
- `test/hnsw.test.ts` - Added test cases
- `docs/search-techniques.md` - Updated documentation

## Testing

### Test Coverage
- [ ] Unit tests pass (`npm test`)
- [ ] Integration tests pass (`npm run test:integration`)
- [ ] Benchmarks pass (`npm run benchmark`)
- [ ] Code coverage maintained (>90%)

### Manual Testing
- [ ] Tested with sample data
- [ ] Verified performance impact
- [ ] Checked edge cases

## Performance Impact

### Benchmarks
```
Operation          | Before   | After    | Change
-------------------|----------|----------|--------
HNSW Search (k=10) | 15.2ms   | 12.8ms   | +18% faster
Memory Usage       | 120MB    | 118MB    | -2% reduction
Index Build Time   | 45.2s    | 42.1s    | +7% faster
```

### Breaking Changes
- [ ] None
- [ ] Minor API changes (documented)
- [ ] Major API changes (migration guide needed)

## Documentation

- [ ] Updated API documentation
- [ ] Added code examples
- [ ] Updated README if needed
- [ ] Added migration guide for breaking changes

## Checklist

- [ ] Code follows project style guidelines
- [ ] All tests pass
- [ ] Documentation updated
- [ ] Performance benchmarks included
- [ ] No linting errors
- [ ] Commit messages follow conventional format
- [ ] PR description is clear and comprehensive

## Related Issues

Closes #123, #124
Related to #125

## Additional Notes

Any additional context, considerations, or follow-up work needed.
```

## Code Review Process

### Review Checklist for Reviewers

**Code Quality:**
- [ ] Code follows TypeScript best practices
- [ ] Proper error handling and validation
- [ ] No console.log statements in production code
- [ ] Functions have appropriate documentation
- [ ] Code is well-structured and readable

**Testing:**
- [ ] Unit tests cover new functionality
- [ ] Integration tests verify end-to-end behavior
- [ ] Edge cases are handled
- [ ] Performance tests included for performance-critical code

**Documentation:**
- [ ] API documentation updated
- [ ] Code comments added for complex logic
- [ ] Examples provided for new features
- [ ] Breaking changes documented

**Performance:**
- [ ] No performance regressions
- [ ] Memory usage considered
- [ ] Scalability implications reviewed

### Review Comments Guidelines

```typescript
// ✅ Good: Specific, actionable feedback
"The error handling here could be more specific. Consider throwing
ValidationError instead of generic Error for better client handling."

// ✅ Good: Suggest alternatives
"Instead of manual array iteration, consider using Array.find() for
better readability and performance."

// ❌ Bad: Unclear feedback
"This looks wrong"

// ❌ Bad: Demotivating
"This is a terrible implementation"
```

## Release Process

### Version Numbering

We follow [Semantic Versioning](https://semver.org/):

```
MAJOR.MINOR.PATCH

- MAJOR: Breaking changes
- MINOR: New features (backward compatible)
- PATCH: Bug fixes (backward compatible)
```

### Release Steps

1. **Preparation**
   ```bash
   # Ensure all tests pass
   npm test

   # Run benchmarks
   npm run benchmark

   # Update version
   npm version minor  # or patch, major
   ```

2. **Changelog Update**
   ```markdown
   ## [1.1.0] - 2024-01-15

   ### Added
   - LSH (Locality-Sensitive Hashing) implementation (#123)
   - REST API server with full CRUD operations (#124)
   - Vector compression with Product Quantization (#125)

   ### Fixed
   - Memory leak in HNSW index building (#126)
   - Search accuracy regression in clustered DB (#127)

   ### Performance
   - 25% faster search with new HNSW optimizations
   - 30% reduction in memory usage for large datasets
   ```

3. **Release**
   ```bash
   # Build production version
   npm run build

   # Create git tag
   git tag v1.1.0

   # Push to repository
   git push origin main --tags

   # Publish to NPM
   npm publish
   ```

4. **Post-Release**
   - Update documentation website
   - Announce release on GitHub
   - Update issue labels and milestones

## Documentation

### API Documentation

Use TSDoc comments for all public APIs:

```typescript
/**
 * Represents a vector database that supports various search algorithms
 * and indexing strategies for high-performance similarity search.
 *
 * @example
 * ```typescript
 * const db = new Database({
 *   vectorSize: 128,
 *   indexing: { buildOnStart: true }
 * });
 *
 * await db.addVector('user-1', embedding);
 * const results = await db.search(queryEmbedding, { k: 5 });
 * ```
 */
export class Database {
  /**
   * Creates a new Database instance with the specified configuration.
   *
   * @param config - Database configuration options
   * @throws {ValidationError} When configuration is invalid
   */
  constructor(config: DatabaseConfig) {
    // Implementation
  }

  /**
   * Adds a vector to the database with optional metadata.
   *
   * @param id - Unique identifier for the vector
   * @param vector - The vector data as Float32Array
   * @param metadata - Optional metadata associated with the vector
   * @returns Promise that resolves when vector is added
   * @throws {ValidationError} When vector dimensions don't match
   * @throws {DuplicateError} When ID already exists
   */
  async addVector(
    id: string,
    vector: Float32Array,
    metadata?: Record<string, any>
  ): Promise<void> {
    // Implementation
  }
}
```

### Performance Documentation

Include performance characteristics in documentation:

```typescript
/**
 * Searches for the k nearest neighbors using HNSW algorithm.
 *
 * Performance characteristics:
 * - Time Complexity: O(log n) average case
 * - Space Complexity: O(n × d + n × M) where M is max connections
 * - Accuracy: 95-99% depending on efSearch parameter
 * - Memory Usage: ~50MB per 100K vectors with M=16
 *
 * @param query - Query vector
 * @param options - Search options
 * @returns Search results sorted by distance
 */
async search(query: Float32Array, options: SearchOptions): Promise<SearchResult[]> {
  // Implementation
}
```

## Getting Help

### Communication Channels

- **GitHub Issues**: For bugs, feature requests, and general questions
- **GitHub Discussions**: For longer-form discussions and Q&A
- **Discord**: For real-time chat and community support
- **Documentation**: Check our comprehensive docs first

### Issue Reporting

When reporting bugs, please include:

1. **Clear Title**: Summarize the issue concisely
2. **Description**: Detailed description of the problem
3. **Steps to Reproduce**: Minimal code example that reproduces the issue
4. **Expected vs Actual**: What you expected vs what happened
5. **Environment**: Node.js version, OS, NBase version
6. **Logs**: Relevant error messages or logs

### Feature Requests

For feature requests, please include:

1. **Use Case**: Describe your specific use case
2. **Current Workaround**: How you currently solve this problem
3. **Proposed Solution**: Your suggested implementation
4. **Alternatives**: Other approaches you've considered

## Recognition

Contributors are recognized in several ways:

- **GitHub Contributors**: Listed in repository contributors
- **Changelog**: Mentioned in release notes
- **Documentation**: Featured in contributor acknowledgments
- **Community**: Highlighted in community discussions

## License

By contributing to NBase, you agree that your contributions will be licensed under the MIT License. All contributors must sign our Contributor License Agreement (CLA) before their contributions can be accepted.

---

Thank you for contributing to NBase! Your contributions help make vector databases more accessible and powerful for everyone. 🚀
