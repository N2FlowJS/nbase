# Contributing Guide

## Development Setup

### Prerequisites
- Node.js >=16
- Git
- TypeScript knowledge
- Vector database concepts understanding

### Local Development
```bash
# Clone repository
git clone https://github.com/n2flowjs/nbase.git

# Install dependencies
npm install

# Run tests
npm test

# Run benchmarks
npm run benchmark
```

## Project Structure

```plaintext
nbase/
├── src/
│   ├── ann/          # Approximate Nearest Neighbor algorithms
│   ├── database/     # Core database implementation
│   ├── search/       # Search engine implementations
│   ├── server/       # HTTP server & API
│   ├── types/        # TypeScript type definitions
│   └── utils/        # Utility functions
├── test/
│   ├── benchmarks/   # Performance benchmarks
│   └── unit/         # Unit tests
└── docs/            # Documentation
```

## Coding Standards

### TypeScript Guidelines
```typescript
// Use interfaces for public APIs
interface SearchOptions {
  k: number;
  includeMetadata?: boolean;
}

// Document complex functions
/**
 * Performs nearest neighbor search using HNSW algorithm.
 * @param query Query vector
 * @param k Number of results
 * @returns Nearest neighbors
 */
function findNearest(query: Vector, k: number): SearchResult[] {
  // Implementation
}
```

### Testing Requirements

1. Unit Tests Coverage
```typescript
describe('Database', () => {
  it('should add vectors correctly', async () => {
    // Test implementation
  });
  
  it('should handle errors gracefully', async () => {
    // Test implementation
  });
});
```

2. Benchmark Tests
```typescript
async function benchmarkSearch() {
  const iterations = 1000;
  console.time('search');
  // Benchmark implementation
  console.timeEnd('search');
}
```

## Pull Request Process

1. Branch Naming:
```bash
feature/add-new-index
fix/memory-leak
docs/update-api-docs
```

2. Commit Messages:
```bash
# Format
<type>(<scope>): <description>

# Examples
feat(search): add LSH index support
fix(memory): resolve vector cache leak
docs(api): update search parameters
```

3. PR Description Template:
```markdown
## Changes
- Added LSH index implementation
- Updated documentation
- Added unit tests

## Testing
- [ ] Unit tests pass
- [ ] Benchmarks run
- [ ] Memory usage verified

## Documentation
- Updated API docs
- Added implementation notes
```

## Release Process

1. Version Bump
```bash
npm version patch|minor|major
```

2. Changelog Update
```markdown
## [1.0.0] - 2024-01-01
### Added
- New LSH index implementation
### Fixed
- Memory leak in vector cache
```

3. Release Steps
```bash
# Build
npm run build

# Test
npm test

# Publish
npm publish
```

## Documentation

### API Documentation
- Use TSDoc comments
- Include examples
- Document edge cases

### Performance Documentation
- Include benchmark results
- Document memory usage
- Specify scaling limits

## Support

- GitHub Issues for bugs
- Discussions for questions
- Pull Requests for contributions

## License

- MIT License
- Contributors must sign CLA
