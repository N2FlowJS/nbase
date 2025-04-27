# Installation Guide

## Requirements

- Node.js >= 16.0.0
- NPM >= 7.0.0
- Memory: 512MB minimum, 2GB+ recommended
- Storage: 100MB minimum for installation

## Installation Methods

### NPM Installation
```bash
npm install @n2flowjs/nbase
```

### Yarn Installation
```bash
yarn add @n2flowjs/nbase
```

### From Source
```bash
git clone https://github.com/n2flowjs/nbase.git
cd nbase
npm install
npm run build
```

## Verification

```typescript
import { Database } from '@n2flowjs/nbase';

async function testInstallation() {
  const db = new Database({
    vectorSize: 128
  });
  
  await db.addVector('test', new Float32Array(128));
  console.log('Installation successful!');
}
```

## Common Issues

### Memory Allocation
```bash
# Increase Node.js memory limit
export NODE_OPTIONS=--max_old_space_size=4096
```

### Build Issues
```bash
# Clean install
rm -rf node_modules
npm cache clean --force
npm install
```

## Environment Setup

### Development
```json
{
  "NODE_ENV": "development",
  "NBASE_DB_PATH": "./data",
  "NBASE_LOG_LEVEL": "debug"
}
```

### Production
```json
{
  "NODE_ENV": "production",
  "NBASE_DB_PATH": "/var/lib/nbase",
  "NBASE_LOG_LEVEL": "info",
  "NBASE_MONITOR_ENABLED": "true"
}
```
```

### Next Steps

1. Check out the [Quick Start Guide](getting-started.md)
2. Review [Best Practices](best-practices.md)
3. Explore [API Documentation](api-reference.md)
