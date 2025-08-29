# NBase Installation Guide

## System Requirements

### Minimum Requirements
- **Node.js**: v16.0.0 or higher
- **NPM**: v7.0.0 or higher (or Yarn v1.22.0+, PNPM v6.0.0+)
- **Memory**: 512MB RAM minimum, 2GB+ recommended
- **Storage**: 100MB for installation, additional space for data
- **Operating System**: Linux, macOS, or Windows

### Recommended Production Setup
- **Node.js**: v18 LTS or v20 LTS
- **Memory**: 8GB+ RAM
- **Storage**: SSD with 10GB+ free space
- **CPU**: 4+ cores
- **Network**: 1Gbps+ connection for high-throughput scenarios

### Supported Platforms
- **Linux**: Ubuntu 18.04+, CentOS 7+, Debian 9+
- **macOS**: 10.15+ (Catalina or later)
- **Windows**: Windows 10+, Windows Server 2019+
- **Docker**: Official Docker images available

## Installation Methods

### Method 1: NPM Installation (Recommended)

```bash
# Install the latest stable version
npm install @n2flowjs/nbase

# Install a specific version
npm install @n2flowjs/nbase@0.1.3

# Install as development dependency
npm install --save-dev @n2flowjs/nbase
```

### Method 2: Yarn Installation

```bash
# Install the latest version
yarn add @n2flowjs/nbase

# Install a specific version
yarn add @n2flowjs/nbase@0.1.3
```

### Method 3: PNPM Installation

```bash
# Install the latest version
pnpm add @n2flowjs/nbase

# Install a specific version
pnpm add @n2flowjs/nbase@0.1.3
```

### Method 4: From Source (Development)

```bash
# Clone the repository
git clone https://github.com/n2flowjs/nbase.git
cd nbase

# Install dependencies
npm install

# Build the project
npm run build

# Optional: Run tests
npm test

# Optional: Generate documentation
npm run docs
```

### Method 5: Docker Installation

```bash
# Pull the official Docker image
docker pull n2flowjs/nbase:latest

# Or build from source
docker build -t nbase .

# Run with Docker
docker run -p 1307:1307 -v /data:/app/data n2flowjs/nbase
```

## Post-Installation Setup

### Environment Configuration

Create a `.env` file in your project root:

```bash
# Database Configuration
NBASE_VECTOR_SIZE=128
NBASE_PARTITION_CAPACITY=10000
NBASE_MAX_ACTIVE_PARTITIONS=3
NBASE_DATA_PATH=./data

# Server Configuration (if using REST API)
NBASE_SERVER_PORT=1307
NBASE_SERVER_HOST=localhost

# Performance Tuning
NBASE_CACHE_SIZE=1000
NBASE_COMPRESSION_ENABLED=true
NBASE_COMPRESSION_ALGORITHM=product_quantization

# Monitoring
NBASE_MONITORING_ENABLED=true
NBASE_METRICS_INTERVAL=5000

# Logging
NBASE_LOG_LEVEL=info
NBASE_LOG_FORMAT=json

# Development
NODE_ENV=development
```

### TypeScript Configuration

If using TypeScript, add to your `tsconfig.json`:

```json
{
  "compilerOptions": {
    "moduleResolution": "node",
    "esModuleInterop": true,
    "allowSyntheticDefaultImports": true,
    "types": ["@n2flowjs/nbase"]
  }
}
```

## Verification & Testing

### Basic Installation Test

```typescript
import { Database } from '@n2flowjs/nbase';

async function testInstallation() {
  try {
    // Create a simple test database
    const db = new Database({
      vectorSize: 128,
      partitionCapacity: 1000
    });

    // Add a test vector
    const testVector = new Float32Array(128);
    for (let i = 0; i < 128; i++) {
      testVector[i] = Math.random();
    }

    await db.addVector('test-vector-1', testVector, {
      type: 'test',
      timestamp: new Date().toISOString()
    });

    // Perform a search
    const results = await db.search(testVector, { k: 1 });
    console.log('✅ Installation successful!');
    console.log(`Found ${results.length} result(s)`);

    // Clean up
    await db.close();

  } catch (error) {
    console.error('❌ Installation test failed:', error);
    process.exit(1);
  }
}

testInstallation();
```

### Server Installation Test

```typescript
import { Server } from '@n2flowjs/nbase';

async function testServerInstallation() {
  try {
    const server = new Server({
      port: 1307,
      database: {
        vectorSize: 128
      }
    });

    await server.start();
    console.log('✅ Server installation successful!');
    console.log('Server running on http://localhost:1307');

    // Test health endpoint
    const response = await fetch('http://localhost:1307/health');
    const health = await response.json();
    console.log('Health check:', health);

    await server.stop();

  } catch (error) {
    console.error('❌ Server installation test failed:', error);
    process.exit(1);
  }
}

testServerInstallation();
```

### Performance Benchmark Test

```typescript
import { Database } from '@n2flowjs/nbase';

async function performanceTest() {
  const db = new Database({
    vectorSize: 128,
    partitionCapacity: 10000
  });

  console.log('Running performance test...');

  // Add test vectors
  const startTime = Date.now();
  const numVectors = 1000;

  for (let i = 0; i < numVectors; i++) {
    const vector = new Float32Array(128);
    for (let j = 0; j < 128; j++) {
      vector[j] = Math.random();
    }
    await db.addVector(`vec-${i}`, vector);
  }

  const addTime = Date.now() - startTime;
  console.log(`✅ Added ${numVectors} vectors in ${addTime}ms`);
  console.log(`Average: ${(addTime / numVectors).toFixed(2)}ms per vector`);

  // Search test
  const queryVector = new Float32Array(128);
  for (let i = 0; i < 128; i++) {
    queryVector[i] = Math.random();
  }

  const searchStart = Date.now();
  const results = await db.search(queryVector, { k: 10 });
  const searchTime = Date.now() - searchStart;

  console.log(`✅ Search completed in ${searchTime}ms`);
  console.log(`Found ${results.length} results`);

  await db.close();
}

performanceTest();
```

## Troubleshooting

### Common Installation Issues

#### 1. Node.js Version Too Old
```bash
# Check Node.js version
node --version

# Update Node.js (using nvm)
nvm install 18
nvm use 18
```

#### 2. Memory Issues During Installation
```bash
# Increase Node.js memory limit
export NODE_OPTIONS="--max-old-space-size=4096"

# Or set in package.json scripts
{
  "scripts": {
    "build": "node --max-old-space-size=4096 ./node_modules/.bin/tsc"
  }
}
```

#### 3. Permission Issues
```bash
# Fix npm permissions (Linux/macOS)
sudo chown -R $(whoami) ~/.npm

# Or use a Node version manager
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
```

#### 4. Build Failures
```bash
# Clean install
rm -rf node_modules package-lock.json
npm cache clean --force
npm install

# Check for missing dependencies
npm ls --depth=0
```

#### 5. TypeScript Compilation Errors
```bash
# Check TypeScript version
npx tsc --version

# Update TypeScript
npm install --save-dev typescript@latest

# Check tsconfig.json
npx tsc --noEmit
```

### Platform-Specific Issues

#### Windows
```powershell
# Enable long paths
reg add "HKLM\SYSTEM\CurrentControlSet\Control\FileSystem" /v LongPathsEnabled /t REG_DWORD /d 1 /f

# Use Windows Subsystem for Linux (WSL) for better compatibility
wsl --install
```

#### macOS
```bash
# Install Xcode command line tools
xcode-select --install

# Update Homebrew and Node.js
brew update
brew upgrade node
```

#### Linux
```bash
# Install build dependencies
sudo apt-get update
sudo apt-get install build-essential python3-dev

# Or for CentOS/RHEL
sudo yum groupinstall 'Development Tools'
```

### Network Issues

#### Corporate Proxy
```bash
# Configure npm proxy
npm config set proxy http://proxy.company.com:8080
npm config set https-proxy http://proxy.company.com:8080

# Configure git proxy
git config --global http.proxy http://proxy.company.com:8080
```

#### Firewall Issues
```bash
# Test connectivity
curl -I https://registry.npmjs.org/

# Configure firewall rules
sudo ufw allow 1307/tcp  # For NBase server
```

## Docker Deployment

### Dockerfile
```dockerfile
FROM node:18-alpine AS base

# Install system dependencies
RUN apk add --no-cache python3 make g++

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci --only=production && npm cache clean --force

# Production stage
FROM node:18-alpine AS production

WORKDIR /app

# Copy installed dependencies
COPY --from=base /app/node_modules ./node_modules

# Copy application code
COPY . .

# Create data directory
RUN mkdir -p /app/data

# Expose port
EXPOSE 1307

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:1307/health || exit 1

# Start application
CMD ["npm", "start"]
```

### Docker Compose
```yaml
version: '3.8'

services:
  nbase:
    build: .
    ports:
      - "1307:1307"
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    environment:
      - NODE_ENV=production
      - NBASE_VECTOR_SIZE=128
      - NBASE_PARTITION_CAPACITY=50000
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:1307/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

## Next Steps

After successful installation:

1. **Read the Getting Started Guide**: Learn basic usage patterns
2. **Explore API Documentation**: Understand available methods and options
3. **Review Configuration**: Optimize settings for your use case
4. **Run Benchmarks**: Test performance with your data
5. **Set Up Monitoring**: Configure logging and metrics collection

## Support

If you encounter issues:

1. Check the [GitHub Issues](https://github.com/n2flowjs/nbase/issues) page
2. Review the [Troubleshooting Guide](troubleshooting.md)
3. Join our [Discord Community](https://discord.gg/nbase)
4. Contact [Support](mailto:support@n2flowjs.com)

## Version Compatibility

| NBase Version | Node.js | TypeScript | Status |
|---------------|---------|------------|--------|
| 0.1.x | 16+ | 4.5+ | Current |
| 0.0.x | 14+ | 4.0+ | Legacy |

Always use the latest stable version for best performance and security.
