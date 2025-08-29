# NBase REST API Server

## Overview

NBase provides a high-performance REST API server built on Express.js, designed for production vector database operations. The server offers comprehensive HTTP endpoints for all database operations, with built-in rate limiting, CORS support, request validation, and monitoring capabilities.

## Quick Start

### Basic Server Setup

```typescript
import { Server } from '@n2flowjs/nbase';

const server = new Server({
  port: 1307,
  host: 'localhost',
  database: {
    vectorSize: 1536,
    partitionCapacity: 10000,
    indexing: {
      buildOnStart: true,
      indexType: 'hnsw'
    }
  }
});

await server.start();
console.log('NBase server running on http://localhost:1307');
```

### Advanced Configuration

```typescript
const server = new Server({
  port: 1307,
  host: '0.0.0.0',
  rateLimit: {
    enable: true,
    maxRequestsPerMinute: 1000,
    windowMs: 60000
  },
  cors: {
    enable: true,
    origin: ['https://yourapp.com'],
    credentials: true
  },
  monitoring: {
    enable: true,
    metricsInterval: 5000
  },
  database: {
    vectorSize: 128,
    partitionCapacity: 50000,
    maxActivePartitions: 5,
    compression: {
      enabled: true,
      algorithm: 'product_quantization'
    }
  }
});
```

## Server Configuration

### Complete Configuration Options

```typescript
interface ServerConfig {
  // Network settings
  port?: number;              // Default: 1307
  host?: string;              // Default: 'localhost'
  https?: {
    enable: boolean;
    keyPath?: string;
    certPath?: string;
  };

  // Rate limiting
  rateLimit?: {
    enable?: boolean;         // Default: true
    maxRequestsPerMinute?: number; // Default: 100
    windowMs?: number;        // Default: 60000
    skipSuccessfulRequests?: boolean;
    skipFailedRequests?: boolean;
  };

  // CORS settings
  cors?: {
    enable?: boolean;         // Default: true
    origin?: string | string[];
    methods?: string[];
    credentials?: boolean;
    maxAge?: number;
  };

  // Monitoring
  monitoring?: {
    enable?: boolean;         // Default: false
    metricsInterval?: number; // Default: 5000ms
    enablePrometheus?: boolean;
  };

  // Database configuration
  database?: DatabaseOptions;

  // Express middleware
  middleware?: RequestHandler[];

  // Logging
  logging?: {
    level?: 'error' | 'warn' | 'info' | 'debug';
    format?: 'json' | 'simple';
    enableAccessLogs?: boolean;
  };

  // Security
  security?: {
    trustProxy?: boolean;
    helmet?: boolean;
    compression?: boolean;
  };

  // Development
  debug?: boolean;
}
```

## API Endpoints

### Health & Status

#### Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "0.1.3",
  "timestamp": "2024-01-01T12:00:00.000Z",
  "uptime": 3600,
  "database": {
    "status": "connected",
    "vectors": 15000,
    "partitions": 3
  }
}
```

#### Server Information
```http
GET /info
```

**Response:**
```json
{
  "server": {
    "version": "0.1.3",
    "nodeVersion": "v18.17.0",
    "platform": "linux",
    "architecture": "x64"
  },
  "database": {
    "vectorSize": 128,
    "partitionCapacity": 10000,
    "activePartitions": 2,
    "compression": {
      "enabled": true,
      "algorithm": "product_quantization"
    }
  },
  "configuration": {
    "rateLimit": {
      "enabled": true,
      "maxRequestsPerMinute": 1000
    },
    "cors": {
      "enabled": true
    }
  }
}
```

### Vector Operations

#### Add Single Vector
```http
POST /api/vectors
Content-Type: application/json
```

**Request Body:**
```json
{
  "id": "doc_123",
  "vector": [0.1, 0.2, 0.3, ..., 0.128],
  "metadata": {
    "title": "Sample Document",
    "category": "documentation",
    "tags": ["sample", "test"],
    "created_at": "2024-01-01T00:00:00Z"
  }
}
```

**Response:**
```json
{
  "success": true,
  "id": "doc_123",
  "partitionId": "p-1749282892289",
  "timestamp": "2024-01-01T00:00:00.000Z"
}
```

#### Bulk Add Vectors
```http
POST /api/vectors/bulk
Content-Type: application/json
```

**Request Body:**
```json
{
  "vectors": [
    {
      "id": "doc_1",
      "vector": [0.1, 0.2, ...],
      "metadata": { "category": "docs" }
    },
    {
      "id": "doc_2",
      "vector": [0.3, 0.4, ...],
      "metadata": { "category": "tutorials" }
    }
  ],
  "options": {
    "skipDuplicates": true,
    "batchSize": 1000
  }
}
```

**Response:**
```json
{
  "success": true,
  "count": 2,
  "processed": 2,
  "failed": 0,
  "partitions": ["p-1749282892289", "p-1749282892290"],
  "duration": 45,
  "errors": []
}
```

#### Search Vectors
```http
POST /api/search
Content-Type: application/json
```

**Request Body:**
```json
{
  "query": [0.1, 0.2, 0.3, ..., 0.128],
  "k": 10,
  "options": {
    "algorithm": "hnsw",
    "efSearch": 100,
    "distanceMetric": "cosine",
    "includeMetadata": true,
    "includeVectors": false,
    "includeScores": true
  },
  "filter": {
    "category": "documentation",
    "tags": ["sample"]
  }
}
```

**Response:**
```json
{
  "results": [
    {
      "id": "doc_123",
      "score": 0.95,
      "distance": 0.05,
      "metadata": {
        "title": "Sample Document",
        "category": "documentation",
        "tags": ["sample", "test"]
      }
    },
    {
      "id": "doc_456",
      "score": 0.89,
      "distance": 0.11,
      "metadata": {
        "title": "Another Document",
        "category": "documentation"
      }
    }
  ],
  "count": 2,
  "totalFound": 2,
  "algorithm": "hnsw",
  "duration": 12,
  "partitionStats": {
    "searched": 2,
    "skipped": 1
  }
}
```

#### Batch Search
```http
POST /api/search/batch
Content-Type: application/json
```

**Request Body:**
```json
{
  "queries": [
    {
      "query": [0.1, 0.2, ...],
      "k": 5,
      "filter": { "category": "docs" }
    },
    {
      "query": [0.3, 0.4, ...],
      "k": 3,
      "filter": { "category": "tutorials" }
    }
  ],
  "options": {
    "maxBatchSize": 50,
    "prioritizeOrder": true,
    "timeout": 5000
  }
}
```

**Response:**
```json
{
  "batch": [
    {
      "results": [/* results for query 1 */],
      "count": 5,
      "duration": 8
    },
    {
      "results": [/* results for query 2 */],
      "count": 3,
      "duration": 6
    }
  ],
  "totalQueries": 2,
  "totalDuration": 14,
  "success": true
}
```

### Vector Management

#### Get Vector
```http
GET /api/vectors/{id}
```

**Response:**
```json
{
  "id": "doc_123",
  "vector": [0.1, 0.2, ...],
  "metadata": {
    "title": "Sample Document"
  },
  "partitionId": "p-1749282892289",
  "created_at": "2024-01-01T00:00:00Z",
  "updated_at": "2024-01-01T00:00:00Z"
}
```

#### Update Vector
```http
PUT /api/vectors/{id}
Content-Type: application/json
```

**Request Body:**
```json
{
  "vector": [0.2, 0.3, ...],  // Optional: update vector
  "metadata": {               // Optional: update metadata
    "title": "Updated Document",
    "version": 2
  }
}
```

**Response:**
```json
{
  "success": true,
  "id": "doc_123",
  "updated": ["vector", "metadata"],
  "timestamp": "2024-01-01T01:00:00.000Z"
}
```

#### Delete Vector
```http
DELETE /api/vectors/{id}
```

**Response:**
```json
{
  "success": true,
  "id": "doc_123",
  "partitionId": "p-1749282892289"
}
```

#### Get Vector Metadata
```http
GET /api/vectors/{id}/metadata
```

**Response:**
```json
{
  "metadata": {
    "title": "Sample Document",
    "category": "documentation",
    "tags": ["sample", "test"],
    "created_at": "2024-01-01T00:00:00Z"
  }
}
```

### Index Management

#### Build Index
```http
POST /api/index/build
Content-Type: application/json
```

**Request Body:**
```json
{
  "indexType": "hnsw",
  "partitions": ["p-1749282892289"],  // Optional: specific partitions
  "options": {
    "M": 16,
    "efConstruction": 200,
    "maxLevel": 10
  },
  "progressCallback": true  // Enable progress tracking
}
```

**Response:**
```json
{
  "success": true,
  "indexType": "hnsw",
  "partitionsProcessed": 1,
  "duration": 15432,
  "indexStats": {
    "nodes": 10000,
    "connections": 160000,
    "memoryUsage": "25MB"
  }
}
```

#### Index Status
```http
GET /api/index/status
```

**Response:**
```json
{
  "indices": {
    "p-1749282892289": {
      "type": "hnsw",
      "built": true,
      "lastBuilt": "2024-01-01T00:00:00Z",
      "stats": {
        "nodes": 10000,
        "memoryUsage": "25MB"
      }
    }
  },
  "pendingBuilds": 0
}
```

#### Delete Index
```http
DELETE /api/index
```

**Request Body:**
```json
{
  "partitions": ["p-1749282892289"]
}
```

### Database Management

#### Database Statistics
```http
GET /api/stats
```

**Response:**
```json
{
  "database": {
    "totalVectors": 25000,
    "totalPartitions": 3,
    "activePartitions": 2,
    "memoryUsage": "450MB",
    "diskUsage": "1.2GB"
  },
  "partitions": [
    {
      "id": "p-1749282892289",
      "vectorCount": 10000,
      "memoryUsage": "180MB",
      "lastAccessed": "2024-01-01T12:00:00Z",
      "indices": {
        "hnsw": {
          "built": true,
          "nodes": 10000
        }
      }
    }
  ],
  "performance": {
    "avgSearchTime": 15.5,
    "avgInsertTime": 2.1,
    "cacheHitRate": 0.85,
    "queriesPerSecond": 64.2
  }
}
```

#### Database Backup
```http
POST /api/database/backup
Content-Type: application/json
```

**Request Body:**
```json
{
  "path": "/backups/nbase-backup-2024-01-01",
  "includeIndices": true,
  "compress": true
}
```

**Response:**
```json
{
  "success": true,
  "backupPath": "/backups/nbase-backup-2024-01-01",
  "duration": 45000,
  "size": "1.2GB",
  "compressed": true
}
```

#### Database Restore
```http
POST /api/database/restore
Content-Type: application/json
```

**Request Body:**
```json
{
  "path": "/backups/nbase-backup-2024-01-01",
  "overwrite": false
}
```

### Monitoring & Metrics

#### Performance Metrics
```http
GET /api/metrics
```

**Response:**
```json
{
  "timestamp": "2024-01-01T12:00:00.000Z",
  "requests": {
    "total": 15432,
    "perSecond": 4.2,
    "perMinute": 252,
    "byEndpoint": {
      "/api/search": 12000,
      "/api/vectors": 3000,
      "/api/stats": 432
    },
    "byMethod": {
      "GET": 8000,
      "POST": 7000,
      "PUT": 400,
      "DELETE": 32
    }
  },
  "latency": {
    "p50": 12,
    "p95": 45,
    "p99": 120,
    "avg": 18.5
  },
  "errors": {
    "total": 23,
    "rate": 0.0015,
    "byType": {
      "400": 15,
      "404": 5,
      "500": 3
    }
  },
  "memory": {
    "heapUsed": "450MB",
    "heapTotal": "1GB",
    "external": "50MB",
    "rss": "600MB"
  },
  "database": {
    "activeConnections": 45,
    "queueLength": 2,
    "cacheHitRate": 0.85
  }
}
```

#### Health Metrics
```http
GET /api/health/detailed
```

**Response:**
```json
{
  "status": "healthy",
  "checks": {
    "database": {
      "status": "healthy",
      "responseTime": 5,
      "lastCheck": "2024-01-01T12:00:00Z"
    },
    "memory": {
      "status": "healthy",
      "usage": 0.45,
      "threshold": 0.9
    },
    "disk": {
      "status": "healthy",
      "usage": 0.6,
      "threshold": 0.95
    },
    "cpu": {
      "status": "healthy",
      "usage": 0.35,
      "threshold": 0.8
    }
  },
  "version": "0.1.3",
  "uptime": 86400
}
```

## Error Handling

### Standardized Error Responses

```typescript
interface ErrorResponse {
  success: false;
  error: {
    message: string;
    code: string;
    status: number;
    details?: any;
    timestamp: string;
    requestId: string;
  };
}
```

### Common Error Codes

| Status Code | Error Code | Description |
|-------------|------------|-------------|
| 400 | `VALIDATION_ERROR` | Invalid request data |
| 400 | `VECTOR_DIMENSION_MISMATCH` | Vector dimension doesn't match database |
| 404 | `VECTOR_NOT_FOUND` | Vector ID not found |
| 404 | `PARTITION_NOT_FOUND` | Partition not found |
| 409 | `VECTOR_EXISTS` | Vector ID already exists |
| 429 | `RATE_LIMIT_EXCEEDED` | Too many requests |
| 500 | `INTERNAL_ERROR` | Server internal error |
| 503 | `SERVICE_UNAVAILABLE` | Service temporarily unavailable |

### Error Examples

```json
{
  "success": false,
  "error": {
    "message": "Vector dimension mismatch. Expected 128, got 64",
    "code": "VECTOR_DIMENSION_MISMATCH",
    "status": 400,
    "details": {
      "expected": 128,
      "received": 64
    },
    "timestamp": "2024-01-01T12:00:00.000Z",
    "requestId": "req_abc123"
  }
}
```

## Security Configuration

### Rate Limiting
```typescript
const server = new Server({
  rateLimit: {
    enable: true,
    maxRequestsPerMinute: 1000,
    windowMs: 60000,
    skipSuccessfulRequests: false,
    skipFailedRequests: true
  }
});
```

### CORS Configuration
```typescript
const server = new Server({
  cors: {
    enable: true,
    origin: [
      'https://yourapp.com',
      'https://app.yourapp.com'
    ],
    methods: ['GET', 'POST', 'PUT', 'DELETE'],
    credentials: true,
    maxAge: 86400
  }
});
```

### Authentication Middleware
```typescript
const server = new Server({
  middleware: [
    // API Key authentication
    (req, res, next) => {
      const apiKey = req.headers['x-api-key'];
      if (!apiKey || !validApiKeys.includes(apiKey)) {
        return res.status(401).json({
          success: false,
          error: {
            message: 'Invalid API key',
            code: 'AUTHENTICATION_FAILED',
            status: 401
          }
        });
      }
      next();
    }
  ]
});
```

### HTTPS Configuration
```typescript
const server = new Server({
  https: {
    enable: true,
    keyPath: '/path/to/private-key.pem',
    certPath: '/path/to/certificate.pem'
  }
});
```

## Performance Optimization

### Server Tuning

#### High-Throughput Configuration
```typescript
const highThroughputConfig = {
  rateLimit: {
    maxRequestsPerMinute: 10000,
    windowMs: 60000
  },
  monitoring: {
    enable: true,
    metricsInterval: 1000
  },
  database: {
    maxActivePartitions: 10,
    cacheSize: 5000,
    compression: {
      enabled: true,
      algorithm: 'product_quantization'
    }
  }
};
```

#### Low-Latency Configuration
```typescript
const lowLatencyConfig = {
  rateLimit: {
    maxRequestsPerMinute: 5000
  },
  database: {
    maxActivePartitions: 3,
    partitionCapacity: 25000,
    cacheSize: 10000
  },
  monitoring: {
    enable: true,
    enablePrometheus: true
  }
};
```

### Monitoring Integration

#### Prometheus Metrics
```typescript
const server = new Server({
  monitoring: {
    enable: true,
    enablePrometheus: true,
    metricsInterval: 5000
  }
});

// Access metrics at /metrics for Prometheus scraping
```

#### Custom Monitoring
```typescript
const server = new Server({
  monitoring: {
    enable: true
  }
});

// Listen to server events
server.on('request:complete', (event) => {
  console.log(`Request ${event.method} ${event.path} took ${event.duration}ms`);
});

server.on('error', (error) => {
  console.error('Server error:', error);
});
```

## Production Deployment

### Docker Configuration
```dockerfile
FROM node:18-alpine

WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production

COPY . .
EXPOSE 1307

CMD ["npm", "start"]
```

### Environment Variables
```bash
# Server configuration
PORT=1307
HOST=0.0.0.0
NODE_ENV=production

# Database configuration
VECTOR_SIZE=128
PARTITION_CAPACITY=50000
MAX_ACTIVE_PARTITIONS=5

# Security
API_KEYS=key1,key2,key3
CORS_ORIGINS=https://yourapp.com

# Monitoring
ENABLE_METRICS=true
METRICS_INTERVAL=5000
```

### Health Checks
```yaml
# Kubernetes readiness probe
readinessProbe:
  httpGet:
    path: /health
    port: 1307
  initialDelaySeconds: 30
  periodSeconds: 10

# Kubernetes liveness probe
livenessProbe:
  httpGet:
    path: /health
    port: 1307
  initialDelaySeconds: 60
  periodSeconds: 30
```

This comprehensive REST API server documentation covers all major endpoints, configuration options, security features, and performance optimization strategies for production deployment.
