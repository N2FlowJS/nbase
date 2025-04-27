# Server API

## Overview

NBase provides a built-in HTTP server for accessing vector database functionality over REST APIs. The server includes rate limiting, CORS support, and comprehensive error handling.

## Quick Start

```typescript
import { Server } from '@n2flowjs/nbase';

const server = new Server({
  port: 1307,
  host: 'localhost',
  database: {
    vectorSize: 1536,
    indexing: { buildOnStart: true }
  }
});

server.start();
```

## Server Configuration

```typescript
interface ServerOptions {
  port?: number;              // Default: 1307
  host?: string;              // Default: localhost
  rateLimit?: {
    enable?: boolean;         // Default: true
    maxRequestsPerMinute?: number; // Default: 100
    windowMs?: number;        // Default: 60000
  };
  middleware?: RequestHandler[]; // Express middleware
  database?: DatabaseOptions;    // Database configuration
  debug?: boolean;              // Enable debug logging
}
```

## API Endpoints

### Health Check
```http
GET /health

Response:
{
  "status": "ok",
  "version": "0.1.3",
  "timestamp": "2024-01-01T00:00:00.000Z"
}
```

### Vector Operations

#### Add Vector
```http
POST /vectors
Content-Type: application/json

{
  "id": "doc1",
  "vector": [0.1, 0.2, ...],
  "metadata": {
    "title": "Example Document",
    "tags": ["example", "doc"]
  }
}

Response:
{
  "success": true,
  "vectorId": "doc1",
  "partitionId": "partition1"
}
```

#### Bulk Add Vectors
```http
POST /vectors/bulk
Content-Type: application/json

{
  "vectors": [
    {
      "id": "doc1",
      "vector": [0.1, 0.2, ...],
      "metadata": { ... }
    },
    // More vectors...
  ]
}

Response:
{
  "success": true,
  "count": 2,
  "partitionsAffected": ["partition1", "partition2"]
}
```

#### Search Vectors
```http
POST /search
Content-Type: application/json

{
  "query": [0.1, 0.2, ...],
  "k": 5,
  "includeMetadata": true,
  "useHNSW": true,
  "filters": {
    "tags": ["example"]
  }
}

Response:
{
  "results": [
    {
      "id": "doc1",
      "score": 0.95,
      "metadata": {
        "title": "Example Document",
        "tags": ["example", "doc"]
      }
    }
  ],
  "count": 1,
  "duration": 15
}
```

### Metadata Operations

#### Get Metadata
```http
GET /vectors/:id/metadata

Response:
{
  "metadata": {
    "title": "Example Document",
    "tags": ["example", "doc"]
  }
}
```

#### Update Metadata
```http
PATCH /vectors/:id/metadata
Content-Type: application/json

{
  "title": "Updated Title",
  "tags": ["updated", "doc"]
}

Response:
{
  "success": true
}
```

### Index Management

#### Build Indices
```http
POST /index/build

{
  "method": "hnsw",
  "options": {
    "M": 16,
    "efConstruction": 200
  }
}

Response:
{
  "success": true,
  "duration": 1500
}
```

### Database Management

#### Database Stats
```http
GET /stats

Response:
{
  "vectors": {
    "total": 1000,
    "inMemory": 500
  },
  "partitions": {
    "total": 2,
    "active": 1
  },
  "indices": {
    "type": "hnsw",
    "built": true
  },
  "memory": {
    "usage": "125MB",
    "available": "1GB"
  }
}
```

## Error Handling

The server uses standardized error responses:

```typescript
interface ErrorResponse {
  error: string;        // Error message
  status: number;       // HTTP status code
  code?: string;        // Error code
  details?: any;        // Additional details
  stack?: string;       // Stack trace (development only)
}
```

Common status codes:
- `400`: Bad Request
- `404`: Not Found
- `429`: Too Many Requests
- `500`: Internal Server Error

## Security

1. **Rate Limiting**
```typescript
const server = new Server({
  rateLimit: {
    enable: true,
    maxRequestsPerMinute: 100,
    windowMs: 60000
  }
});
```

2. **CORS Configuration**
```typescript
const server = new Server({
  middleware: [
    cors({
      origin: ['https://yourdomain.com'],
      methods: ['GET', 'POST']
    })
  ]
});
```

3. **Custom Middleware**
```typescript
const server = new Server({
  middleware: [
    // Authentication
    (req, res, next) => {
      const apiKey = req.headers['x-api-key'];
      if (!apiKey) return res.status(401).json({ error: 'Unauthorized' });
      next();
    },
    // Logging
    morgan('combined')
  ]
});
```

## Performance Monitoring

The server exposes metrics for monitoring:

```http
GET /metrics

Response:
{
  "requests": {
    "total": 1000,
    "perMinute": 16.7,
    "byEndpoint": {
      "/search": 800,
      "/vectors": 200
    }
  },
  "latency": {
    "p50": 15,
    "p95": 45,
    "p99": 75
  },
  "memory": {
    "heapUsed": "125MB",
    "heapTotal": "500MB"
  }
}
```

## Best Practices

1. **Rate Limiting**
   - Enable rate limiting in production
   - Set appropriate limits based on capacity
   - Implement client-side retry logic

2. **Error Handling**
   - Implement proper error handling on client
   - Monitor error rates
   - Use appropriate status codes

3. **Performance**
   - Enable compression
   - Use bulk operations when possible
   - Monitor server metrics
   - Scale based on metrics

4. **Security**
   - Use HTTPS in production
   - Implement authentication
   - Configure CORS appropriately
   - Regular security updates
