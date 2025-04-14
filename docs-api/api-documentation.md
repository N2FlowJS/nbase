# NBase API Documentation

This document provides detailed specifications for all API endpoints in the NBase vector database server.

## Table of Contents

- [Health and Status](#health-and-status)
  - [Health Check](#health-check)
  - [Database Statistics](#database-statistics)
- [Vector Management](#vector-management)
  - [Add Vector](#add-vector)
  - [Bulk Add Vectors](#bulk-add-vectors)
  - [Get Vector](#get-vector)
  - [Check Vector Exists](#check-vector-exists)
  - [Update Vector Metadata](#update-vector-metadata)
  - [Delete Vector](#delete-vector)
  - [Find Similar Vectors](#find-similar-vectors)
- [Vector Search](#vector-search)
  - [Search](#search)
  - [Extract Relationships](#extract-relationships)
  - [Extract Communities](#extract-communities)

## Health and Status

### Health Check

Get the current health status of the API server.

**URL:** `/health`

**Method:** `GET`

**Response:**

```json
{
  "status": "ok",
  "version": "1.0.0",
  "timestamp": "2023-04-01T12:34:56.789Z"
}
```

### Database Statistics

Get statistics about the current database state.

**URL:** `/stats`

**Method:** `GET`

**Response:**

```json
{
  "vectorCount": 1000,
  "dimensions": {
    "768": 800,
    "1536": 200
  },
  "partitionCount": 4,
  "partitionStats": [
    {
      "id": "partition1",
      "vectorCount": 300,
      "memoryUsage": 15000000
    },
    {
      "id": "partition2",
      "vectorCount": 700,
      "memoryUsage": 35000000
    }
  ],
  "indexStats": {
    "isIndexed": true,
    "indexType": "hnsw",
    "indexingProgress": 100
  },
  "memoryUsage": {
    "total": 50000000,
    "vectors": 48000000,
    "metadata": 2000000
  },
  "timestamps": {
    "created": "2023-04-01T10:00:00.000Z",
    "lastModified": "2023-04-01T12:30:00.000Z"
  }
}
```

## Vector Management

### Add Vector

Add a single vector to the database.

**URL:** `/api/vectors`

**Method:** `POST`

**Request Body:**

```json
{
  "id": "doc123",           // Optional: Custom ID for the vector
  "vector": [0.1, 0.2, ...], // Required: Vector embedding
  "metadata": {             // Optional: Metadata associated with the vector
    "title": "Example Document",
    "source": "website",
    "tags": ["example", "documentation"]
  }
}
```

**Response:**

```json
{
  "success": true,
  "count": 1,
  "ids": ["doc123"],
  "dimensions": {
    "1536": 1
  },
  "partitionsAffected": {
    "partition1": 1
  },
  "duration": 15
}
```

**Error Responses:**

- `400 Bad Request`: Missing or invalid vector
- `503 Service Unavailable`: Database is not ready

### Bulk Add Vectors

Add multiple vectors to the database in a single request.

**URL:** `/api/vectors`

**Method:** `POST`

**Request Body:**

```json
{
  "vectors": [
    {
      "id": "doc123",
      "vector": [0.1, 0.2, ...],
      "metadata": {
        "title": "Example Document 1",
        "source": "website"
      }
    },
    {
      "id": "doc456",
      "vector": [0.3, 0.4, ...],
      "metadata": {
        "title": "Example Document 2",
        "source": "pdf"
      }
    }
  ]
}
```

**Response:**

```json
{
  "success": true,
  "count": 2,
  "dimensions": {
    "1536": 2
  },
  "partitionsAffected": {
    "partition1": 1,
    "partition2": 1
  },
  "duration": 25
}
```

**Error Responses:**

- `400 Bad Request`: Missing or invalid vector array
- `503 Service Unavailable`: Database is not ready

### Get Vector

Retrieve a vector by its ID.

**URL:** `/api/vectors/:id`

**Method:** `GET`

**URL Parameters:**

- `id`: Vector ID

**Query Parameters:**

- `includeVector`: Include the vector in the response (`true` or `false`, default: `false`)
- `includeMetadata`: Include metadata in the response (`true` or `false`, default: `true`)

**Response:**

```json
{
  "success": true,
  "id": "doc123",
  "partitionId": "partition1",
  "dimension": 1536,
  "vector": [0.1, 0.2, ...],
  "metadata": {
    "title": "Example Document",
    "source": "website",
    "dimension": 1536,
    "createdAt": 1680346800000
  },
  "duration": 5
}
```

**Error Responses:**

- `404 Not Found`: Vector not found
- `503 Service Unavailable`: Database is not ready

### Check Vector Exists

Check if a vector with the specified ID exists in the database.

**URL:** `/api/vectors/:id/exists`

**Method:** `GET`

**URL Parameters:**

- `id`: Vector ID

**Response:**

```json
{
  "success": true,
  "exists": true,
  "id": "doc123",
  "foundId": "doc123",
  "dimension": 1536,
  "duration": 3
}
```

**Error Responses:**

- `503 Service Unavailable`: Database is not ready

### Update Vector Metadata

Update the metadata for a vector.

**URL:** `/api/vectors/:id/metadata`

**Method:** `PATCH`

**URL Parameters:**

- `id`: Vector ID

**Request Body:**

```json
{
  "metadata": {
    "title": "Updated Title",
    "tags": ["updated", "metadata"]
  },
  "operation": "merge"  // Optional: "merge" (default) or "replace"
}
```

**Response:**

```json
{
  "success": true,
  "id": "doc123",
  "operation": "merge",
  "dimension": 1536,
  "duration": 8
}
```

**Error Responses:**

- `400 Bad Request`: Missing or invalid metadata
- `404 Not Found`: Vector not found
- `503 Service Unavailable`: Database is not ready

### Delete Vector

Delete a vector from the database.

**URL:** `/api/vectors/:id`

**Method:** `DELETE`

**URL Parameters:**

- `id`: Vector ID

**Response:**

```json
{
  "success": true,
  "id": "doc123",
  "duration": 6
}
```

**Error Responses:**

- `404 Not Found`: Vector not found
- `503 Service Unavailable`: Database is not ready

### Find Similar Vectors

Find vectors similar to a specific vector.

**URL:** `/api/vectors/:id/similar`

**Method:** `GET`

**URL Parameters:**

- `id`: Vector ID of the reference vector

**Query Parameters:**

- `k`: Number of similar vectors to return (default: `10`)
- `includeMetadata`: Include metadata in results (`true` or `false`, default: `true`)
- `includeVectors`: Include vectors in results (`true` or `false`, default: `false`)

**Response:**

```json
{
  "success": true,
  "queryId": "doc123",
  "queryDimension": 1536,
  "results": [
    {
      "id": "doc456",
      "score": 0.92,
      "partitionId": "partition2",
      "metadata": {
        "title": "Similar Document 1",
        "source": "pdf"
      }
    },
    {
      "id": "doc789",
      "score": 0.85,
      "partitionId": "partition1",
      "metadata": {
        "title": "Similar Document 2",
        "source": "website"
      }
    }
  ],
  "count": 2,
  "duration": 12
}
```

**Error Responses:**

- `404 Not Found`: Source vector not found
- `503 Service Unavailable`: Database is not ready

## Vector Search

### Search

Search for similar vectors based on a query vector.

**URL:** `/api/search`

**Method:** `POST`

**Request Body:**

```json
{
  "query": [0.1, 0.2, ...],  // Required: Query vector
  "k": 10,                    // Optional: Number of results to return (default: 10)
  "method": "hnsw",          // Optional: Search method ("hnsw" or default clustered)
  "partitionIds": ["partition1", "partition2"],  // Optional: Specific partitions to search
  "efSearch": 100,           // Optional: Exploration factor for HNSW search
  "distanceMetric": "cosine", // Optional: Distance metric (default: cosine)
  "rerank": true,            // Optional: Whether to rerank results
  "rerankingMethod": "cross_encoder", // Optional: Method for reranking
  "rerankLambda": 0.5,       // Optional: Lambda parameter for reranking
  "filters": {               // Optional: Metadata filters
    "source": "website",
    "tags": "example"
  },
  "includeMetadata": true,   // Optional: Include metadata in results (default: true)
  "includeVectors": false,   // Optional: Include vectors in results (default: false)
  "skipCache": false,        // Optional: Bypass result caching (default: false)
  "searchTimeoutMs": 5000    // Optional: Search timeout in milliseconds
}
```

**Alternative Filter Format:**

```json
"filters": [
  { "field": "source", "operator": "$eq", "value": "website" },
  { "field": "rating", "operator": "$gt", "value": 4 }
]
```

**Response:**

```json
{
  "results": [
    {
      "id": "doc456",
      "score": 0.92,
      "partitionId": "partition2",
      "metadata": {
        "title": "Similar Document 1",
        "source": "website"
      }
    },
    {
      "id": "doc789",
      "score": 0.85,
      "partitionId": "partition1",
      "metadata": {
        "title": "Similar Document 2",
        "source": "website"
      }
    }
  ],
  "count": 2,
  "duration": 18,
  "searchOptions": {
    "k": 10,
    "method": "hnsw",
    "partitionsSearched": 2,
    "rerankApplied": true,
    "cacheUsed": true,
    "filtersApplied": true
  }
}
```

**Error Responses:**

- `400 Bad Request`: Missing or invalid query vector
- `500 Internal Server Error`: Search error (with details)

### Extract Relationships

Find relationships between vectors based on a distance threshold.

**URL:** `/api/search/relationships`

**Method:** `POST`

**Request Body:**

```json
{
  "threshold": 0.3,           // Required: Maximum distance between vectors to consider them related
  "metric": "cosine",         // Optional: Distance metric (e.g., 'cosine', 'euclidean'). Default depends on database implementation.
  "partitionIds": ["p1", "p2"] // Optional: Array of partition IDs to restrict the search. Searches all loaded partitions if omitted.
}
```

**Response:**

```json
{
  "relationships": [
    {
      "vector1": { "id": 123, "partitionId": "partition1" },
      "vector2": { "id": 456, "partitionId": "partition1" },
      "distance": 0.25
    },
    {
      "vector1": { "id": 789, "partitionId": "partition2" },
      "vector2": { "id": 101, "partitionId": "partition2" },
      "distance": 0.15
    }
    // ... more relationships
  ],
  "count": 2, // Total number of relationships found
  "duration": 345 // Time taken in milliseconds
}
```

**Error Responses:**

- `400 Bad Request`: Missing or invalid `threshold`
- `500 Internal Server Error`: Error during relationship extraction

### Extract Communities

Find communities (clusters) of related vectors based on a distance threshold.

**URL:** `/api/search/communities`

**Method:** `POST`

**Request Body:**

```json
{
  "threshold": 0.3,           // Required: Maximum distance between vectors to consider them related
  "metric": "cosine",         // Optional: Distance metric (e.g., 'cosine', 'euclidean'). Default depends on database implementation.
  "partitionIds": ["p1", "p2"], // Optional: Array of partition IDs to restrict the search. Searches all loaded partitions if omitted.
  "includeMetadata": true     // Optional: Whether to include metadata for each vector in the results (default: true)
}
```

**Response:**

```json
{
  "communities": [
    [
      { "id": 123, "partitionId": "partition1", "metadata": { "label": "doc1" } },
      { "id": 456, "partitionId": "partition1", "metadata": { "label": "doc2" } }
    ],
    [
      { "id": 789, "partitionId": "partition2", "metadata": { "label": "doc3" } },
      { "id": 101, "partitionId": "partition2", "metadata": { "label": "doc4" } },
      { "id": 102, "partitionId": "partition2", "metadata": { "label": "doc5" } }
    ]
    // ... more communities
  ],
  "count": 2, // Number of communities
  "totalVectors": 5, // Total number of vectors across all communities
  "duration": 345 // Time taken in milliseconds
}
```

**Error Responses:**

- `400 Bad Request`: Missing or invalid `threshold`
- `500 Internal Server Error`: Error during community extraction

## Filter Operators

The following operators are supported in filter configurations:

| Operator | Description | Example |
|----------|-------------|---------|
| `$eq` | Equal to | `{ "field": "type", "operator": "$eq", "value": "article" }` |
| `$ne` | Not equal to | `{ "field": "type", "operator": "$ne", "value": "draft" }` |
| `$gt` | Greater than | `{ "field": "rating", "operator": "$gt", "value": 4 }` |
| `$gte` | Greater than or equal to | `{ "field": "rating", "operator": "$gte", "value": 4 }` |
| `$lt` | Less than | `{ "field": "rating", "operator": "$lt", "value": 3 }` |
| `$lte` | Less than or equal to | `{ "field": "rating", "operator": "$lte", "value": 3 }` |
| `$in` | In array | `{ "field": "type", "operator": "$in", "value": ["article", "post"] }` |
| `$nin` | Not in array | `{ "field": "type", "operator": "$nin", "value": ["draft", "deleted"] }` |
| `$exists` | Field exists | `{ "field": "rating", "operator": "$exists", "value": true }` |
| `$regex` | Matches regex pattern | `{ "field": "title", "operator": "$regex", "value": "^Getting Started" }` |

## Best Practices

1. **Vector Dimensions**: Ensure vectors have consistent dimensions within a partition.

2. **Metadata Fields**: Use descriptive field names and consistent types for metadata.

3. **ID Management**: Consistently use either string or number IDs to avoid type conversion costs.

4. **Bulk Operations**: Use bulk add for better performance when adding multiple vectors.

5. **Partitioning**: Consider using separate partitions for vectors with different dimensions or from different sources.

6. **Search Performance**:
   - Use HNSW method for faster search at a slight cost in recall
   - Include proper filters to reduce the search space
   - Limit the number of partitions searched when possible

7. **Rate Limiting**: Be aware of server-side rate limiting settings to avoid request failures.
