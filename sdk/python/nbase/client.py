"""Small, dependency-free client for an nbase REST server."""

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class NBaseError(Exception):
    """An HTTP or transport error returned by nbase."""

    def __init__(self, message, status=None, payload=None):
        super().__init__(message)
        self.status = status
        self.payload = payload


class NBaseClient:
    """Connect to ``nbased`` and access vectors, metadata, and search."""

    def __init__(self, url="http://127.0.0.1:3000", api_key=None, timeout=30):
        self.url = url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def _request(self, method, path, body=None):
        data = None if body is None else json.dumps(body, separators=(",", ":")).encode()
        headers = {"Accept": "application/json"}
        if data is not None:
            headers["Content-Type"] = "application/json"
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        req = Request(self.url + path, data=data, headers=headers, method=method)
        try:
            with urlopen(req, timeout=self.timeout) as response:
                raw = response.read()
        except HTTPError as exc:
            raw = exc.read()
            try:
                payload = json.loads(raw) if raw else {}
            except (ValueError, UnicodeDecodeError):
                payload = {"error": raw.decode(errors="replace")}
            message = payload.get("error", payload.get("message", str(exc))) if isinstance(payload, dict) else str(exc)
            raise NBaseError(message, exc.code, payload) from exc
        except (URLError, TimeoutError) as exc:
            raise NBaseError(str(exc)) from exc
        if not raw:
            return None
        try:
            return json.loads(raw)
        except (ValueError, UnicodeDecodeError) as exc:
            raise NBaseError("Server returned invalid JSON") from exc

    def health(self):
        return self._request("GET", "/health")

    def stats(self):
        return self._request("GET", "/stats")

    def add(self, id, vector, metadata=None, text=None):
        body = {"id": id, "vector": vector}
        if metadata is not None:
            body["metadata"] = metadata
        if text is not None:
            body["text"] = text
        return self._request("POST", "/api/vectors", body)

    def upsert(self, id, vector, metadata=None, text=None):
        body = {"id": id, "vector": vector}
        if metadata is not None:
            body["metadata"] = metadata
        if text is not None:
            body["text"] = text
        return self._request("POST", "/api/vectors/upsert", body)

    def add_many(self, vectors, upsert=False, threads=None):
        body = {"vectors": vectors}
        if threads is not None:
            body["threads"] = threads
        path = "/api/vectors/upsert" if upsert else "/api/vectors"
        return self._request("POST", path, body)

    def get(self, id, include_vector=True, include_metadata=True, include_text=False):
        from urllib.parse import urlencode
        query = urlencode({"includeVector": str(include_vector).lower(), "includeMetadata": str(include_metadata).lower(), "includeText": str(include_text).lower()})
        return self._request("GET", f"/api/vectors/{id}?{query}")

    def get_many(self, ids, include_vectors=True, include_metadata=True, include_text=False):
        return self._request("POST", "/api/vectors/get", {"ids": ids, "includeVectors": include_vectors, "includeMetadata": include_metadata, "includeText": include_text})

    def delete(self, id):
        return self._request("DELETE", f"/api/vectors/{id}")

    def delete_many(self, ids):
        return self._request("POST", "/api/vectors/delete", {"ids": ids})

    def count(self, filters=None):
        return self._request("GET", "/api/vectors/count") if filters is None else self._request("POST", "/api/vectors/count", {"filters": filters})

    def search(self, query, k=10, **options):
        return self._request("POST", "/api/search", {"query": query, "k": k, **options})

    def search_batch(self, queries, k=10, **options):
        return self._request("POST", "/api/search/batch", {"queries": queries, "k": k, **options})

    def search_text(self, query, k=10, **options):
        return self._request("POST", "/api/search/text", {"query": query, "k": k, **options})

    def search_hybrid(self, query, text, k=10, **options):
        return self._request("POST", "/api/search/hybrid", {"query": query, "text": text, "k": k, **options})

    def set_metadata(self, id, metadata, update=False):
        method = "PATCH" if update else "POST"
        return self._request(method, "/api/vectors/metadata", {"items": [{"id": id, "metadata": metadata}]})

    def set_text(self, id, text):
        return self._request("POST", "/api/vectors/text", {"items": [{"id": id, "text": text}]})

    def close(self):
        """Provided for parity with clients that own persistent connections."""
        return None

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()
