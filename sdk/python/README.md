# nbase Python SDK

Dependency-free Python client for an nbase `nbased` server (Python 3.8+).

```bash
pip install .
```

```python
from nbase import NBaseClient

db = NBaseClient("http://localhost:1307", api_key="secret")
db.add(1, [0.1, 0.2, 0.3], metadata={"kind": "example"}, text="hello world")
hits = db.search([0.1, 0.2, 0.3], k=5, includeMetadata=True)
print(hits)
```

Methods return the decoded response JSON. Non-2xx and network failures raise `NBaseError` (`status` and `payload` are available for HTTP errors). Search accepts server options such as `filters`, `metric`, `mode`, `maxDistance`, and `includeMetadata`.
