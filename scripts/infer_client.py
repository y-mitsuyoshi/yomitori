#!/usr/bin/env python3
"""Simple test client for the Yomitori inference server."""
import base64
import json
import sys
import urllib.request

image_path = sys.argv[1] if len(sys.argv) > 1 else "data/samples/sample_license.jpg"
doc_type = sys.argv[2] if len(sys.argv) > 2 else None

with open(image_path, "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

payload = {"image": b64}
if doc_type:
    payload["document_type"] = doc_type

req = urllib.request.Request(
    "http://localhost:8080/invocations",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)

try:
    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read())
        print(json.dumps(result, ensure_ascii=False, indent=2))
except urllib.error.HTTPError as e:
    print(f"HTTP Error {e.code}: {e.read().decode()}", file=sys.stderr)
    sys.exit(1)
except urllib.error.URLError as e:
    print(f"Connection Error: {e}", file=sys.stderr)
    print("Is the server running? Start it with: docker compose up serve", file=sys.stderr)
    sys.exit(1)