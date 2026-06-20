#!/usr/bin/env python3
import json
import time
import urllib.request

BASE = "http://localhost:17843"

TOML = """version = 1

[[sequences]]
protein.id = "A"
protein.sequence = "MTEYKLVVVGAGGVGKSALTIQLIQ"
"""

def post(path, data=None):
    body = json.dumps(data).encode() if data else None
    req = urllib.request.Request(
        BASE + path,
        data=body,
        headers={"Content-Type": "application/json"} if body else {},
        method="POST" if body else "GET",
    )
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())

def get(path):
    with urllib.request.urlopen(BASE + path) as r:
        return json.loads(r.read())

print("health:", get("/health"))

print("submitting job...")
res = post("/predict", {"toml": TOML})
job_id = res["job_id"]
print("job_id:", job_id)

while True:
    status = get(f"/jobs/{job_id}")
    print("status:", status["status"])
    if status["status"] in ("completed", "failed"):
        break
    time.sleep(10)

if status["status"] == "completed":
    result = get(f"/jobs/{job_id}/result")
    print("confidence:", result.get("confidence"))
    if "structures" in result:
        for k, v in result["structures"].items():
            print(f"structure [{k}]: {len(v)} chars")
else:
    print("error:", status.get("error"))
