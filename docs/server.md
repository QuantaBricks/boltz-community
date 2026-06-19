# Boltz Prediction Server

A local FastAPI server that accepts TOML prediction requests and dispatches them to `boltz predict`.

## Setup

```bash
uv sync --extra server
```

## Start

```bash
# Real GPU mode
uv run python server.py

# Mock mode (no GPU required, returns fake results after ~3 s)
BOLTZ_MOCK=1 uv run python server.py

# Custom host / port
uv run python server.py --host 0.0.0.0 --port 8080
```

Default address: `http://127.0.0.1:17843`

---

## API

### `GET /health`

```bash
curl http://localhost:17843/health
# {"status":"ok"}
```

---

### `POST /predict`

Submit a prediction job. Returns `202` immediately with a `job_id`.

| Field | Type | Default | Description |
|---|---|---|---|
| `toml` | string | required | Boltz input in TOML format (see below) |
| `extra_args` | string | `""` | Extra flags forwarded to `boltz predict` |

```bash
curl -X POST http://localhost:17843/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "toml": "version = 1\n\n[[sequences]]\nprotein.id = \"A\"\nprotein.sequence = \"MVTPEG...\"\n\n[[sequences]]\nligand.id = \"B\"\nligand.smiles = \"N[C@@H](Cc1ccc(O)cc1)C(=O)O\"\n\n[[properties]]\naffinity.binder = \"B\"\n"
  }'
# {"job_id": "6c45ef27...", "status": "pending"}
```

---

### `GET /jobs/{job_id}`

Poll job status. Status flow: `pending` → `running` → `completed` | `failed`

```bash
curl http://localhost:17843/jobs/6c45ef27...
```

```json
{
  "job_id": "6c45ef27...",
  "status": "running",
  "created_at": 1750000000.0,
  "updated_at": 1750000003.5
}
```

On failure an `"error"` field is added with the reason.

---

### `GET /jobs/{job_id}/result`

Fetch results when the job is `completed`. Returns `409` if still running, `422` if failed.

```bash
curl http://localhost:17843/jobs/6c45ef27.../result
```

```json
{
  "job_id": "6c45ef27...",
  "structures": {
    "input": "data_...\nATOM  ..."
  },
  "confidence": {
    "input": {
      "confidence_score": 0.85,
      "ptm": 0.82,
      "iptm": 0.78,
      "complex_plddt": 0.83
    }
  },
  "affinity": {
    "input": {
      "affinity_pred_value": 7.42,
      "affinity_probability_binary": 0.91
    }
  }
}
```

`affinity` is only present when the input includes an affinity property.

---

### `GET /jobs`

List recent jobs.

```bash
curl 'http://localhost:17843/jobs?limit=20'
# {"jobs": [...], "total": 5}
```

---

## TOML Input Format

The TOML format is a direct translation of the boltz YAML schema. Each sequence entry uses a dotted key for the entity type.

### Protein structure prediction

```toml
version = 1

[[sequences]]
protein.id = "A"
protein.sequence = "QLEDSEVEAVAKGLEEMYANGVTEDNFKNYVKNNFAQQEISSVEEELNVNISDSCVANK"
```

### Protein–ligand affinity

```toml
version = 1

[[sequences]]
protein.id = "A"
protein.sequence = "MVTPEGNVSLVDESLLVG..."

[[sequences]]
ligand.id = "B"
ligand.smiles = "N[C@@H](Cc1ccc(O)cc1)C(=O)O"

[[properties]]
affinity.binder = "B"
```

### Multimer

```toml
version = 1

[[sequences]]
protein.id = "A"
protein.sequence = "MVTPEG..."

[[sequences]]
protein.id = "B"
protein.sequence = "ACDEFG..."
```

Multiple chain IDs can be given as a list to clone the same sequence:

```toml
[[sequences]]
protein.id = ["B", "C"]
protein.sequence = "ACDEFG..."
```

### Supported entity types

| Key | Required fields | Optional fields |
|---|---|---|
| `protein` | `id`, `sequence` | `msa` |
| `dna` | `id`, `sequence` | |
| `rna` | `id`, `sequence` | |
| `ligand` (SMILES) | `id`, `smiles` | |
| `ligand` (CCD code) | `id`, `ccd` | |

---

## Templates

Pass a known homolog structure to guide prediction. The server accepts template content **inline** via a `content` field — no need to upload a separate file.

The `cif` or `pdb` key acts as a format tag (the value is ignored); the server writes `content` to a temporary file and passes the path to boltz.

### CIF template

```toml
version = 1

[[sequences]]
protein.id = "A"
protein.sequence = "MVTPEG..."

[[templates]]
cif = "placeholder"
template_id = "A"
content = """
data_my_homolog
_entry.id my_homolog
loop_
_atom_site.group_PDB
_atom_site.id
...
"""
```

### PDB template (auto-converted to CIF by the server)

```toml
[[templates]]
pdb = "placeholder"
template_id = "A"
content = """
ATOM      1  N   MET A   1       1.000   2.000   3.000  1.00  0.00           N
...
"""
```

### Multiple chains, separate templates

```toml
[[sequences]]
protein.id = "A"
protein.sequence = "MVTPEG..."

[[sequences]]
protein.id = "B"
protein.sequence = "ACDEFG..."

[[templates]]
cif = "placeholder"
template_id = "A"
content = "..."

[[templates]]
cif = "placeholder"
template_id = "B"
content = "..."
```

### Optional template fields

| Field | Type | Default | Description |
|---|---|---|---|
| `template_id` | string | all protein chains | Chain ID in `sequences` this template applies to |
| `chain_id` | string or list | auto-detected | Chain(s) in the template structure to use |
| `force` | bool | `false` | Force the template even if alignment score is low |
| `threshold` | float | required if `force=true` | Minimum alignment score threshold |

---

## Passing a TOML file from the command line

```bash
# Read a .toml file and POST it
curl -X POST http://localhost:17843/predict \
  -H 'Content-Type: application/json' \
  -d "{\"toml\": $(python3 -c 'import sys,json; print(json.dumps(open("input.toml").read()))')}"
```

Or with Python:

```python
import json, requests

toml_text = open("input.toml").read()
resp = requests.post("http://localhost:17843/predict", json={"toml": toml_text})
job_id = resp.json()["job_id"]

# Poll until done
import time
while True:
    status = requests.get(f"http://localhost:17843/jobs/{job_id}").json()["status"]
    print(status)
    if status in ("completed", "failed"):
        break
    time.sleep(5)

result = requests.get(f"http://localhost:17843/jobs/{job_id}/result").json()
print(result["confidence"])
```
