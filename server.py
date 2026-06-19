"""Boltz FastAPI server.

Accepts TOML prediction requests and runs boltz predict locally.

TOML format mirrors the boltz YAML format. Templates use inline `content`:

    version = 1

    [[sequences]]
    protein.id = "A"
    protein.sequence = "MVTPEG..."

    [[sequences]]
    ligand.id = "B"
    ligand.smiles = "N[C@@H](Cc1ccc(O)cc1)C(=O)O"

    [[properties]]
    affinity.binder = "B"

    # Template with inline CIF content (server writes it to a temp file)
    [[templates]]
    cif = "placeholder"
    content = \"\"\"
    data_my_template
    ...
    \"\"\"
    template_id = "A"

Usage:
    pip install fastapi uvicorn
    python server.py           # starts on :17843
    python server.py --port 8000 --host 0.0.0.0
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Optional

import uvicorn
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ImportError as e:
        msg = "Install tomli: pip install tomli"
        raise ImportError(msg) from e

# Set BOLTZ_MOCK=1 to skip real GPU computation and return fake results.
MOCK = os.environ.get("BOLTZ_MOCK", "0") == "1"

app = FastAPI(title="Boltz Predict API")

# In-memory job store: job_id -> {status, out_dir, ...}
_jobs: dict[str, dict] = {}


class PredictRequest(BaseModel):
    toml: str
    extra_args: str = ""


def _toml_to_yaml_with_templates(toml_str: str, work_dir: Path) -> Path:
    """Parse TOML, extract template content to temp files, write YAML."""
    data = tomllib.loads(toml_str)

    templates = data.get("templates", [])
    for tmpl in templates:
        content = tmpl.pop("content", None)
        if content is None:
            continue
        if "cif" in tmpl:
            suffix = ".cif"
            key = "cif"
        elif "pdb" in tmpl:
            suffix = ".pdb"
            key = "pdb"
        else:
            continue
        tmpl_file = work_dir / f"template_{uuid.uuid4().hex[:8]}{suffix}"
        tmpl_file.write_text(content)
        tmpl[key] = str(tmpl_file)

    yaml_path = work_dir / "input.yaml"
    yaml_path.write_text(yaml.dump(data, allow_unicode=True, sort_keys=False))
    return yaml_path


async def _run_boltz(job_id: str, yaml_path: Path, out_dir: Path, extra_args: str) -> None:
    job = _jobs[job_id]
    job["status"] = "running"
    job["updated_at"] = time.time()

    cmd = [sys.executable, "-m", "boltz", "predict", str(yaml_path), "--out_dir", str(out_dir)]
    if extra_args:
        cmd += extra_args.split()

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await proc.communicate()
        job["log"] = stdout.decode(errors="replace")

        if proc.returncode != 0:
            job["status"] = "failed"
            job["error"] = f"boltz exited with code {proc.returncode}"
        else:
            job["status"] = "completed"
    except Exception as exc:  # noqa: BLE001
        job["status"] = "failed"
        job["error"] = str(exc)

    job["updated_at"] = time.time()


async def _run_mock(job_id: str, yaml_path: Path, out_dir: Path) -> None:
    """Fake a boltz run: sleep a few seconds then write dummy result files."""
    job = _jobs[job_id]
    job["status"] = "running"
    job["updated_at"] = time.time()

    await asyncio.sleep(3)  # simulate computation time

    try:
        data = yaml.safe_load(yaml_path.read_text())
        # derive a record name from the yaml filename stem
        record_id = yaml_path.stem  # "input"

        record_dir = out_dir / f"boltz_results_{record_id}" / "predictions" / record_id
        record_dir.mkdir(parents=True)

        (record_dir / f"{record_id}_model_0.cif").write_text(
            "# mock CIF structure\nATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00  0.00           C\n"
        )

        confidence = {
            "confidence_score": 0.85,
            "ptm": 0.82,
            "iptm": 0.78,
            "complex_plddt": 0.83,
        }
        (record_dir / f"confidence_{record_id}_model_0.json").write_text(
            json.dumps(confidence, indent=2)
        )

        sequences = data.get("sequences", [])
        has_affinity = any("affinity" in p for p in data.get("properties", []))
        if has_affinity:
            affinity = {
                "affinity_pred_value": 7.42,
                "affinity_probability_binary": 0.91,
            }
            (record_dir / f"affinity_{record_id}.json").write_text(
                json.dumps(affinity, indent=2)
            )

        job["status"] = "completed"
    except Exception as exc:  # noqa: BLE001
        job["status"] = "failed"
        job["error"] = str(exc)

    job["updated_at"] = time.time()


@app.post("/predict", status_code=202)
async def predict(req: PredictRequest):
    job_id = uuid.uuid4().hex
    work_dir = Path(tempfile.mkdtemp(prefix=f"boltz_{job_id}_"))
    out_dir = work_dir / "out"
    out_dir.mkdir()

    try:
        yaml_path = _toml_to_yaml_with_templates(req.toml, work_dir)
    except Exception as exc:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise HTTPException(status_code=422, detail=f"TOML parse error: {exc}") from exc

    _jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "work_dir": str(work_dir),
        "out_dir": str(out_dir),
        "created_at": time.time(),
        "updated_at": time.time(),
    }

    if MOCK:
        asyncio.create_task(_run_mock(job_id, yaml_path, out_dir))
    else:
        asyncio.create_task(_run_boltz(job_id, yaml_path, out_dir, req.extra_args))
    return {"job_id": job_id, "status": "pending"}


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return {k: v for k, v in job.items() if k not in ("work_dir", "out_dir", "log")}


@app.get("/jobs/{job_id}/result")
def get_result(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] == "failed":
        raise HTTPException(status_code=422, detail=job.get("error", "prediction failed"))
    if job["status"] != "completed":
        raise HTTPException(status_code=409, detail=f"Job status: {job['status']}")

    out_dir = Path(job["out_dir"])
    result: dict = {"job_id": job_id}

    # boltz writes to out_dir/boltz_results_{stem}/predictions/{record_id}/
    preds_dirs = list(out_dir.glob("boltz_results_*/predictions/*"))
    for record_dir in preds_dirs:
        if not record_dir.is_dir():
            continue

        # Structure: first ranked model
        for ext in ("*.cif", "*.pdb"):
            structs = sorted(record_dir.glob(ext))
            if structs:
                result.setdefault("structures", {})[record_dir.name] = structs[0].read_text()
                break

        # Confidence
        conf_files = sorted(record_dir.glob("confidence_*.json"))
        if conf_files:
            result.setdefault("confidence", {})[record_dir.name] = json.loads(
                conf_files[0].read_text()
            )

        # Affinity
        aff_files = sorted(record_dir.glob("affinity_*.json"))
        if aff_files:
            result.setdefault("affinity", {})[record_dir.name] = json.loads(
                aff_files[0].read_text()
            )

    return JSONResponse(result)


@app.get("/jobs")
def list_jobs(limit: int = 20):
    jobs = [
        {k: v for k, v in j.items() if k not in ("work_dir", "out_dir", "log")}
        for j in list(_jobs.values())[-limit:]
    ]
    return {"jobs": jobs, "total": len(_jobs)}


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=17843)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
