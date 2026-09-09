"""Boltz FastAPI server.

Accepts TOML prediction requests and runs boltz predict locally.

## TOML format

Mirrors the boltz YAML schema. Templates use inline `content`. Protein sequences
can include an optional `msa_content` field (raw A3M text) that is saved to the
MSA cache so future jobs skip the MSA server for that sequence.

    version = 1

    [[sequences]]
    protein.id = "A"
    protein.sequence = "MVTPEG..."
    # Optional: inline A3M to seed the cache (omit if already cached or if
    # you want the server to fetch it automatically).
    protein.msa_content = \"\"\"
    >query
    MVTPEG...
    >hit1
    MVTPEG...
    \"\"\"

    [[sequences]]
    ligand.id = "B"
    ligand.smiles = "N[C@@H](Cc1ccc(O)cc1)C(=O)O"

    [[properties]]
    affinity.binder = "B"

    # Template with inline CIF/PDB content
    [[templates]]
    cif = "placeholder"
    content = \"\"\"data_template\\n...\"\"\"
    template_id = "A"

## MSA caching

Cache dir: $BOLTZ_MSA_CACHE (default ~/.boltz/msa_cache/).
Cache key: first 24 hex chars of SHA-256 of the uppercase stripped sequence.

Hit  → injects `msa: <cache_path>` into the YAML; no server call needed.
Miss → adds --use_msa_server; after a successful run the generated CSV is
       saved to cache automatically.

## Queue

Jobs are queued and drained by a worker pool (default size 1 = strict serial;
one GPU cannot run two boltz predictions at once). POST /predict returns
immediately with status "pending" and a queue_position. Poll GET /jobs/{id}
(status pending->running->completed/failed, plus queue_position while pending).
POST /jobs/{id}/cancel drops a still-pending job. GET /queue and GET /health
show running/pending.

## Environment variables

  BOLTZ_MOCK=1            Return fake results (~3 s) without a GPU.
  BOLTZ_MSA_CACHE=path    Override MSA cache directory.
  BOLTZ_MAX_CONCURRENCY=N Worker pool size (default 1). >1 only with real GPU
                         headroom or per-worker device pinning.
  BOLTZ_WORKER_DEVICES=.. Comma-separated CUDA_VISIBLE_DEVICES per worker slot
                         (e.g. "0,1" or MIG UUIDs); slot i uses entry i.
  BOLTZ_GPU_LOCK=path     Advisory flock held around each boltz subprocess so
                         other GPU users (batch_predict.py, MD) serialize too.
                         Unnecessary with MIG isolation.

## Usage

    uv run python server.py                    # localhost:17843
    BOLTZ_MOCK=1 uv run python server.py
    uv run python server.py --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

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

MOCK = os.environ.get("BOLTZ_MOCK", "0") == "1"

SERVER_DIR = Path(__file__).parent
JOBS_DIR = SERVER_DIR / "jobs"
JOBS_DIR.mkdir(exist_ok=True)

MSA_CACHE_DIR = Path(os.environ.get("BOLTZ_MSA_CACHE", Path.home() / ".boltz" / "msa_cache"))
MSA_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# --- serial execution ------------------------------------------------------
# One GPU cannot run two boltz predictions at once (measured: slower + 2x VRAM
# -> OOM). Jobs are queued and drained by a fixed pool of workers. Default pool
# size 1 = strict serial. Set BOLTZ_MAX_CONCURRENCY>1 only with real GPU
# headroom or per-worker device pinning (see BOLTZ_WORKER_DEVICES).
MAX_CONCURRENCY = max(1, int(os.environ.get("BOLTZ_MAX_CONCURRENCY", "1")))
# Optional per-worker CUDA_VISIBLE_DEVICES, comma-separated (e.g. MIG UUIDs or
# "0,1"); index i is used by worker i. Empty -> inherit the server's env.
WORKER_DEVICES = [d for d in os.environ.get("BOLTZ_WORKER_DEVICES", "").split(",") if d]
# Optional advisory lock path shared with other GPU users (batch_predict.py, MD
# scripts). When set, a worker flock()s it around the boltz subprocess so
# unrelated processes serialize too. Pointless with MIG isolation.
GPU_LOCK_PATH = os.environ.get("BOLTZ_GPU_LOCK") or None

_jobs: dict[str, dict] = {}
_queue_order: list[str] = []          # pending job_ids, FIFO, for queue_position
_active: dict[int, str | None] = {}   # worker slot -> job_id it is running
_workers: list[asyncio.Task] = []
_wake: asyncio.Event | None = None    # created in lifespan (loop-bound)


def _signal() -> None:
    if _wake is not None:
        _wake.set()


def _public(job: dict) -> dict:
    """Job dict without internal keys, plus live queue_position."""
    out = {k: v for k, v in job.items() if not k.startswith("_") and k not in ("log",)}
    if job.get("status") == "pending":
        try:
            out["queue_position"] = _queue_order.index(job["job_id"]) + 1
        except ValueError:
            out["queue_position"] = None
    return out


def _flock_blocking(path: str):
    import fcntl

    fh = open(path, "w")  # noqa: SIM115
    fcntl.flock(fh, fcntl.LOCK_EX)
    return fh


def _funlock(fh) -> None:
    import fcntl

    try:
        fcntl.flock(fh, fcntl.LOCK_UN)
        fh.close()
    except OSError:
        pass


def _claim_next(slot: int) -> str | None:
    """Pop the first runnable pending job and mark it running. Runs sync on the
    event loop, so claim + status flip is atomic against other workers."""
    taken = set(_active.values())
    for jid in list(_queue_order):
        job = _jobs.get(jid)
        if job is None or job["status"] == "cancelled":
            _queue_order.remove(jid)
            continue
        if job["status"] != "pending" or jid in taken:
            continue
        _queue_order.remove(jid)
        job["status"] = "running"
        job["updated_at"] = time.time()
        _active[slot] = jid
        return jid
    return None


async def _worker(slot: int) -> None:
    assert _wake is not None
    while True:
        job_id = _claim_next(slot)
        if job_id is None:
            _wake.clear()
            await _wake.wait()
            continue
        job = _jobs[job_id]
        try:
            yaml_path = Path(job["_yaml_path"])
            out_dir = Path(job["_out_dir"])
            device = WORKER_DEVICES[slot] if slot < len(WORKER_DEVICES) else None
            lock_fh = None
            if GPU_LOCK_PATH and not MOCK:
                lock_fh = await asyncio.to_thread(_flock_blocking, GPU_LOCK_PATH)
            try:
                if MOCK:
                    await _run_mock(job_id, yaml_path, out_dir)
                else:
                    await _run_boltz(
                        job_id, yaml_path, out_dir, job["_extra_args"], job["_needs_server"], device
                    )
            finally:
                if lock_fh is not None:
                    await asyncio.to_thread(_funlock, lock_fh)
        except Exception as exc:  # noqa: BLE001
            job["status"] = "failed"
            job["error"] = f"worker error: {exc}"
            job["updated_at"] = time.time()
        finally:
            _active[slot] = None
            _wake.set()  # re-check for more work


@asynccontextmanager
async def lifespan(_app: "FastAPI"):
    global _wake
    _wake = asyncio.Event()
    for i in range(MAX_CONCURRENCY):
        _active[i] = None
        _workers.append(asyncio.create_task(_worker(i)))
    try:
        yield
    finally:
        for w in _workers:
            w.cancel()


app = FastAPI(title="Boltz Predict API", lifespan=lifespan)


class PredictRequest(BaseModel):
    toml: str
    extra_args: str = ""


# ---------------------------------------------------------------------------
# MSA cache helpers
# ---------------------------------------------------------------------------

def _seq_hash(seq: str) -> str:
    return hashlib.sha256(seq.strip().upper().encode()).hexdigest()[:24]


def _cache_path_for(seq: str) -> Path | None:
    """Return the cached MSA path for a sequence, or None if not cached."""
    h = _seq_hash(seq)
    for suffix in (".a3m", ".csv"):
        p = MSA_CACHE_DIR / f"{h}{suffix}"
        if p.exists():
            return p
    return None


def _save_msa_from_run(out_dir: Path) -> None:
    """Scan boltz-generated MSA CSVs and save uncached ones to the cache dir."""
    for csv_path in out_dir.glob("boltz_results_*/msa/*.csv"):
        try:
            lines = csv_path.read_text().splitlines()
            # CSV format: header row "key,sequence", then data rows
            if len(lines) < 2:
                continue
            query_seq = lines[1].split(",", 1)[1]
            h = _seq_hash(query_seq)
            dest = MSA_CACHE_DIR / f"{h}.csv"
            if not dest.exists():
                shutil.copy(csv_path, dest)
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# TOML → YAML conversion (handles templates + MSA cache injection)
# ---------------------------------------------------------------------------

def _prepare_input(toml_str: str, work_dir: Path) -> tuple[Path, bool]:
    """Parse TOML, resolve templates and MSA cache, write YAML.

    Returns (yaml_path, needs_msa_server).
    """
    data = tomllib.loads(toml_str)

    # --- templates: write inline content to temp files ---
    for tmpl in data.get("templates", []):
        content = tmpl.pop("content", None)
        if content is None:
            continue
        if "cif" in tmpl:
            key, suffix = "cif", ".cif"
        elif "pdb" in tmpl:
            key, suffix = "pdb", ".pdb"
        else:
            continue
        tmpl_file = work_dir / f"template_{uuid.uuid4().hex[:8]}{suffix}"
        tmpl_file.write_text(content)
        tmpl[key] = str(tmpl_file)

    # --- MSA: inject cache hits, flag misses ---
    needs_server = False
    for entry in data.get("sequences", []):
        prot = entry.get("protein")
        if not prot:
            continue
        seq = prot.get("sequence", "").strip()
        if not seq:
            continue

        # User-supplied inline A3M → save to cache, inject path
        msa_content = prot.pop("msa_content", None)
        if msa_content:
            h = _seq_hash(seq)
            cache_file = MSA_CACHE_DIR / f"{h}.a3m"
            cache_file.write_text(msa_content)
            prot["msa"] = str(cache_file)
            continue

        # Already has an explicit msa path → leave it alone
        if "msa" in prot:
            continue

        # Check cache
        cached = _cache_path_for(seq)
        if cached:
            prot["msa"] = str(cached)
        else:
            needs_server = True  # this sequence needs the MSA server

    yaml_path = work_dir / "input.yaml"
    yaml_path.write_text(yaml.dump(data, allow_unicode=True, sort_keys=False))
    return yaml_path, needs_server


# ---------------------------------------------------------------------------
# Job runners
# ---------------------------------------------------------------------------

async def _run_boltz(
    job_id: str,
    yaml_path: Path,
    out_dir: Path,
    extra_args: str,
    needs_server: bool,
    device: str | None = None,
) -> None:
    job = _jobs[job_id]
    job["status"] = "running"
    job["updated_at"] = time.time()

    boltz_bin = str(Path(sys.executable).parent / "boltz")
    cmd = [boltz_bin, "predict", str(yaml_path), "--out_dir", str(out_dir)]
    if needs_server:
        cmd += ["--use_msa_server"]
    if extra_args:
        cmd += extra_args.split()

    env = None
    if device is not None:
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": device}

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
        )
        stdout, _ = await proc.communicate()
        job["log"] = stdout.decode(errors="replace")

        if proc.returncode != 0:
            job["status"] = "failed"
            job["error"] = f"boltz exited with code {proc.returncode}"
        else:
            job["status"] = "completed"
            _save_msa_from_run(out_dir)
    except Exception as exc:  # noqa: BLE001
        job["status"] = "failed"
        job["error"] = str(exc)

    job["updated_at"] = time.time()


async def _run_mock(job_id: str, yaml_path: Path, out_dir: Path) -> None:
    """Fake a boltz run: sleep ~3 s then write dummy result files."""
    job = _jobs[job_id]
    job["status"] = "running"
    job["updated_at"] = time.time()

    await asyncio.sleep(3)

    try:
        data = yaml.safe_load(yaml_path.read_text())
        record_id = yaml_path.stem  # "input"

        record_dir = out_dir / f"boltz_results_{record_id}" / "predictions" / record_id
        record_dir.mkdir(parents=True)

        (record_dir / f"{record_id}_model_0.cif").write_text(
            "# mock CIF structure\nATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00  0.00           C\n"
        )
        (record_dir / f"confidence_{record_id}_model_0.json").write_text(
            json.dumps({"confidence_score": 0.85, "ptm": 0.82, "iptm": 0.78, "complex_plddt": 0.83}, indent=2)
        )
        if any("affinity" in p for p in data.get("properties", [])):
            (record_dir / f"affinity_{record_id}.json").write_text(
                json.dumps({"affinity_pred_value": 7.42, "affinity_probability_binary": 0.91}, indent=2)
            )

        # Simulate saving a generated MSA CSV to cache for uncached proteins
        msa_dir = out_dir / f"boltz_results_{record_id}" / "msa"
        msa_dir.mkdir(parents=True)
        for entry in data.get("sequences", []):
            prot = entry.get("protein")
            if not prot or "msa" in prot:
                continue
            seq = prot.get("sequence", "").strip()
            if not seq:
                continue
            h = _seq_hash(seq)
            csv_content = f"key,sequence\n0,{seq}\n"
            (msa_dir / f"input_mock.csv").write_text(csv_content)
            dest = MSA_CACHE_DIR / f"{h}.csv"
            if not dest.exists():
                dest.write_text(csv_content)

        job["status"] = "completed"
    except Exception as exc:  # noqa: BLE001
        job["status"] = "failed"
        job["error"] = str(exc)

    job["updated_at"] = time.time()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

_HELP = {
    "service": "Boltz-2 predict API",
    "overview": (
        "POST a TOML prediction request; it is queued and run by a worker pool "
        "(default 1 -> strict serial, since one GPU cannot run two boltz "
        "predictions at once). Poll the job until it is 'completed', then GET "
        "its result. One request = one boltz run = one output complex (which "
        "may contain several chains). To screen many ligands, submit many "
        "requests (see 'screening_many_ligands')."
    ),
    "interactive_docs": "GET /docs  (Swagger UI, full request/response schema)",
    "endpoints": {
        "POST /predict": {
            "body": {
                "toml": "string, required -- the prediction spec, see 'toml_format'",
                "extra_args": "string, optional -- extra flags passed verbatim to `boltz predict` "
                              "(e.g. '--recycling_steps 1 --sampling_steps 50 --diffusion_samples_affinity 1')",
            },
            "returns": "202  {job_id, status:'pending', msa_cache_hits, queue_position}",
            "errors": "422 if the TOML does not parse",
        },
        "GET /jobs/{job_id}": {
            "returns": "{job_id, status, msa_cache_hits, created_at, updated_at, "
                       "queue_position (while pending), error (if failed)}",
            "status_values": ["pending", "running", "completed", "failed", "cancelled"],
        },
        "GET /jobs/{job_id}/result": {
            "returns": "{job_id, structures:{<record>: <mmCIF text>}, "
                       "confidence:{<record>: {...}}, affinity:{<record>: {...}}}",
            "errors": "409 if not finished yet, 422 if the job failed, 404 if unknown",
        },
        "POST /jobs/{job_id}/cancel": {
            "effect": "drops a job that is still 'pending' (returns 409 once it is running)",
            "returns": "{job_id, status:'cancelled'}",
        },
        "GET /jobs?limit=N": "recent jobs (default 20), same shape as GET /jobs/{id}",
        "GET /queue": "{workers, running:[job_id...], pending:[job_id...]}",
        "GET /msa/cache": "{cache_dir, entries, files} -- sequences currently cached",
        "GET /health": "{status, mock, msa_cache_dir, workers, running, queued, gpu_lock}",
        "GET /help": "this document",
    },
    "polling": (
        "After POST /predict, poll GET /jobs/{job_id} every ~10 s. While "
        "'pending', queue_position tells you how many jobs are ahead. Fetch "
        "GET /jobs/{job_id}/result once status is 'completed'."
    ),
    "toml_format": {
        "note": "Mirrors the boltz YAML schema. See GET /docs and docs/prediction.md.",
        "minimal": (
            "version = 1\n\n"
            '[[sequences]]\n'
            'protein.id = "A"\n'
            'protein.sequence = "MVTPEG..."\n\n'
            '[[sequences]]\n'
            'ligand.id = "B"\n'
            'ligand.smiles = "CCO"\n\n'
            '[[properties]]\n'
            'affinity.binder = "B"\n'
        ),
        "fields": {
            "[[sequences]] protein": "protein.id, protein.sequence; optional protein.msa (path) "
                                     "or protein.msa_content (raw A3M text, saved to the cache); "
                                     "protein.cyclic = true",
            "[[sequences]] ligand": "ligand.id + one of ligand.smiles / ligand.ccd",
            "[[properties]] affinity": "affinity.binder = <ligand chain id>. Boltz-2 supports "
                                       "exactly ONE affinity binder per request.",
            "[[templates]]": 'cif = "placeholder" (or pdb=...) + content = """<file text>""" + template_id',
            "[[constraints]]": "pocket / bond / contact -- see docs/prediction.md",
        },
    },
    "msa": {
        "cache_dir": str(MSA_CACHE_DIR),
        "key": "sha256(uppercase stripped sequence)[:24]",
        "hit": "the cached MSA path is injected into the YAML; no MSA server call",
        "miss": "the server adds --use_msa_server; the generated CSV is saved to the cache after a "
                "successful run, so the next request for that sequence is a hit",
        "seed_it_yourself": "put raw A3M in protein.msa_content to prime the cache without a server call",
    },
    "screening_many_ligands": {
        "one_complex_vs_many_jobs": (
            "Putting several ligand [[sequences]] blocks in ONE toml cofolds them "
            "together as a single complex (and you still get at most one affinity "
            "value). For a screen -- N ligands each scored against the same "
            "receptor -- submit N separate POST /predict requests, each with the "
            "receptor + one ligand + its own affinity.binder."
        ),
        "recipe": [
            "1. Warm the receptor MSA once: submit the first ligand (or any job "
            "with that receptor). After it finishes the MSA is cached, so every "
            "later request for the same receptor is a cache hit (no ~60 s MSA "
            "fetch).",
            "2. Submit the remaining ligands. They queue behind each other and "
            "run one at a time; each response carries queue_position.",
            "3. Poll GET /queue (or each GET /jobs/{id}) until all are "
            "'completed'/'failed', then collect GET /jobs/{id}/result.",
            "4. A failed ligand does not block the rest -- its job goes 'failed' "
            "and the queue moves on.",
        ],
        "fast_screening_flags": (
            "pass via extra_args: '--recycling_steps 1 --sampling_steps 50 "
            "--diffusion_samples 1 --sampling_steps_affinity 50 "
            "--diffusion_samples_affinity 1 --recycling_steps_affinity 3 "
            "--matmul_precision high --no_write_full_pae --no_write_full_pde'. "
            "Measured ~5 s/ligand on an RTX 5090 (384 aa receptor) vs ~17 s with "
            "defaults; accuracy drops modestly -- use as a first-pass filter."
        ),
        "local_cli": (
            "scripts/batch_predict.py does this end to end against a local "
            "`boltz predict` (not this server): one receptor + a SMILES list -> "
            "one process, per-ligand results.json kept even on a crash, "
            "--resume, auto batch size, OOM fallback. See docs/batch_screening.md. "
            "It is faster than N requests here because a single `boltz predict` "
            "over a directory loads the model once; this server reloads per job "
            "unless BOLTZ_MAX_CONCURRENCY workers stay warm."
        ),
        "timing_note": (
            "10 ligands / 384 aa / fast flags, measured: N separate `boltz "
            "predict` = 429 s; one batched `boltz predict <dir>` = 95 s; a warm "
            "resident worker fed one at a time = 131 s (10 s/ligand after the "
            "first). Batch wins when you have all ligands up front; a warm "
            "worker wins when ligands arrive over time."
        ),
    },
    "concurrency": {
        "BOLTZ_MAX_CONCURRENCY": "worker pool size (default 1 = serial). >1 only with real GPU "
                                 "headroom or per-worker device isolation.",
        "BOLTZ_WORKER_DEVICES": "comma-separated CUDA_VISIBLE_DEVICES per worker slot (e.g. '0,1' "
                                "or H100 MIG instance UUIDs); slot i uses entry i.",
        "BOLTZ_GPU_LOCK": "path to an advisory flock held around each boltz subprocess so other "
                          "GPU users (batch_predict.py, MD jobs) serialize too. Unneeded with MIG.",
        "BOLTZ_MOCK": "1 -> fake ~3 s results, no GPU (for testing).",
        "BOLTZ_MSA_CACHE": "override the MSA cache directory.",
    },
}


@app.get("/help")
def help_doc():
    return _HELP


@app.get("/")
def root():
    return {"service": "Boltz-2 predict API", "help": "GET /help", "docs": "GET /docs"}


@app.post("/predict", status_code=202)
async def predict(req: PredictRequest):
    job_id = uuid.uuid4().hex
    work_dir = JOBS_DIR / job_id
    work_dir.mkdir()
    out_dir = work_dir / "out"
    out_dir.mkdir()

    try:
        yaml_path, needs_server = _prepare_input(req.toml, work_dir)
    except Exception as exc:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise HTTPException(status_code=422, detail=f"TOML parse error: {exc}") from exc

    _jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "msa_cache_hits": not needs_server,
        "created_at": time.time(),
        "updated_at": time.time(),
        "_work_dir": str(work_dir),
        "_out_dir": str(out_dir),
        "_yaml_path": str(yaml_path),
        "_extra_args": req.extra_args,
        "_needs_server": needs_server,
    }
    _queue_order.append(job_id)
    _signal()

    return {
        "job_id": job_id,
        "status": "pending",
        "msa_cache_hits": not needs_server,
        "queue_position": len(_queue_order),
    }


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return _public(job)


@app.post("/jobs/{job_id}/cancel", status_code=200)
async def cancel_job(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "pending":
        raise HTTPException(status_code=409, detail=f"Cannot cancel a job in status '{job['status']}'")
    job["status"] = "cancelled"
    job["updated_at"] = time.time()
    if job_id in _queue_order:
        _queue_order.remove(job_id)
    return {"job_id": job_id, "status": "cancelled"}


@app.get("/jobs/{job_id}/result")
def get_result(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] == "failed":
        raise HTTPException(status_code=422, detail=job.get("error", "prediction failed"))
    if job["status"] != "completed":
        raise HTTPException(status_code=409, detail=f"Job status: {job['status']}")

    out_dir = Path(job["_out_dir"])
    result: dict = {"job_id": job_id}

    for record_dir in out_dir.glob("boltz_results_*/predictions/*"):
        if not record_dir.is_dir():
            continue
        for ext in ("*.cif", "*.pdb"):
            structs = sorted(record_dir.glob(ext))
            if structs:
                result.setdefault("structures", {})[record_dir.name] = structs[0].read_text()
                break
        conf_files = sorted(record_dir.glob("confidence_*.json"))
        if conf_files:
            result.setdefault("confidence", {})[record_dir.name] = json.loads(conf_files[0].read_text())
        aff_files = sorted(record_dir.glob("affinity_*.json"))
        if aff_files:
            result.setdefault("affinity", {})[record_dir.name] = json.loads(aff_files[0].read_text())

    return JSONResponse(result)


@app.get("/jobs")
def list_jobs(limit: int = 20):
    jobs = [_public(j) for j in list(_jobs.values())[-limit:]]
    return {"jobs": jobs, "total": len(_jobs)}


@app.get("/msa/cache")
def msa_cache_info():
    """List sequences currently in the MSA cache."""
    entries = [p.name for p in sorted(MSA_CACHE_DIR.iterdir()) if p.suffix in (".a3m", ".csv")]
    return {"cache_dir": str(MSA_CACHE_DIR), "entries": len(entries), "files": entries}


@app.get("/health")
def health():
    running = [jid for jid in _active.values() if jid]
    return {
        "status": "ok",
        "mock": MOCK,
        "msa_cache_dir": str(MSA_CACHE_DIR),
        "workers": MAX_CONCURRENCY,
        "running": running,
        "queued": len(_queue_order),
        "gpu_lock": GPU_LOCK_PATH,
    }


@app.get("/queue")
def queue_state():
    return {
        "workers": MAX_CONCURRENCY,
        "running": [jid for jid in _active.values() if jid],
        "pending": list(_queue_order),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=17843)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
