#!/usr/bin/env python3
"""Batch cofold + affinity for one receptor against many ligands.

Screening N ligands against one protein means N structure(+affinity) predictions
of ``protein + ligand_i``.  Two ways to run them through the stock ``boltz`` CLI:

* **single** : one ``boltz predict <file.yaml>`` process per ligand.  Every
  process re-imports torch/Lightning and re-loads the Boltz-2 checkpoints onto
  the GPU (~35 s of pure overhead per ligand, measured on an RTX 5090).
* **batch**  : one ``boltz predict <dir>/`` process over a directory of all N
  YAMLs.  The model loads once and is reused; ``--batch_size K`` additionally
  packs K structure inferences into one forward pass (the affinity phase always
  runs at batch_size 1).

Auto batch size
---------------
Packing several records per forward is a memory gamble: the triangular
attention/multiplication working set grows with the token count
(protein residues + ligand heavy atoms).  ``--batch-size`` is chosen
automatically from that estimate and the GPU's free memory unless you pass an
explicit value:

    n_tokens <= 300  -> 8      small system: pack aggressively
    n_tokens <= 500  -> 4
    n_tokens <= 700  -> 2
    n_tokens  > 700  -> 1      large system: never pack

then scaled by (free_GB / 24).  If a batch still OOMs, the run is retried with
the batch size halved until it fits (down to 1).

MSA
---
Resolution order, no flag needed in the common case:
  1. ``--msa PATH``                              explicit
  2. server-style cache hit                      $BOLTZ_MSA_CACHE or ~/.boltz/msa_cache/
                                                 keyed by sha256(seq)[:24]
  3. ``--no-msa``                                the 'empty' single-seq sentinel
  4. otherwise                                   fetch once from the MSA server and
                                                 write it into the cache for next time

Reproducibility
---------------
* The featurizer uses a hard-coded local RNG (seed 42), independent of mode.
* Diffusion draws from the global torch RNG, seeded once per process by
  ``seed_everything(--seed)``.  So: same mode + same seed + same ordering =
  byte-identical on re-run (``--check-repro`` verifies).  batch vs single differ
  slightly per ligand because boltz seeds once per process and runs the
  structure phase over all records before the affinity phase -- expected, not a
  bug.

Usage
-----
    python scripts/batch_predict.py --protein rec.fasta --ligands ligs.smi \
        --out runs/screen1 --seed 42                     # auto everything

    python scripts/batch_predict.py --protein SEQ --smiles CCO --smiles c1ccccc1 \
        --out runs/smoke --no-msa --mode both --check-repro \
        --boltz-arg=--sampling_steps=50 --boltz-arg=--recycling_steps=1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OOM_RE = re.compile(
    r"out of memory|CUDA out of memory|CUBLAS_STATUS_ALLOC_FAILED|HIP out of memory|OutOfMemoryError",
    re.I,
)


def boltz_bin() -> str:
    """Locate the ``boltz`` console script (``python -m boltz`` has no __main__)."""
    if os.environ.get("BOLTZ_BIN"):
        return os.environ["BOLTZ_BIN"]
    cand = Path(sys.executable).parent / "boltz"
    if cand.exists():
        return str(cand)
    found = shutil.which("boltz")
    if found:
        return found
    sys.exit("cannot find the 'boltz' entry point; set BOLTZ_BIN=/path/to/venv/bin/boltz")


# ---------------------------------------------------------------------------
# input assembly
# ---------------------------------------------------------------------------
def read_protein(spec: str) -> str:
    """Return a bare protein sequence from a raw string, a .fasta or a .txt file."""
    p = Path(spec)
    try:
        is_file = p.is_file()
    except OSError:  # spec is a long sequence, not a path
        is_file = False
    if is_file:
        text = p.read_text()
        if p.suffix.lower() in {".fasta", ".fa", ".faa"} or text.lstrip().startswith(">"):
            seq = "".join(
                line.strip() for line in text.splitlines() if line and not line.startswith(">")
            )
        else:
            seq = "".join(text.split())
        return seq.upper()
    return "".join(spec.split()).upper()


def read_ligands(args: argparse.Namespace) -> list[str]:
    ligs: list[str] = list(args.smiles or [])
    if args.ligands:
        for line in Path(args.ligands).read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            ligs.append(line.split()[0])  # "SMILES" or "SMILES <name>"
    if not ligs:
        sys.exit("no ligands: pass --ligands FILE and/or --smiles SMILES ...")
    return ligs


def heavy_atom_count(smiles: str) -> int:
    """Heavy-atom count for a SMILES, via rdkit if importable, else a crude scan."""
    try:
        from rdkit import Chem  # noqa: PLC0415

        m = Chem.MolFromSmiles(smiles)
        if m is not None:
            return m.GetNumHeavyAtoms()
    except Exception:  # noqa: BLE001
        pass
    # fallback: element letters, dropping explicit hydrogens and ring-bond digits
    toks = re.findall(r"Cl|Br|[BCNOPSFIbcnops]", re.sub(r"\[[^\]]*H[^\]]*\]", "X", smiles))
    return max(1, len(toks))


# ---------------------------------------------------------------------------
# MSA resolution
# ---------------------------------------------------------------------------
def _msa_cache_dir() -> Path:
    return Path(os.environ.get("BOLTZ_MSA_CACHE", Path.home() / ".boltz" / "msa_cache"))


def _seq_hash(seq: str) -> str:
    """Same key server.py uses, so caches interoperate."""
    return hashlib.sha256(seq.strip().upper().encode()).hexdigest()[:24]


def resolve_msa(seq: str, args: argparse.Namespace, out: Path) -> str | None:
    if args.msa:
        return str(Path(args.msa).resolve())
    if args.no_msa:
        return "empty"

    cache_dir = _msa_cache_dir()
    h = _seq_hash(seq)
    for suffix in (".a3m", ".csv"):
        hit = cache_dir / f"{h}{suffix}"
        if hit.exists():
            print(f"[msa] cache hit -> {hit}", flush=True)
            return str(hit.resolve())

    # miss: fetch once, drop a copy in the cache for next time
    from boltz.data.msa.mmseqs2 import run_mmseqs2  # noqa: PLC0415

    print(f"[msa] miss; querying {args.msa_server_url} for the receptor ({len(seq)} aa) ...", flush=True)
    a3m_lines, _ = run_mmseqs2(
        [seq], str(out / "msa_query"), use_env=True, use_pairing=False, host_url=args.msa_server_url
    )
    local = out / "receptor.a3m"
    local.write_text(a3m_lines[0])
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(local, cache_dir / f"{h}.a3m")
        print(f"[msa] cached -> {cache_dir / f'{h}.a3m'}", flush=True)
    except OSError as e:  # cache dir not writable -> local copy still works
        print(f"[msa] (cache write skipped: {e})", flush=True)
    return str(local.resolve())


def write_yaml(path: Path, seq: str, smiles: str, msa: str | None, affinity: bool) -> None:
    lines = ["version: 1", "sequences:", "  - protein:", "      id: A", f"      sequence: {seq}"]
    if msa is not None:
        lines.append(f"      msa: {msa}")
    lines += ["  - ligand:", "      id: B", f"      smiles: '{smiles}'"]
    if affinity:
        lines += ["properties:", "  - affinity:", "      binder: B"]
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# batch-size heuristic
# ---------------------------------------------------------------------------
def gpu_free_gb() -> float | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        )
        vals = [float(x) for x in out.stdout.split()]
        return min(vals) / 1024 if vals else None
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def auto_batch_size(n_tokens: int, n_ligs: int, free_gb: float | None, cap: int) -> tuple[int, str]:
    if n_tokens <= 300:
        base = 8
    elif n_tokens <= 500:
        base = 4
    elif n_tokens <= 700:
        base = 2
    else:
        base = 1
    scaled = base
    if free_gb:
        scaled = max(1, int(base * min(free_gb, 96) / 24))
    bs = max(1, min(scaled, cap, n_ligs))
    why = (
        f"n_tokens~{n_tokens} -> base {base}"
        + (f", x free {free_gb:.0f}GB/24 -> {scaled}" if free_gb else ", (GPU mem unknown)")
        + f", cap {cap}, n_ligs {n_ligs} => batch_size {bs}"
    )
    return bs, why


# ---------------------------------------------------------------------------
# running boltz
# ---------------------------------------------------------------------------
def boltz_cmd(
    data: Path,
    out_dir: Path,
    seed: int,
    extra: list[str],
    batch_size: int,
    *,
    override: bool = True,
    skip_bad: bool = True,
) -> list[str]:
    cmd = [
        boltz_bin(), "predict", str(data),
        "--out_dir", str(out_dir),
        "--seed", str(seed),
        "--batch_size", str(batch_size),
    ]
    if override:
        cmd.append("--override")
    if skip_bad:
        cmd.append("--skip_bad_inputs")  # one bad ligand must not abort the rest
    return cmd + extra


def run(cmd: list[str], log: Path) -> tuple[float, int]:
    log.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with log.open("w") as fh:
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=REPO_ROOT)
    return time.perf_counter() - start, proc.returncode


# ---------------------------------------------------------------------------
# result collection
# ---------------------------------------------------------------------------
CONF_KEYS = ("confidence_score", "ptm", "iptm", "complex_plddt")
AFF_KEYS = ("affinity_pred_value", "affinity_probability_binary")


def collect(out_dir: Path) -> dict[str, dict]:
    """record stem -> {cif_sha256, confidence:{...}, affinity:{...}}"""
    results: dict[str, dict] = {}
    for pred_dir in sorted(out_dir.glob("boltz_results_*/predictions/*")):
        if not pred_dir.is_dir():
            continue
        stem = pred_dir.name
        entry: dict = {}
        cifs = sorted(pred_dir.glob("*_model_0.cif")) or sorted(pred_dir.glob("*.cif"))
        if cifs:
            entry["cif_sha256"] = hashlib.sha256(cifs[0].read_bytes()).hexdigest()[:16]
        conf = sorted(pred_dir.glob("confidence_*_model_0.json")) or sorted(
            pred_dir.glob("confidence_*.json")
        )
        if conf:
            d = json.loads(conf[0].read_text())
            entry["confidence"] = {k: d.get(k) for k in CONF_KEYS}
        aff = sorted(pred_dir.glob("affinity_*.json"))
        if aff:
            d = json.loads(aff[0].read_text())
            entry["affinity"] = {k: d.get(k) for k in AFF_KEYS}
        results[stem] = entry
    return results


def collect_many(root: Path) -> dict[str, dict]:
    merged: dict[str, dict] = {}
    if not root.exists():
        return merged
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        merged.update(collect(sub))
    return merged


def full_affinity(out_dir: Path, stem: str) -> dict | None:
    """The complete affinity_*.json for one record (all fields), or None.

    ``**`` so it works for both the batch layout (out/boltz_results_inputs/...)
    and the single layout (rep0/<stem>/boltz_results_<stem>/...).
    """
    for f in out_dir.glob(f"**/predictions/{stem}/affinity_*.json"):
        try:
            return json.loads(f.read_text())
        except (OSError, json.JSONDecodeError):
            return None
    return None


def screen_results(
    ligands: list[str], stems: list[str], out_dir: Path, results: dict[str, dict], want_affinity: bool
) -> list[dict]:
    """Per-ligand status, always computable from whatever landed on disk."""
    rows = []
    for i, (smi, stem) in enumerate(zip(ligands, stems)):
        entry = results.get(stem, {})
        has_struct = "cif_sha256" in entry
        has_aff = "affinity" in entry
        if has_struct and (has_aff or not want_affinity):
            status = "completed"
        elif has_struct:
            status = "structure_only"  # affinity requested but missing
        else:
            status = "failed"
        rows.append({
            "index": i,
            "smiles": smi,
            "stem": stem,
            "status": status,
            "confidence": entry.get("confidence"),
            "affinity": full_affinity(out_dir, stem) if has_aff else None,
        })
    return rows


def diff_numbers(a: dict, b: dict) -> float:
    worst = 0.0
    for grp in ("confidence", "affinity"):
        for k, va in (a.get(grp) or {}).items():
            vb = (b.get(grp) or {}).get(k)
            if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                worst = max(worst, abs(va - vb))
    return worst


# ---------------------------------------------------------------------------
# modes
# ---------------------------------------------------------------------------
def run_single(
    yamls: list[Path], root: Path, seed: int, extra: list[str], rep: int,
    *, override: bool, skip_bad: bool = True,
) -> dict:
    per: dict[str, float] = {}
    t0 = time.perf_counter()
    for y in yamls:
        out_dir = root / f"rep{rep}" / y.stem
        wall, rc = run(
            boltz_cmd(y, out_dir, seed, extra, 1, override=override, skip_bad=skip_bad),
            out_dir / "boltz.log",
        )
        per[y.stem] = wall
        print(f"[single rep{rep}] {y.stem}: {wall:7.1f}s  {'ok' if rc == 0 else f'FAIL(rc={rc})'}", flush=True)
    return {
        "total_s": time.perf_counter() - t0,
        "per_ligand_s": per,
        "results": collect_many(root / f"rep{rep}"),
    }


def run_batch(
    yamls: list[Path],
    root: Path,
    seed: int,
    extra: list[str],
    batch_size: int,
    rep: int,
    oom_fallback: bool,
    *,
    override: bool = True,
    skip_bad: bool = True,
) -> dict:
    in_dir = root / f"rep{rep}" / "inputs"
    in_dir.mkdir(parents=True, exist_ok=True)
    for y in yamls:
        shutil.copy(y, in_dir / y.name)
    out_dir = root / f"rep{rep}" / "out"
    log = out_dir / "boltz.log"

    bs = max(1, min(batch_size, len(yamls)))
    attempts: list[dict] = []
    ovr = override
    t0 = time.perf_counter()
    while True:
        wall, rc = run(boltz_cmd(in_dir, out_dir, seed, extra, bs, override=ovr, skip_bad=skip_bad), log)
        oom = rc != 0 and bool(OOM_RE.search(log.read_text(errors="replace")))
        attempts.append({"batch_size": bs, "wall_s": round(wall, 1), "rc": rc, "oom": oom})
        tag = "ok" if rc == 0 else ("OOM" if oom else f"FAIL(rc={rc})")
        print(f"[batch  rep{rep}] {len(yamls)} ligands @ bs={bs}: {wall:7.1f}s  {tag}", flush=True)
        if oom and oom_fallback and bs > 1:
            bs = max(1, bs // 2)
            ovr = False  # keep records already written before the OOM; resume the rest
            print(f"[batch  rep{rep}] -> retry at batch_size={bs} (resuming)", flush=True)
            continue
        break

    return {
        "total_s": time.perf_counter() - t0,      # includes any failed OOM attempts
        "predict_wall_s": attempts[-1]["wall_s"],  # the successful run alone
        "returncode": attempts[-1]["rc"],
        "batch_size_used": bs,
        "attempts": attempts,
        "results": collect(out_dir),
    }


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------
def assert_reproducible(label: str, rep0: dict, rep1: dict) -> bool:
    ok = True
    for stem, a in rep0.items():
        b = rep1.get(stem, {})
        if a.get("cif_sha256") != b.get("cif_sha256"):
            print(f"  [{label}] {stem}: CIF differs")
            ok = False
        d = diff_numbers(a, b)
        if d > 0:
            print(f"  [{label}] {stem}: scalar |delta| = {d:.3e}")
            ok = False
    print(f"  [{label}] reproducible across 2 runs: {'YES' if ok else 'NO'}")
    return ok


def cross_mode_table(single: dict, batch: dict) -> None:
    print("\ncross-mode agreement (single vs batch, same seed)")
    print(f"  {'ligand':<16} {'cif match':>10} {'max|delta|':>12}")
    worst = 0.0
    for stem in sorted(single):
        a, b = single[stem], batch.get(stem, {})
        match = "yes" if a.get("cif_sha256") == b.get("cif_sha256") else "no"
        d = diff_numbers(a, b)
        worst = max(worst, d)
        print(f"  {stem:<16} {match:>10} {d:>12.3e}")
    print(f"  {'(worst)':<16} {'':>10} {worst:>12.3e}")


def summary(n: int, single: dict | None, batch: dict | None) -> dict:
    print("\n" + "=" * 60)
    out: dict = {"n_ligands": n}
    if single:
        st = single["total_s"]
        out |= {"single_total_s": round(st, 1), "single_mean_s": round(st / n, 1)}
        print(f"single : {st:8.1f}s total   {st / n:6.1f}s / ligand")
    if batch:
        bt = batch["total_s"]
        out |= {"batch_total_s": round(bt, 1), "batch_mean_s": round(bt / n, 1),
                "batch_size_used": batch["batch_size_used"]}
        print(f"batch  : {bt:8.1f}s total   {bt / n:6.1f}s / ligand   (bs={batch['batch_size_used']})")
    if single and batch:
        sp = single["total_s"] / batch["total_s"]
        saved = single["total_s"] - batch["total_s"]
        out |= {"speedup_x": round(sp, 2), "wall_saved_s": round(saved, 1)}
        if n > 1:
            out["per_job_overhead_s"] = round(saved / (n - 1), 1)
        print(f"speedup: {sp:8.2f}x   ({saved:.1f}s saved)")
        if n > 1:
            print(f"         ~{saved / (n - 1):.1f}s/ligand fixed overhead removed by batching")
    print("=" * 60)
    return out


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--protein", required=True, help="sequence string, or .fasta / .txt path")
    ap.add_argument("--ligands", help="file with one SMILES per line")
    ap.add_argument("--smiles", action="append", help="a SMILES (repeatable)")
    ap.add_argument("--out", required=True, type=Path, help="output root directory")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mode", choices=["both", "single", "batch"], default="batch")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="boltz --batch_size (default: auto from system size + free VRAM)")
    ap.add_argument("--max-batch-size", type=int, default=16, help="cap for the auto heuristic")
    ap.add_argument("--no-oom-fallback", action="store_true",
                    help="do not retry with a halved batch size on CUDA OOM")
    ap.add_argument("--affinity", dest="affinity", action="store_true", default=True)
    ap.add_argument("--no-affinity", dest="affinity", action="store_false")
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--msa", help="a3m/csv path to reuse for the receptor in every job")
    grp.add_argument("--no-msa", action="store_true", help="use the 'empty' single-seq sentinel")
    ap.add_argument("--use-msa-server", action="store_true",
                    help="(default behaviour on cache miss; kept for explicitness)")
    ap.add_argument("--msa-server-url", default="https://api.colabfold.com")
    ap.add_argument("--boltz-arg", action="append", default=[],
                    help="extra arg for boltz predict, repeatable, use = form: --boltz-arg=--sampling_steps=50")
    ap.add_argument("--screen", action="store_true",
                    help="fast screening preset: recycling 1, sampling 50, 1 diffusion sample, "
                    "affinity 50 steps / 1 sample, TF32 matmul, skip full pae/pde npz. "
                    "Your own --boltz-arg values override individual settings.")
    ap.add_argument("--flash-attn", action="store_true", help="pass boltz --flash_attn (SDPA/FA2)")
    ap.add_argument("--check-repro", action="store_true", help="run each mode twice and assert identical")
    ap.add_argument("--resume", action="store_true",
                    help="keep predictions already on disk in --out and only compute what is missing "
                    "(drops boltz --override)")
    ap.add_argument("--abort-on-bad-input", action="store_true",
                    help="abort the whole run if any ligand fails preprocessing "
                    "(default: skip it and keep going)")
    ap.add_argument(
        "--deterministic",
        action="store_true",
        help="pass boltz --deterministic_per_record to every run so batch reproduces "
        "single element-wise (forces structure batch_size=1; slower)",
    )
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(line_buffering=True)  # live logs when redirected to a file
    except (AttributeError, ValueError):
        pass

    seq = read_protein(args.protein)
    ligands = read_ligands(args)
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"receptor: {len(seq)} aa | ligands: {len(ligands)} | seed: {args.seed} | affinity: {args.affinity}")

    msa_ref = resolve_msa(seq, args, args.out)

    yaml_dir = args.out / "yamls"
    yaml_dir.mkdir(exist_ok=True)
    lig_atoms = [heavy_atom_count(s) for s in ligands]
    yamls = []
    for i, smi in enumerate(ligands):
        y = yaml_dir / f"lig{i:04d}.yaml"
        write_yaml(y, seq, smi, msa_ref, args.affinity)
        yamls.append(y)

    # ---- batch size ----
    n_tokens = len(seq) + max(lig_atoms)
    free_gb = gpu_free_gb()
    if args.deterministic:
        bs, why = 1, "forced to 1 by --deterministic (per-record reseed)"
    elif args.batch_size is not None:
        bs, why = args.batch_size, f"explicit --batch-size {args.batch_size}"
    else:
        bs, why = auto_batch_size(n_tokens, len(yamls), free_gb, args.max_batch_size)
    print(f"[batch-size] {why}")

    # preset first, user --boltz-arg last so it wins; --flash_attn / --deterministic appended
    preset: list[str] = []
    if args.screen:
        preset = [
            "--recycling_steps=1", "--sampling_steps=50", "--diffusion_samples=1",
            "--sampling_steps_affinity=50", "--diffusion_samples_affinity=1",
            "--matmul_precision=high", "--no_write_full_pae", "--no_write_full_pde",
        ]
    extra = preset + list(args.boltz_arg)
    if args.flash_attn:
        extra.append("--flash_attn")
    if args.deterministic:
        extra.append("--deterministic_per_record")
    reps = 2 if args.check_repro else 1
    bench: dict = {
        "seed": args.seed,
        "n_ligands": len(yamls),
        "receptor_len": len(seq),
        "max_ligand_atoms": max(lig_atoms),
        "n_tokens_est": n_tokens,
        "gpu_free_gb": round(free_gb, 1) if free_gb else None,
        "batch_size": bs,
        "batch_size_auto": args.batch_size is None,
        "batch_size_reason": why,
        "oom_fallback": not args.no_oom_fallback,
        "extra_args": extra,
    }

    override = not args.resume
    skip_bad = not args.abort_on_bad_input
    bench["resume"] = args.resume
    bench["skip_bad_inputs"] = skip_bad

    single_runs, batch_runs = [], []
    for rep in range(reps):
        if args.mode in ("both", "single"):
            single_runs.append(run_single(
                yamls, args.out / "single", args.seed, extra, rep,
                override=override, skip_bad=skip_bad,
            ))
        if args.mode in ("both", "batch"):
            batch_runs.append(run_batch(
                yamls, args.out / "batch", args.seed, extra, bs, rep, not args.no_oom_fallback,
                override=override, skip_bad=skip_bad,
            ))

    if args.check_repro:
        print("\nreproducibility check (same seed, 2 runs):")
        if single_runs:
            bench["single_reproducible"] = assert_reproducible(
                "single", single_runs[0]["results"], single_runs[1]["results"]
            )
        if batch_runs:
            bench["batch_reproducible"] = assert_reproducible(
                "batch", batch_runs[0]["results"], batch_runs[1]["results"]
            )

    s0 = single_runs[0] if single_runs else None
    b0 = batch_runs[0] if batch_runs else None
    if b0:
        bench["batch_attempts"] = b0["attempts"]
    if s0 and b0:
        cross_mode_table(s0["results"], b0["results"])
    bench["timing"] = summary(len(yamls), s0, b0)
    if s0:
        bench["single_per_ligand_s"] = {k: round(v, 1) for k, v in s0["per_ligand_s"].items()}

    # ---- per-ligand deliverable, written even if a run crashed partway ----
    stems = [y.stem for y in yamls]
    if b0 is not None:
        prod_out, prod_res = args.out / "batch" / "rep0" / "out", b0["results"]
    else:
        prod_out = args.out / "single" / "rep0"
        prod_res = collect_many(prod_out)
    rows = screen_results(ligands, stems, prod_out, prod_res, args.affinity)
    done = sum(r["status"] == "completed" for r in rows)
    bench["completed"] = done
    bench["results"] = rows
    (args.out / "results.json").write_text(json.dumps(
        {"receptor_len": len(seq), "n_ligands": len(rows), "completed": done,
         "seed": args.seed, "affinity": args.affinity, "results": rows},
        indent=2,
    ))
    (args.out / "benchmark.json").write_text(json.dumps(bench, indent=2))
    print(f"\ncompleted {done}/{len(rows)} ligands  "
          f"({sum(r['status'] == 'structure_only' for r in rows)} structure-only, "
          f"{sum(r['status'] == 'failed' for r in rows)} failed)")
    print(f"wrote {args.out / 'results.json'} and {args.out / 'benchmark.json'}")


if __name__ == "__main__":
    main()
