# Proposal: warm in-process model worker

**Status:** proposal + non-wired skeleton (`scripts/boltz_worker.py`). Not enabled
anywhere. Needs review before it goes near `main.py` or the prediction server.

## Problem

Every `boltz predict` invocation reloads the checkpoints onto the GPU. Measured
on an RTX 5090 (117 aa receptor + ligand, affinity on):

| phase | time |
|---|---|
| process start + `import torch` / rdkit / lightning + CUDA init | ~12–15 s |
| `Boltz2.load_from_checkpoint` (structure, 2.3 GB) → GPU | ~15–20 s |
| affinity checkpoint (2.1 GB) → GPU | ~10–15 s |
| **fixed overhead per invocation** | **~35 s** |
| actual compute (this small system, reduced sampling) | ~2–5 s |

`scripts/batch_predict.py` already removes this for a *known* ligand set by
handing `boltz predict` a directory (model loads once for the whole set). What it
does **not** help:

* a long-running screening service that receives ligands over time,
* re-running the same receptor across many separate calls,
* the `server.py` FastAPI path, which still spawns one `boltz predict`
  subprocess per request.

A process that keeps the model resident in VRAM answers all three: the ~35 s is
paid once per worker lifetime instead of once per request.

## Why this is not just "call `predict()` in a loop"

`boltz.main.predict` is a single `click` command (`# noqa: C901, PLR0915,
PLR0912`) that, from top to bottom:

1. patches `torch.load`, sets `CUEQ_*` env, `seed_everything`, grad off,
2. `download_boltz2(cache)`, `check_inputs`, `process_inputs` (featurise + MSA),
3. `Manifest.load`, `filter_inputs_structure`,
4. builds `Boltz2DiffusionParams` / `PairformerArgsV2` / `MSAModuleArgs` /
   `BoltzSteeringParams`, a `BoltzWriter`, a `Trainer`,
5. `Boltz2.load_from_checkpoint(...)` ← **the expensive line**,
6. `trainer.predict(...)` for structure,
7. repeats 4–6 with the affinity checkpoint and `BoltzAffinityWriter`.

Steps 5 (both) are the only ones worth caching. Everything else must still run
per request. So the worker needs to reproduce steps 1–7 but substitute a cached
model object at step 5 when the relevant args match.

## Two ways to build it

### A. Extract a shared helper in `main.py` (clean, invasive)

Move steps 4–7 into `_run_prediction(processed, manifest, out_dir, cfg,
model_cache=None)`. `predict()` calls it with `model_cache=None` → byte-identical
CLI behaviour. The worker calls it with a live `model_cache` dict.

* **Pro:** one code path, no drift.
* **Con:** touches a function the production `server.py` depends on. Requires the
  full regression suite (`tests/test_regression_v2.py`,
  `tests/test_inference_smoke.py`, `tests/test_affinity.py`,
  `tests/test_cli.py`) to pass unchanged, on GPU, before merge.

### B. Standalone worker that duplicates orchestration (safe, some drift)

`scripts/boltz_worker.py` replicates steps 1–7 itself and never imports
`predict`. `main.py` is untouched.

* **Pro:** zero risk to the CLI / server.
* **Con:** ~120 lines of orchestration duplicated; must be kept in sync by hand
  when `predict()` changes. Higher chance of subtle divergence for input shapes
  the skeleton was not tested against (templates, constraints, multimers,
  boltz1, potentials).

The committed skeleton is **B**, deliberately minimal (Boltz-2, structure +
affinity, no potentials/templates), as a starting point for review. Recommended
end state is **A** once the regression suite can gate it.

## Model cache key

Reuse a loaded `Boltz2` only when everything that changes weights or graph shape
is identical:

```
key = (
    checkpoint_path, checkpoint_mtime,
    no_kernels, flash_attn,
    tuple(sorted(asdict(pairformer_args).items())),
    tuple(sorted(asdict(msa_args).items())),          # subsample_msa, num_subsampled_msa, ...
    step_scale,
    affinity_mw_correction,   # affinity model only
)
```

`predict_args` (recycling_steps, sampling_steps, diffusion_samples,
max_parallel_samples, write_* ) are set on `model_module.predict_args` **after**
load, so they can change between requests without a reload — the worker just
reassigns them.

`seed_everything` / `--deterministic_per_record` still run per request, so
reproducibility is unaffected by model reuse.

## Protocol (skeleton)

One JSON object per line on stdin, one per line on stdout:

```
<- {"id":"j1","yaml_path":"/abs/in.yaml","out_dir":"/abs/out","seed":42,
    "recycling_steps":3,"sampling_steps":200,"diffusion_samples":1,
    "affinity":true,"deterministic_per_record":false}
-> {"id":"j1","status":"ok","out_dir":"/abs/out/boltz_results_in","elapsed_s":6.1,
    "model_cache":"hit"}
```

`server.py` would talk to one long-lived worker per GPU instead of spawning
`boltz predict`; `batch_predict.py` could target it for `--mode batch` to also
skip the per-run reload.

## Test plan before enabling

1. `pytest tests/test_regression_v2.py tests/test_affinity.py
   tests/test_inference_smoke.py -q` on GPU — unchanged.
2. Worker vs `boltz predict` on the same YAML+seed → identical CIF sha256 and
   identical `confidence_*.json` / `affinity_*.json` (with
   `--deterministic_per_record` to remove RNG-stream noise).
3. Worker job #2 (cache hit) wall time ≈ job #1 − ~35 s.
4. Long-run soak: 200 sequential jobs, watch `nvidia-smi` for VRAM creep
   (fragmentation / leak), confirm results still match #2.

## Expected payoff

For a screening service, per-ligand wall time drops from `~35 s + compute` to
`~compute` after the first request. On the small benchmark that is ~40 s → ~5 s
(~8×); on realistic systems (bigger receptor, real MSA, default 200 sampling
steps, 5 affinity samples) the compute term dominates more, so expect ~1.5–3×
end-to-end, with the full ~35 s still saved on every request.
