# Batch ligand screening (`scripts/batch_predict.py`)

Cofold one receptor against many ligands and get a Boltz‑2 affinity per ligand,
running them through a single `boltz predict` process instead of one process per
ligand.

## Why

`boltz predict <one.yaml>` reloads the ~2.3 GB structure checkpoint (and the
~2.1 GB affinity checkpoint) onto the GPU every invocation. Measured on an RTX
5090 that fixed cost is **~35 s per ligand**, dwarfing the actual compute for a
small system. Pointing `boltz predict` at a *directory* of YAMLs loads the model
once and reuses it for every record; `--batch_size K` additionally packs K
structure inferences into one forward pass (the affinity phase always runs one
record at a time).

Measured on an RTX 5090, affinity on, `--seed 42`:

| receptor | MSA | sampling | ligands | per‑process | batched | speed‑up |
|---|---|---|--:|--:|--:|--:|
| 117 aa | none | 20/20 steps | 3 | 114 s | 43 s | 2.7× |
| 384 aa | real (cached) | 100/100 steps, recycle 2 | 6 | 281 s | 93 s | 3.0× |

The per‑ligand fixed overhead removed is ~36–38 s in every run (process start +
`import torch` + both checkpoints → GPU). It is independent of the ligand count,
so the speed‑up grows with N and asymptotes at `(load + compute) / compute`
(≈5× for the 384 aa case, ≈15× for the tiny one). Extrapolated:

| ligands | 117 aa | 384 aa |
|--:|--:|--:|
| 10 | ~6× | ~4× |
| 100 | ~13× | ~5× |

## Usage

```bash
python scripts/batch_predict.py \
    --protein receptor.fasta \        # sequence string, .fasta, or .txt
    --ligands ligands.smi \           # one SMILES per line ("SMILES <name>" ok)
    --out runs/screen1 \
    --seed 42 \
    --mode batch
```

Each ligand becomes `protein (chain A) + ligand (chain B)` with
`properties: [{affinity: {binder: B}}]` (drop with `--no-affinity`).

Results land under `runs/screen1/batch/rep0/out/boltz_results_inputs/predictions/<ligNNNN>/`,
plus two machine‑readable files at the root of `--out`:

* **`results.json`** — one row per ligand, **written even if the run crashed
  partway**: `{index, smiles, stem, status, confidence, affinity}` where `status`
  is `completed` / `structure_only` (structure done, affinity missing) /
  `failed`. `boltz predict` writes each record's outputs as it finishes, so a
  mid‑run OOM / RDKit abort / kill keeps every ligand computed so far.
* **`benchmark.json`** — timings, the batch‑size decision, OOM‑retry attempts,
  reproducibility flags, and the same per‑ligand rows.

`--skip_bad_inputs` is passed to `boltz` by default (one bad SMILES is skipped,
not fatal — `--abort-on-bad-input` to opt out). `--resume` re‑runs a `--out`
directory and computes only the ligands still missing.

### MSA

No flag needed in the common case. Resolution order:

1. `--msa PATH` — explicit a3m/csv, reused for every job.
2. **Cache hit** — `$BOLTZ_MSA_CACHE` (default `~/.boltz/msa_cache/`), keyed by
   `sha256(sequence)[:24]`, the same scheme `server.py` uses, so the caches
   interoperate.
3. `--no-msa` — the `empty` single‑sequence sentinel (fast, low accuracy).
4. Otherwise — fetch once from `--msa-server-url`, then drop a copy in the cache
   so the next run is a hit.

### Auto batch size

`--batch_size` is chosen from the estimated token count
(`receptor residues + largest ligand heavy‑atom count`) and the GPU's free
memory:

| n_tokens | base batch size |
|---------:|----------------:|
| ≤ 300    | 8               |
| ≤ 500    | 4               |
| ≤ 700    | 2               |
| > 700    | 1 (never pack)  |

then multiplied by `free_GB / 24` and capped by `--max-batch-size` (16) and the
ligand count. Pass an explicit `--batch-size N` to override the heuristic.

If a batch still hits CUDA OOM, the run is retried with the batch size halved
until it fits (down to 1); disable with `--no-oom-fallback`. Every attempt is
recorded in `benchmark.json` under `batch_attempts`.

## Reproducibility

* The featurizer uses a hard‑coded local RNG (seed 42), so featurization is
  identical regardless of mode or `--seed`.
* Diffusion sampling draws from the **global** torch RNG, which `boltz` seeds
  once per process (`seed_everything(--seed)`) and then runs the structure phase
  over *all* records before the affinity phase. Consequences:
  * **Same mode, same seed, same ordering → byte‑identical on re‑run.**
    `--check-repro` runs each mode twice and asserts identical CIF hashes and
    confidence/affinity numbers.
  * **batch vs single differ slightly per ligand** (~0.05–0.17 in
    `affinity_pred_value`, ≤0.02 in `affinity_probability_binary`) because a
    ligand in a batch sees RNG left by the records before it. This is inherent
    to how `boltz` sequences the RNG, not a bug.

### Exact cross‑mode reproducibility

`scripts/batch_predict.py --deterministic` (or `boltz predict
--deterministic_per_record --seed S` directly) re‑seeds the global RNG to `S`
before every record and before the affinity phase and disables cuDNN
autotuning, so a record no longer inherits the RNG stream left by the records
before it.

Measured effect (117 aa, 3 ligands, seed 42): cross‑mode `affinity_pred_value`
disagreement drops from ~0.05–0.17 to **~0.0–0.08** (one ligand matched
exactly), and `affinity_probability_binary` — the binder/decoy score you rank
on — stays within ~0.01 either way. A small residual remains: `boltz` runs the
structure phase over all records before the affinity phase, and some GPU
reductions are not bit‑reproducible across different execution histories in one
process. So this tightens agreement, it does not guarantee bit‑identity.

Cost: forces structure `batch_size=1`, giving up the intra‑forward batching
speed‑up (the load‑amortisation speed‑up is unaffected). Use it when a batched
number must line up with a single‑ligand rerun; leave it off for throughput
screening, where the batched numbers are already self‑consistent and fully
reproducible run‑to‑run.

## Benchmark mode

`--mode both` runs the ligand set both ways and prints a timing summary plus a
cross‑mode agreement table. `--mode single` / `--mode batch` run just one.

## Speed knobs — measured (RTX 5090, 20 ligands, 384 aa, bs=1, affinity)

| config | wall | peak VRAM | note |
|---|--:|--:|---|
| baseline (recycle 1, sampling 50, writes pae/pde) | 153 s | 9.3 GB | |
| `--screen` (adds TF32 matmul + skips pae/pde npz) | **144 s** | 9.3 GB | ~6%, from TF32; safe |
| `--screen --flash-attn` | 187 s | 15.9 GB | **regression** on sm_120 — do not use here |
| `--screen` with TF32 off | 152 s | 9.3 GB | confirms the 6% is TF32 |

- **`--screen`** is the recommended screening preset. It is `--matmul_precision=high`
  (TF32) + `recycling_steps 1` + `sampling_steps 50` + one diffusion sample +
  affinity 50 steps / 1 sample + `--no_write_full_pae/pde`. Your own `--boltz-arg`
  values override any of these.
- **`--flash_attn` hurts on this GPU** (Blackwell / sm_120): PyTorch SDPA does not
  pick a faster kernel for these shapes and adds reshape + memory overhead. It may
  still help on Ampere (A100) — measure before enabling.
- **cuEquivariance kernels** are not installed on the hrs box and its torch is
  `cu130` (no matching `cuequivariance-ops-torch` wheel), so the fused
  trimul/triangle-attention path is unavailable here; boltz falls back to PyTorch.
  On A100 with a `cu12` build this is a real win.
- **`--batch_size K>1` does not speed up affinity screening** (measured flat
  150–153 s for K=1/2/4/8) — the affinity phase is serial at `batch_size=1` and
  the structure phase is not the bottleneck at low sampling. K only raises peak
  VRAM (K=8 → 31 GB). Leave it at 1 unless doing structure-only at high sampling.
- **Two boltz processes on one GPU is slower** (160 s vs 149 s for 20 ligands) —
  a single process already saturates the card; a second just adds context-switch
  overhead. Multi-process only helps across multiple GPUs.

## Affinity recycling (`--recycling_steps_affinity`)

Upstream hardcodes the affinity trunk to 5 recycling steps; this fork exposes it.
Measured (20 ligands, 384 aa, `--screen`, vs steps=5):

| steps | wall | prob_binary Δ (mean / max) | pred_value Δ (mean / max) |
|--:|--:|--|--|
| 5 | 144 s | — | — |
| 3 | 136 s (-5%) | 0.020 / 0.071 | 0.15 / 0.35 |
| 1 | 127 s (-12%) | 0.033 / 0.085 | 0.21 / 0.61 |

`--boltz-arg=--recycling_steps_affinity=3` is a reasonable further cut if you rank
on `affinity_probability_binary` (mean shift ~0.02); `affinity_pred_value`
(log10 IC50) drifts more, so keep 5 if that number matters. Not included in
`--screen` by default.

## Where the time goes, and what's left

At `--screen` settings the per-ligand compute is ~5.2 s (384 aa receptor), split
fairly evenly between the structure and affinity forwards; there is no single
dominant term left to cut with flags. Config tuning tops out around 10–15% over
the already-lean `--screen` baseline (TF32 ~6%, affinity recycling ~5%).

Further speed-up needs one of:

* **Batched affinity** — run N pocket-crop affinity forwards together instead of
  serially. The affinity crop is small (~pocket only), so unlike the
  full-receptor structure forward it may not saturate the GPU at batch 1, i.e.
  there may be real headroom. Needs `BoltzAffinityWriter` (indexes `[0]` /
  `.item()`), the affinity forward, and the datamodule collate audited for
  `batch_size>1`; gate behind `--batch_size_affinity` and run the regression
  suite. Estimated up to ~1.5–1.7× on an affinity-heavy 100-ligand batch **if**
  the headroom is real.
* **Trunk / MSA-module reuse across ligands** for a fixed receptor — research
  effort, big for large same-receptor campaigns.
* **A second GPU** — `--devices 0,1` shards records across cards, near-linear;
  multi-process on *one* card does not help (measured).
