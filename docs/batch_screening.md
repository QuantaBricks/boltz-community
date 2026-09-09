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
