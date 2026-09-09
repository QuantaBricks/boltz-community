#!/usr/bin/env python3
"""EXPERIMENTAL warm-model worker for Boltz-2 (see docs/warm_model_worker.md).

A long-lived process that loads the Boltz-2 structure + affinity checkpoints onto
the GPU once and then answers many prediction requests, so the ~35 s
import + checkpoint-load cost is paid once per worker rather than once per job.

NOT wired into anything. This is option B from the proposal: it duplicates the
orchestration in ``boltz.main.predict`` (Boltz-2 only, GPU only, no
``--use_potentials``, no boltz1) rather than refactoring that function. Treat it
as a review starting point, and validate against ``boltz predict`` on real inputs
(templates, constraints, multimers) before trusting it.

Protocol: one JSON object per line on stdin, one reply per line on stdout.

    <- {"id":"j1","yaml_path":"/abs/in.yaml","out_dir":"/abs/out","seed":42,
        "recycling_steps":3,"sampling_steps":200,"diffusion_samples":1,
        "max_parallel_samples":5,"affinity":true,
        "sampling_steps_affinity":200,"diffusion_samples_affinity":5,
        "deterministic_per_record":false,"batch_size":1}
    -> {"id":"j1","status":"ok","elapsed_s":6.1,"model_cache":"hit",
        "out_dir":".../boltz_results_in"}

The checkpoints stay resident in VRAM for the life of the process. With no
request for --idle-timeout seconds (default 60) the worker prints
{"status":"idle_shutdown"} and exits, freeing all VRAM; the caller spawns a
fresh worker on the next request (paying the ~35 s load once). Set
--idle-timeout 0 to stay up forever.

Start:  python scripts/boltz_worker.py --cache ~/.boltz [--idle-timeout 60]
                                       [--flash-attn] [--no-kernels]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path


def log(*a: object) -> None:
    print(*a, file=sys.stderr, flush=True)


class Worker:
    def __init__(self, cache: Path, *, no_kernels: bool, flash_attn: bool) -> None:
        self.cache = cache.expanduser()
        self.no_kernels = no_kernels
        self.flash_attn = flash_attn
        self._models: dict[tuple, object] = {}  # cache key -> Boltz2 module

        # one-time global setup, mirrors boltz.main.predict
        import torch
        from pytorch_lightning import seed_everything  # noqa: F401  (used per job)

        if not getattr(torch.load, "_boltz_patched", False):
            _orig = torch.load

            def _patched(*a, _orig=_orig, **kw):  # noqa: ANN002, ANN003
                return _orig(*a, **{**kw, "weights_only": False})

            _patched._boltz_patched = True  # type: ignore[attr-defined]
            torch.load = _patched  # type: ignore[assignment]

        for key in ("CUEQ_DEFAULT_CONFIG", "CUEQ_DISABLE_AOT_TUNING"):
            os.environ[key] = os.environ.get(key, "1")
        torch.set_grad_enabled(False)
        torch.set_float32_matmul_precision("highest")

        from boltz.main import download_boltz2

        download_boltz2(self.cache)
        self._torch = torch
        log(f"[worker] ready; cache={self.cache} kernels={not no_kernels} flash_attn={flash_attn}")

    # -- model loading with reuse -------------------------------------------------
    def _structure_model(self, step_scale: float):
        from boltz.model.models.boltz2 import Boltz2
        from boltz.main import (
            Boltz2DiffusionParams,
            BoltzSteeringParams,
            MSAModuleArgs,
            PairformerArgsV2,
        )

        ckpt = self.cache / "boltz2_conf.ckpt"
        diff = Boltz2DiffusionParams()
        diff.step_scale = step_scale
        pf = PairformerArgsV2()
        msa = MSAModuleArgs(subsample_msa=True, num_subsampled_msa=1024, use_paired_feature=True)
        steer = BoltzSteeringParams()
        steer.fk_steering = False
        steer.physical_guidance_update = False
        steer.contact_guidance_update = False  # this worker forces batch_size 1

        key = ("struct", str(ckpt), ckpt.stat().st_mtime_ns, self.no_kernels, self.flash_attn, step_scale)
        if key not in self._models:
            log("[worker] loading structure checkpoint ...")
            t0 = time.perf_counter()
            m = Boltz2.load_from_checkpoint(
                ckpt, strict=True,
                predict_args={},  # set per request below
                map_location="cuda",
                diffusion_process_args=asdict(diff),
                ema=False,
                use_kernels=not self.no_kernels,
                use_flash_attn=self.flash_attn,
                pairformer_args=asdict(pf),
                msa_args=asdict(msa),
                steering_args=asdict(steer),
            )
            m.eval()
            self._models[key] = m
            log(f"[worker] structure ready in {time.perf_counter() - t0:.1f}s")
        return self._models[key], self._models[key] is not None

    def _affinity_model(self, step_scale: float, mw_correction: bool | None):
        from boltz.model.models.boltz2 import Boltz2
        from boltz.main import (
            Boltz2DiffusionParams,
            BoltzSteeringParams,
            MSAModuleArgs,
            PairformerArgsV2,
        )

        ckpt = self.cache / "boltz2_aff.ckpt"
        diff = Boltz2DiffusionParams()
        diff.step_scale = step_scale
        pf = PairformerArgsV2()
        msa = MSAModuleArgs(subsample_msa=True, num_subsampled_msa=1024, use_paired_feature=True)
        steer = BoltzSteeringParams()
        steer.fk_steering = False
        steer.physical_guidance_update = False
        steer.contact_guidance_update = False

        key = ("aff", str(ckpt), ckpt.stat().st_mtime_ns, mw_correction)
        if key not in self._models:
            log("[worker] loading affinity checkpoint ...")
            t0 = time.perf_counter()
            m = Boltz2.load_from_checkpoint(
                ckpt, strict=True,
                predict_args={},
                map_location="cuda",
                diffusion_process_args=asdict(diff),
                ema=False,
                pairformer_args=asdict(pf),
                msa_args=asdict(msa),
                steering_args=asdict(steer),
                affinity_mw_correction=mw_correction,
            )
            m.eval()
            self._models[key] = m
            log(f"[worker] affinity ready in {time.perf_counter() - t0:.1f}s")
        return self._models[key]

    # -- one job ---------------------------------------------------------------
    def handle(self, req: dict) -> dict:
        from pytorch_lightning import Callback, Trainer, seed_everything

        from boltz.data.module.inferencev2 import Boltz2InferenceDataModule
        from boltz.data.types import Manifest
        from boltz.data.write.writer import BoltzAffinityWriter, BoltzWriter
        from boltz.main import (
            check_inputs,
            expand_affinity_records,
            filter_inputs_affinity,
            filter_inputs_structure,
            process_inputs,
            _resolve_affinity_max_parallel_samples,
        )

        yaml_path = Path(req["yaml_path"]).expanduser()
        out_root = Path(req["out_dir"]).expanduser()
        seed = int(req.get("seed", 42))
        det = bool(req.get("deterministic_per_record", False))
        step_scale = float(req.get("step_scale", 1.5))
        want_affinity = bool(req.get("affinity", True))

        seed_everything(seed)

        class _Reseed(Callback):
            def on_predict_batch_start(self, *a: object, **k: object) -> None:
                self._t.manual_seed(seed)
                if self._t.cuda.is_available():
                    self._t.cuda.manual_seed_all(seed)

        _Reseed._t = self._torch  # type: ignore[attr-defined]

        out_dir = out_root / f"boltz_results_{yaml_path.stem}"
        out_dir.mkdir(parents=True, exist_ok=True)

        data = check_inputs(yaml_path)
        process_inputs(
            data=data,
            out_dir=out_dir,
            ccd_path=self.cache / "ccd.pkl",
            mol_dir=self.cache / "mols",
            use_msa_server=False,
            msa_server_url="https://api.colabfold.com",
            msa_pairing_strategy="greedy",
            boltz2=True,
            preprocessing_threads=max(1, (os.cpu_count() or 2) // 2),
            max_msa_seqs=8192,
        )
        manifest = Manifest.load(out_dir / "processed" / "manifest.json")
        processed_dir = out_dir / "processed"

        callbacks: list = []
        if det:
            callbacks.append(_Reseed())

        # ---- structure phase ----
        writer = BoltzWriter(
            data_dir=processed_dir / "structures",
            output_dir=out_dir / "predictions",
            output_format="mmcif",
            boltz2=True,
        )
        trainer = Trainer(
            default_root_dir=out_dir,
            strategy="auto",
            callbacks=[writer, *callbacks],
            accelerator="gpu",
            devices=1,
            precision="bf16-mixed",
        )

        filtered = filter_inputs_structure(manifest=manifest, outdir=out_dir, override=True)
        model, _ = self._structure_model(step_scale)
        model.predict_args = {
            "recycling_steps": int(req.get("recycling_steps", 3)),
            "sampling_steps": int(req.get("sampling_steps", 200)),
            "diffusion_samples": int(req.get("diffusion_samples", 1)),
            "max_parallel_samples": req.get("max_parallel_samples", 5),
            "write_confidence_summary": True,
            "write_full_pae": bool(req.get("write_full_pae", False)),
            "write_full_pde": bool(req.get("write_full_pde", False)),
        }
        dm = Boltz2InferenceDataModule(
            manifest=filtered,
            target_dir=processed_dir / "structures",
            msa_dir=processed_dir / "msa",
            mol_dir=self.cache / "mols",
            num_workers=2,
            batch_size=1,
            constraints_dir=(processed_dir / "constraints") if (processed_dir / "constraints").exists() else None,
            template_dir=(processed_dir / "templates") if (processed_dir / "templates").exists() else None,
            extra_mols_dir=(processed_dir / "mols") if (processed_dir / "mols").exists() else None,
            pin_memory=True,
        )
        trainer.predict(model, datamodule=dm, return_predictions=False)

        # ---- affinity phase ----
        if want_affinity and any(r.affinity for r in manifest.records):
            aff_manifest = filter_inputs_affinity(
                manifest=expand_affinity_records(manifest), outdir=out_dir, override=True
            )
            if aff_manifest.records:
                aff_writer = BoltzAffinityWriter(
                    data_dir=processed_dir / "structures",
                    output_dir=out_dir / "predictions",
                )
                aff_dm = Boltz2InferenceDataModule(
                    manifest=aff_manifest,
                    target_dir=out_dir / "predictions",
                    msa_dir=processed_dir / "msa",
                    mol_dir=self.cache / "mols",
                    num_workers=2,
                    batch_size=1,
                    constraints_dir=(processed_dir / "constraints") if (processed_dir / "constraints").exists() else None,
                    template_dir=(processed_dir / "templates") if (processed_dir / "templates").exists() else None,
                    extra_mols_dir=(processed_dir / "mols") if (processed_dir / "mols").exists() else None,
                    override_method="other",
                    affinity=True,
                    pin_memory=True,
                )
                aff_model = self._affinity_model(step_scale, req.get("affinity_mw_correction"))
                aff_model.predict_args = {
                    "recycling_steps": 5,
                    "sampling_steps": int(req.get("sampling_steps_affinity", 200)),
                    "diffusion_samples": int(req.get("diffusion_samples_affinity", 5)),
                    "max_parallel_samples": _resolve_affinity_max_parallel_samples(
                        req.get("max_parallel_samples", 5),
                        int(req.get("diffusion_samples_affinity", 5)),
                    ),
                    "write_confidence_summary": False,
                    "write_full_pae": False,
                    "write_full_pde": False,
                }
                trainer.callbacks[0] = aff_writer
                trainer.predict(aff_model, datamodule=aff_dm, return_predictions=False)

        return {"out_dir": str(out_dir)}


def main() -> None:
    import select

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, default=Path.home() / ".boltz")
    ap.add_argument("--no-kernels", action="store_true")
    ap.add_argument("--flash-attn", action="store_true")
    ap.add_argument(
        "--idle-timeout", type=float, default=60.0,
        help="exit (releasing all VRAM) after this many seconds with no request; 0 = never",
    )
    args = ap.parse_args()

    # stdout is the protocol channel (one JSON object per line). Everything boltz
    # / Lightning / tqdm prints must not land there -> send it to stderr.
    import contextlib

    proto = sys.stdout
    with contextlib.redirect_stdout(sys.stderr):
        worker = Worker(args.cache, no_kernels=args.no_kernels, flash_attn=args.flash_attn)
    print(json.dumps({"status": "ready", "idle_timeout_s": args.idle_timeout}), file=proto, flush=True)

    while True:
        # Block for a request, but no longer than the idle timeout. When it
        # fires we return, the process exits, and the CUDA context (with the
        # resident checkpoints) is torn down by the OS -> VRAM freed. The caller
        # spawns a fresh worker on the next request, paying the load cost once.
        if args.idle_timeout > 0:
            ready, _, _ = select.select([sys.stdin], [], [], args.idle_timeout)
            if not ready:
                log(f"[worker] idle {args.idle_timeout:.0f}s, shutting down (releasing VRAM)")
                print(json.dumps({"status": "idle_shutdown"}), file=proto, flush=True)
                return
        line = sys.stdin.readline()
        if not line:  # stdin closed
            log("[worker] stdin closed, exiting")
            return
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            print(json.dumps({"status": "error", "error": f"bad json: {e}"}), file=proto, flush=True)
            continue
        rid = req.get("id")
        t0 = time.perf_counter()
        n_cached = len(worker._models)
        try:
            with contextlib.redirect_stdout(sys.stderr):
                res = worker.handle(req)
            print(json.dumps({
                "id": rid, "status": "ok",
                "elapsed_s": round(time.perf_counter() - t0, 2),
                "model_cache": "hit" if len(worker._models) == n_cached else "miss",
                **res,
            }), file=proto, flush=True)
        except Exception as e:  # noqa: BLE001
            log(traceback.format_exc())
            print(json.dumps({
                "id": rid, "status": "error", "error": str(e),
                "elapsed_s": round(time.perf_counter() - t0, 2),
            }), file=proto, flush=True)


if __name__ == "__main__":
    main()
