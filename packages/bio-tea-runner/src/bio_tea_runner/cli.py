# Copyright 2026 tznurmin
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

from bio_tea_runner.training.cli import exp1_report_main as _exp1_report_main
from bio_tea_runner.training.cli import exp1_validate_main as _exp1_validate_main
from bio_tea_runner.training.experiment_runner import run_experiment_matrix
from bio_tea_runner.training.exp1_runner import run_exp1_matrix
from bio_tea_runner.training.exp1_runner import validate_exp1_source_root
from bio_tea_runner.training.run_reporting import build_exp1_report_from_summary
from bio_tea_runner.training.calibration import (
    apply_epoch_overrides_to_config,
    load_per_epoch_rows,
    summarize_calibration_for_model,
    write_calibrated_config_and_manifest,
)


def _load_config(path: Path) -> dict:
    if path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is required to load YAML configs")
        return json.loads(json.dumps(yaml.safe_load(path.read_text(encoding="utf-8")) or {}))
    return dict(json.loads(path.read_text(encoding="utf-8")) or {})


def _dump_config(path: Path, obj: dict) -> None:
    if path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is required to write YAML configs")
        text = yaml.safe_dump(obj, sort_keys=False)
    else:
        text = json.dumps(obj, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _csv(v: str | None) -> list[str] | None:
    if not v:
        return None
    out = [x.strip() for x in str(v).split(",") if x.strip()]
    return out or None


def _csv_int(v: str | None) -> list[int] | None:
    vals = _csv(v)
    if not vals:
        return None
    out: list[int] = []
    for raw in vals:
        try:
            out.append(int(raw))
        except Exception as e:
            raise ValueError(f"Invalid integer value in CSV list: {raw}") from e
    return out or None


def _json_arg(v: str | None) -> dict | None:
    if not v:
        return None
    s = str(v).strip()
    if not s:
        return None
    if s.startswith("{"):
        return dict(json.loads(s))
    p = Path(s)
    if not p.exists():
        raise RuntimeError(f"report-ci path does not exist: {p}")
    return _load_config(p)


def _apply_sealed_offline_overrides(cfg: dict) -> None:
    tr = dict(cfg.get("training") or {})
    hf = dict(tr.get("hf") or {})
    hf["preload_assets"] = True
    hf["seal_network_after_preload"] = True
    hf["require_asset_manifest_match"] = True
    hf["write_asset_manifest"] = True
    tr["hf"] = hf
    cfg["training"] = tr


def _apply_hparam_overrides(
    cfg: dict,
    *,
    experiment_id: str,
    batch_size: int | None,
    per_device_train_batch_size: int | None,
    per_device_eval_batch_size: int | None,
) -> None:
    if batch_size is not None and batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if per_device_train_batch_size is not None and per_device_train_batch_size <= 0:
        raise ValueError("--per-device-train-batch-size must be > 0")
    if per_device_eval_batch_size is not None and per_device_eval_batch_size <= 0:
        raise ValueError("--per-device-eval-batch-size must be > 0")

    train_bs = per_device_train_batch_size if per_device_train_batch_size is not None else batch_size
    eval_bs = per_device_eval_batch_size if per_device_eval_batch_size is not None else batch_size
    if train_bs is None and eval_bs is None:
        return

    tr = dict(cfg.get("training") or {})
    exp = dict(tr.get(str(experiment_id)) or {})
    hparams = dict(exp.get("hparams") or {})
    if train_bs is not None:
        hparams["per_device_train_batch_size"] = int(train_bs)
    if eval_bs is not None:
        hparams["per_device_eval_batch_size"] = int(eval_bs)
    exp["hparams"] = hparams
    tr[str(experiment_id)] = exp
    cfg["training"] = tr


def _apply_seeds_override(cfg: dict, *, experiment_id: str, seeds: list[int] | None) -> None:
    if not seeds:
        return
    tr = dict(cfg.get("training") or {})
    exp = dict(tr.get(str(experiment_id)) or {})
    exp["seeds"] = [int(s) for s in seeds]
    tr[str(experiment_id)] = exp
    cfg["training"] = tr


def _apply_epochs_override(cfg: dict, *, experiment_id: str, epochs: int | None) -> None:
    if epochs is None:
        return
    if int(epochs) <= 0:
        raise ValueError("--epochs must be > 0")
    tr = dict(cfg.get("training") or {})
    exp = dict(tr.get(str(experiment_id)) or {})
    hparams = dict(exp.get("hparams") or {})
    hparams["epochs"] = int(epochs)
    exp["hparams"] = hparams
    tr[str(experiment_id)] = exp
    cfg["training"] = tr


def _task_experiment_root_from_job(job: dict, *, experiment_id: str) -> Path | None:
    t = str(job.get("task_experiment_root") or "").strip()
    if t:
        return Path(t)
    if str(experiment_id) == "exp1":
        t1 = str(job.get("task_exp1_root") or "").strip()
        if t1:
            return Path(t1)

    run_dir_raw = str(job.get("run_dir") or "").strip()
    if not run_dir_raw:
        return None
    p = Path(run_dir_raw)
    cur = p
    exp_id = str(experiment_id)
    while cur != cur.parent:
        if cur.name == "runs" and cur.parent.name == exp_id:
            return cur.parent
        cur = cur.parent
    return None


def _group_summary_by_task_experiment_root(summary: dict, *, experiment_id: str) -> list[tuple[Path, dict]]:
    groups: dict[str, dict] = {}
    for j in list(summary.get("jobs") or []):
        root = _task_experiment_root_from_job(j, experiment_id=experiment_id)
        if root is None:
            continue
        k = str(root)
        groups.setdefault(k, {"jobs": []})
        groups[k]["jobs"].append(j)

    out: list[tuple[Path, dict]] = []
    for k in sorted(groups.keys()):
        g = groups[k]
        rows = list(g["jobs"])
        n_ok = 0
        n_failed = 0
        n_skipped = 0
        for row in rows:
            st = str((row or {}).get("status") or "completed").strip().lower()
            if st == "skipped":
                n_skipped += 1
            elif st == "failed":
                n_failed += 1
            else:
                n_ok += 1
        sub = {
            "source_root": summary.get("source_root"),
            "experiment_id": str(experiment_id),
            "n_jobs": len(rows),
            "n_ok": n_ok,
            "n_failed": n_failed,
            "n_skipped": n_skipped,
            "jobs": rows,
        }
        out.append((Path(k), sub))
    return out


def _run_experiment_main(
    *,
    argv: list[str] | None,
    experiment_id: str,
    runner_fn,
) -> int:
    exp_id = str(experiment_id)
    ap = argparse.ArgumentParser(description=f"Run {exp_id} training/evaluation matrix.")
    ap.add_argument("--config", required=True, type=str)
    ap.add_argument("--source-root", default=None, type=str, help="Optional base dataset root override")
    ap.add_argument("--tasks", default=None, type=str, help="Comma-separated task names")
    ap.add_argument("--sets", default=None, type=str, help="Comma-separated set names (for example set1,set2)")
    ap.add_argument("--seeds", default=None, type=str, help="Comma-separated seed integers (for example 0,1,2)")
    ap.add_argument("--epochs", default=None, type=int, help="Override max epochs for this run")
    ap.add_argument("--summary-out", default=None, type=str, help="Optional output path for run summary JSON")
    ap.add_argument("--report-out", default=None, type=str, help="Optional output path for aggregated run report JSON")
    ap.add_argument(
        "--report-ci",
        default=None,
        type=str,
        help="Optional CI config JSON object or path to JSON/YAML file for aggregated report",
    )
    ap.add_argument(
        "--report-point",
        default="mean",
        choices=["mean", "median"],
        type=str,
        help="Point estimator for aggregated report",
    )
    ap.add_argument(
        "--sealed-offline",
        action="store_true",
        help="Force preload+checksum-check+sealed-offline HF runtime overrides for this run",
    )
    ap.add_argument(
        "--batch-size",
        default=None,
        type=int,
        help="Override per-device train/eval batch size for this run",
    )
    ap.add_argument(
        "--per-device-train-batch-size",
        default=None,
        type=int,
        help="Override per-device train batch size for this run",
    )
    ap.add_argument(
        "--per-device-eval-batch-size",
        default=None,
        type=int,
        help="Override per-device eval batch size for this run",
    )
    args = ap.parse_args(argv)

    cfg = _load_config(Path(args.config))
    if bool(args.sealed_offline):
        _apply_sealed_offline_overrides(cfg)
    _apply_hparam_overrides(
        cfg,
        experiment_id=exp_id,
        batch_size=args.batch_size,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
    )
    _apply_seeds_override(cfg, experiment_id=exp_id, seeds=_csv_int(args.seeds))
    _apply_epochs_override(cfg, experiment_id=exp_id, epochs=args.epochs)
    report_ci_cfg = _json_arg(args.report_ci)
    report_point = str(args.report_point)

    kwargs = {
        "cfg": cfg,
        "tasks": _csv(args.tasks),
        "set_names": _csv(args.sets),
        "source_root": Path(args.source_root) if args.source_root else None,
    }
    if runner_fn is run_experiment_matrix:
        kwargs["experiment_id"] = exp_id

    summary = runner_fn(**kwargs)

    out = json.dumps(summary, indent=2, sort_keys=True)
    print(out)
    if args.summary_out:
        p = Path(args.summary_out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(out + "\n", encoding="utf-8")

    if args.report_out:
        report = build_exp1_report_from_summary(
            summary,
            ci_cfg=report_ci_cfg,
            point_estimator=report_point,
        )
        rp = Path(args.report_out)
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    for exp_root, sub_summary in _group_summary_by_task_experiment_root(summary, experiment_id=exp_id):
        exp_root.mkdir(parents=True, exist_ok=True)
        (exp_root / "run_summary.json").write_text(
            json.dumps(sub_summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        sub_report = build_exp1_report_from_summary(
            sub_summary,
            ci_cfg=report_ci_cfg,
            point_estimator=report_point,
        )
        (exp_root / "run_report.json").write_text(
            json.dumps(sub_report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


def run_exp1_main(argv: list[str] | None = None) -> int:
    return _run_experiment_main(argv=argv, experiment_id="exp1", runner_fn=run_exp1_matrix)


def run_exp2_main(argv: list[str] | None = None) -> int:
    return _run_experiment_main(argv=argv, experiment_id="exp2", runner_fn=run_experiment_matrix)


def run_exp3_main(argv: list[str] | None = None) -> int:
    return _run_experiment_main(argv=argv, experiment_id="exp3", runner_fn=run_experiment_matrix)


def check_exp1_source_root_main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Validate that an Exp1 source root contains all required train/eval artifacts.")
    ap.add_argument("--config", required=True, type=str)
    ap.add_argument("--source-root", required=True, type=str)
    ap.add_argument("--tasks", default=None, type=str, help="Comma-separated task names")
    ap.add_argument("--sets", default=None, type=str, help="Comma-separated set names")
    ap.add_argument("--output", default=None, type=str)
    args = ap.parse_args(argv)

    cfg = _load_config(Path(args.config))
    tasks = _csv(args.tasks)
    sets = _csv(args.sets)
    if not tasks:
        raise RuntimeError("--tasks is required for source-root validation")
    if not sets:
        raise RuntimeError("--sets is required for source-root validation")

    report = validate_exp1_source_root(
        cfg=cfg,
        tasks=tasks,
        set_names=sets,
        source_root=Path(args.source_root),
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output:
        p = Path(args.output)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text + "\n", encoding="utf-8")
    return 0 if bool(report.get("ok", False)) else 2


def exp1_report_main(argv: list[str] | None = None) -> int:
    return _exp1_report_main(argv)


def exp1_validate_main(argv: list[str] | None = None) -> int:
    return _exp1_validate_main(argv)


def report_main(argv: list[str] | None = None) -> int:
    return _exp1_report_main(argv)


def validate_main(argv: list[str] | None = None) -> int:
    return _exp1_validate_main(argv)


def calibrate_exp1_main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Calibrate per-model Exp1 epochs from per-epoch run metrics and emit editable config + hash manifest.",
    )
    ap.add_argument("--config", required=True, type=str, help="Source config used for runner settings")
    ap.add_argument("--results-root", default="results", type=str)
    ap.add_argument("--experiment-id", default="exp1", type=str)
    ap.add_argument("--tasks", default=None, type=str, help="Optional comma-separated task filter")
    ap.add_argument("--model-ids", default=None, type=str, help="Optional comma-separated model id filter")
    ap.add_argument("--train-variants", default="all", type=str, help="Comma-separated train variants to include")
    ap.add_argument("--eval-split", default="dev", type=str, help="Eval split used for calibration rows")
    ap.add_argument(
        "--objective",
        default="balanced_f1",
        choices=["balanced_f1", "unaugmented_f1", "augmented_exclusive_f1", "balanced_loss"],
        type=str,
    )
    ap.add_argument("--min-epoch", default=1, type=int)
    ap.add_argument("--max-epoch", default=None, type=int)
    ap.add_argument("--exclude-collapsed", action="store_true")
    ap.add_argument("--collapse-epsilon", default=0.0, type=float)
    ap.add_argument("--allow-missing-models", action="store_true")
    ap.add_argument("--output-config", required=True, type=str, help="Path for calibrated editable config")
    ap.add_argument("--manifest-out", default=None, type=str, help="Optional calibration manifest path")
    args = ap.parse_args(argv)

    src_cfg_path = Path(args.config)
    cfg = _load_config(src_cfg_path)
    task_filter = _csv(args.tasks)
    model_filter = _csv(args.model_ids)
    variant_filter = _csv(args.train_variants)
    rows = load_per_epoch_rows(
        results_root=Path(args.results_root),
        experiment_id=str(args.experiment_id),
        tasks=task_filter or None,
        model_ids=model_filter or None,
        train_variants=variant_filter or None,
        eval_split=str(args.eval_split),
    )
    if not rows:
        raise RuntimeError("No per-epoch run rows matched calibration filters")

    models_in_rows = sorted({str(r.get("model_id") or "") for r in rows if str(r.get("model_id") or "").strip()})
    selected_models = model_filter if model_filter else models_in_rows
    model_epochs: dict[str, int] = {}
    model_reports: dict[str, dict] = {}
    for mid in selected_models:
        mrows = [r for r in rows if str(r.get("model_id") or "") == str(mid)]
        if not mrows:
            if bool(args.allow_missing_models):
                continue
            raise RuntimeError(f"No calibration rows found for model_id={mid}")
        rep = summarize_calibration_for_model(
            rows=mrows,
            objective=str(args.objective),
            min_epoch=int(args.min_epoch),
            max_epoch=(None if args.max_epoch is None else int(args.max_epoch)),
            exclude_collapsed=bool(args.exclude_collapsed),
            collapse_epsilon=float(args.collapse_epsilon),
        )
        model_reports[str(mid)] = rep
        best = rep.get("best_epoch")
        if best is None:
            if bool(args.allow_missing_models):
                continue
            raise RuntimeError(f"Unable to determine best_epoch for model_id={mid}")
        model_epochs[str(mid)] = int(best)

    if not model_epochs:
        raise RuntimeError("No model epochs resolved from calibration results")

    calibrated = apply_epoch_overrides_to_config(
        cfg=cfg,
        experiment_id=str(args.experiment_id),
        model_epochs=model_epochs,
    )

    out_cfg_path = Path(args.output_config)
    _dump_config(out_cfg_path, dict(calibrated))
    manifest_path = Path(args.manifest_out) if args.manifest_out else out_cfg_path.with_suffix(out_cfg_path.suffix + ".calibration_manifest.json")
    manifest = write_calibrated_config_and_manifest(
        source_config_path=src_cfg_path,
        source_obj=cfg,
        calibrated_obj=calibrated,
        output_config_path=out_cfg_path,
        manifest_path=manifest_path,
        metadata={
            "experiment_id": str(args.experiment_id),
            "results_root": str(args.results_root),
            "eval_split": str(args.eval_split),
            "objective": str(args.objective),
            "min_epoch": int(args.min_epoch),
            "max_epoch": (None if args.max_epoch is None else int(args.max_epoch)),
            "exclude_collapsed": bool(args.exclude_collapsed),
            "collapse_epsilon": float(args.collapse_epsilon),
            "train_variants": list(variant_filter or []),
            "tasks": list(task_filter or []),
            "selected_models": list(selected_models or []),
            "model_epochs": dict(model_epochs),
            "model_reports": model_reports,
        },
    )
    print(
        json.dumps(
            {
                "ok": True,
                "output_config": str(out_cfg_path),
                "manifest": str(manifest_path),
                "model_epochs": model_epochs,
                "output_config_sha256": str(manifest.get("output_config_sha256") or ""),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0
