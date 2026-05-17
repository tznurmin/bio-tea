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

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from bio_tea.datasets.io import write_json
from bio_tea.datasets.results_layout import resolve_results_root
from bio_tea.datasets.tokenizer_loader import load_tokenizer

from .experiment_matrix import DEFAULT_TRAIN_VARIANTS, build_run_matrix, run_dir_for_job
from .exp1_runner import (
    _archive_existing_run_artifacts,
    _asset_manifest_from_preload,
    _collect_run_provenance,
    _default_backend,
    _enforce_model_max_length_guard,
    _hf_runtime_cfg,
    _job_asset_refs,
    _job_with_local_only,
    _load_json,
    _model_artifacts_manifest,
    _normalize_metrics_payload,
    _preload_assets_for_jobs,
    _resolve_asset_manifest_path,
    _run_is_resume_compatible,
    _seal_hf_network,
    _sha256_obj,
    _model_limits_runtime_cfg,
    _tokenization_runtime_cfg,
    _validate_asset_manifest_match,
    _validate_backend_metrics,
    _write_run_state,
)
from .prepared_cache import prepare_job_datasets

REQUIRED_METRICS = ("loss", "precision", "recall", "f1")


def _discover_tasks(source_root: Path) -> list[str]:
    tasks: list[str] = []
    if not source_root.exists():
        return tasks
    for p in sorted(source_root.iterdir()):
        if p.is_dir():
            tasks.append(p.name)
    return tasks


def _discover_sets(source_root: Path, task: str, *, experiment_id: str) -> list[str]:
    exp_id = str(experiment_id or "exp1")
    task_dir = source_root / task
    if not task_dir.exists():
        return []
    out: list[str] = []
    if exp_id == "exp1":
        for p in sorted(task_dir.iterdir()):
            if not p.is_dir():
                continue
            if not p.name.startswith("set"):
                continue
            if (p / "train").exists():
                out.append(p.name)
        return out

    exp_dir = task_dir / exp_id
    if not exp_dir.exists():
        return []
    for p in sorted(exp_dir.iterdir()):
        if not p.is_dir():
            continue
        if not p.name.startswith("set"):
            continue
        if (p / "train").exists():
            out.append(p.name)
    return out


def _discover_train_variants_for_set(
    *,
    source_root: Path,
    task: str,
    set_name: str,
    experiment_id: str,
) -> list[str]:
    exp_id = str(experiment_id or "exp1")
    variants: set[str] = set()

    if exp_id == "exp1":
        train_dir = source_root / task / set_name / "train"
    else:
        train_dir = source_root / task / exp_id / set_name / "train"
    if not train_dir.exists():
        return []

    for fp in sorted(train_dir.glob("*.set")):
        if fp.is_file():
            variants.add(fp.stem)
    for d in sorted(train_dir.iterdir()):
        if not d.is_dir():
            continue
        if (d / "train.set").exists():
            variants.add(d.name)
    return sorted(variants)


def _resolve_train_variants(
    *,
    cfg: Mapping[str, Any],
    source_root: Path,
    experiment_id: str,
    tasks: Sequence[str],
    set_names: Sequence[str],
) -> list[str]:
    exp_id = str(experiment_id or "exp1")
    exp_cfg = dict(((cfg.get("training") or {}).get(exp_id) or {})
    )
    explicit = [str(v) for v in list(exp_cfg.get("train_variants") or []) if str(v).strip()]
    if explicit:
        return explicit

    defaults = list(DEFAULT_TRAIN_VARIANTS.get(exp_id) or [])
    if defaults:
        return [str(v) for v in defaults]

    discovered: set[str] | None = None
    for task in tasks:
        for set_name in set_names:
            cur = set(
                _discover_train_variants_for_set(
                    source_root=source_root,
                    task=str(task),
                    set_name=str(set_name),
                    experiment_id=exp_id,
                )
            )
            if discovered is None:
                discovered = cur
            else:
                discovered &= cur
    out = sorted(list(discovered or set()))
    if not out:
        raise RuntimeError(
            f"No train variants discovered for experiment={exp_id}. "
            "Provide training.<experiment>.train_variants explicitly."
        )
    return out


def _experiment_runtime_cfg(cfg: Mapping[str, Any], *, experiment_id: str) -> dict[str, bool]:
    exp_cfg = dict(((cfg.get("training") or {}).get(str(experiment_id) or {}) or {})
    )
    return {
        "resume": bool(exp_cfg.get("resume", True)),
        "force": bool(exp_cfg.get("force", False)),
    }


def _as_str_set(v: Any) -> set[str]:
    if isinstance(v, (list, tuple)):
        return {str(x) for x in v}
    if v is None:
        return set()
    s = str(v).strip()
    return {s} if s else set()


def _as_int_set(v: Any) -> set[int]:
    if isinstance(v, (list, tuple)):
        out: set[int] = set()
        for x in v:
            try:
                out.add(int(x))
            except Exception:
                pass
        return out
    if v is None:
        return set()
    try:
        return {int(v)}
    except Exception:
        return set()


def _job_matches_selected_save_policy(*, job: Mapping[str, Any], cfg: Mapping[str, Any], selected: Mapping[str, Any], experiment_id: str) -> bool:
    model_ids = _as_str_set(selected.get("model_ids"))
    tasks = _as_str_set(selected.get("tasks"))
    train_variants = _as_str_set(selected.get("train_variants"))
    sets = _as_str_set(selected.get("sets"))
    seeds = _as_int_set(selected.get("seeds"))
    epochs = _as_int_set(selected.get("epochs"))

    if model_ids and str(job.get("model_id") or "") not in model_ids:
        return False
    if tasks and str(job.get("task") or "") not in tasks:
        return False
    if train_variants and str(job.get("train_variant") or "") not in train_variants:
        return False
    if sets and str(job.get("set") or "") not in sets:
        return False
    if seeds and int(job.get("seed", 0)) not in seeds:
        return False
    if epochs:
        hparams = dict(job.get("hparams") or {})
        if not hparams:
            hparams = dict((((cfg.get("training") or {}).get(experiment_id) or {}).get("hparams") or {})
            )
        try:
            cur_epochs = int(hparams.get("epochs", 1))
        except Exception:
            cur_epochs = 1
        if cur_epochs not in epochs:
            return False
    return True


def _save_model_cfg(cfg: Mapping[str, Any], *, experiment_id: str) -> dict[str, Any]:
    exp_cfg = dict(((cfg.get("training") or {}).get(str(experiment_id) or {}) or {})
    )
    sm = exp_cfg.get("save_model")
    if not isinstance(sm, Mapping):
        sm = {}
    mode = str(sm.get("mode", "none")).strip().lower() or "none"
    if mode not in {"none", "all", "selected"}:
        raise ValueError(f"training.{experiment_id}.save_model.mode must be one of: none, all, selected")
    selected = dict(sm.get("selected") or {})
    return {
        "mode": mode,
        "selected": selected,
    }


def _should_save_model_for_job(*, job: Mapping[str, Any], cfg: Mapping[str, Any], save_cfg: Mapping[str, Any], experiment_id: str) -> bool:
    mode = str(save_cfg.get("mode", "none"))
    if mode == "none":
        return False
    if mode == "all":
        return True
    selected = dict(save_cfg.get("selected") or {})
    return _job_matches_selected_save_policy(
        job=job,
        cfg=cfg,
        selected=selected,
        experiment_id=experiment_id,
    )


def _has_eval_artifacts(
    *,
    source_root: Path,
    task: str,
    set_name: str,
    eval_name: str,
    experiment_id: str,
    eval_split: str,
) -> bool:
    exp_id = str(experiment_id or "exp1")
    split_name = str(eval_split or "test")
    candidates = [
        (
            source_root / task / exp_id / set_name / split_name / f"{eval_name}.set",
            source_root / task / exp_id / set_name / split_name / f"{eval_name}.meta.jsonl",
        ),
        (
            source_root / task / exp_id / split_name / f"{eval_name}.set",
            source_root / task / exp_id / split_name / f"{eval_name}.meta.jsonl",
        ),
    ]
    if any(sp.exists() and mp.exists() for sp, mp in candidates):
        return True
    if str(eval_name) == "unaugmented":
        fb_set_set = source_root / task / set_name / "splits" / split_name / "unaugmented.set"
        fb_meta_set = source_root / task / set_name / "splits" / split_name / "unaugmented.meta.jsonl"
        if fb_set_set.exists() and fb_meta_set.exists():
            return True
        fb_set = source_root / task / split_name / "unaugmented.set"
        fb_meta = source_root / task / split_name / "unaugmented.meta.jsonl"
        return fb_set.exists() and fb_meta.exists()
    return False


def _has_train_artifacts(*, source_root: Path, task: str, set_name: str, variant: str, experiment_id: str) -> bool:
    exp_id = str(experiment_id or "exp1")
    candidates: list[tuple[Path, Path]] = []
    if exp_id == "exp1":
        candidates.append(
            (
                source_root / task / set_name / "train" / f"{variant}.set",
                source_root / task / set_name / "train" / f"{variant}.meta.jsonl",
            )
        )
    candidates.append(
        (
            source_root / task / exp_id / set_name / "train" / f"{variant}.set",
            source_root / task / exp_id / set_name / "train" / f"{variant}.meta.jsonl",
        )
    )
    candidates.append(
        (
            source_root / task / exp_id / set_name / "train" / variant / "train.set",
            source_root / task / exp_id / set_name / "train" / variant / "train.meta.jsonl",
        )
    )
    return any(sp.exists() and mp.exists() for sp, mp in candidates)


def _source_root_has_job_artifacts(source_root: Path, job: Mapping[str, Any], *, experiment_id: str) -> bool:
    task = str(job.get("task") or "")
    set_name = str(job.get("set") or "")
    variant = str(job.get("train_variant") or "")
    eval_split = str(job.get("eval_split") or "test")
    eval_sets = [str(x) for x in (job.get("eval_sets") or [])]

    if not _has_train_artifacts(
        source_root=source_root,
        task=task,
        set_name=set_name,
        variant=variant,
        experiment_id=experiment_id,
    ):
        return False
    for eval_name in eval_sets:
        if not _has_eval_artifacts(
            source_root=source_root,
            task=task,
            set_name=set_name,
            eval_name=eval_name,
            experiment_id=experiment_id,
            eval_split=eval_split,
        ):
            return False
    return True


def _resolve_source_root_for_job(
    *,
    preferred_source_root: Path,
    base_results_root: Path,
    profiles_dir: str,
    job: Mapping[str, Any],
    experiment_id: str,
) -> Path:
    tried: list[Path] = []

    def _already_tried(p: Path) -> bool:
        ps = str(p)
        return any(str(t) == ps for t in tried)

    def _try(p: Path) -> Path | None:
        cp = Path(p)
        if _already_tried(cp):
            return None
        tried.append(cp)
        if _source_root_has_job_artifacts(cp, job, experiment_id=experiment_id):
            return cp
        return None

    direct = _try(Path(preferred_source_root))
    if direct is not None:
        return direct

    prof = _try(base_results_root / profiles_dir / str(job.get("profile_id") or ""))
    if prof is not None:
        return prof

    matches: list[Path] = []
    profiles_root = base_results_root / profiles_dir
    if profiles_root.exists():
        for p in sorted(profiles_root.iterdir()):
            if not p.is_dir() or _already_tried(p):
                continue
            if _source_root_has_job_artifacts(p, job, experiment_id=experiment_id):
                matches.append(p)

    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError(
            f"Ambiguous source roots for {experiment_id} job; matched multiple profile roots: "
            + ", ".join(str(p) for p in matches)
            + ". Use --source-root to select one explicitly."
        )

    task = str(job.get("task") or "")
    set_name = str(job.get("set") or "")
    variant = str(job.get("train_variant") or "")
    eval_split = str(job.get("eval_split") or "test")
    raise FileNotFoundError(
        f"Missing train/eval artifacts for experiment={experiment_id} task={task} set={set_name} "
        f"variant={variant} eval_split={eval_split}. "
        f"Tried source roots: {', '.join(str(p) for p in tried)}"
    )


def _build_run_spec(
    *,
    experiment_id: str,
    job: Mapping[str, Any],
    prepared_manifest: Mapping[str, Any],
    cfg: Mapping[str, Any],
    provenance: Mapping[str, Any] | None = None,
) -> dict:
    exp_id = str(experiment_id or "exp1")
    exp_cfg = dict(((cfg.get("training") or {}).get(exp_id) or {})
    )
    tokenization_runtime = _tokenization_runtime_cfg(cfg, experiment_id=exp_id)
    model_limits_runtime = _model_limits_runtime_cfg(cfg, experiment_id=exp_id)
    hf_cfg = dict(((cfg.get("training") or {}).get("hf") or {}))
    payload = {
        "schema_version": 1,
        "experiment_id": exp_id,
        "provenance": dict(provenance or {}),
        "job": {
            "experiment_id": exp_id,
            "task": str(job.get("task") or ""),
            "set": str(job.get("set") or ""),
            "seed": int(job.get("seed", 0)),
            "model_id": str(job.get("model_id") or ""),
            "model_name_or_path": str(job.get("model_name_or_path") or ""),
            "model_revision": str(job.get("model_revision") or ""),
            "train_variant": str(job.get("train_variant") or ""),
            "eval_split": str(job.get("eval_split") or "test"),
            "eval_sets": [str(x) for x in (job.get("eval_sets") or [])],
            "max_length": int(job.get("max_length", 0)),
            "profile_id": str(job.get("profile_id") or ""),
            "tokenizer": dict(job.get("tokenizer") or {}),
        },
        "prepared": {
            "fingerprint": str(prepared_manifest.get("fingerprint") or ""),
            "n_train_examples": int(prepared_manifest.get("n_train_examples", 0)),
            "n_eval_examples": dict(prepared_manifest.get("n_eval_examples") or {}),
        },
        "training": {
            "hparams": dict(job.get("hparams") or exp_cfg.get("hparams") or {}),
            "save_model": dict((exp_cfg.get("save_model") or {})),
            "tokenization": dict(tokenization_runtime),
            "model_limits": dict(model_limits_runtime),
            "hf": {
                "local_files_only": bool(hf_cfg.get("local_files_only", False)),
                "revision": str(hf_cfg.get("revision") or ""),
                "dataloader_pin_memory": bool(hf_cfg.get("dataloader_pin_memory", False)),
            },
        },
    }
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "signature": _sha256_obj(payload),
        "payload": payload,
    }


def run_experiment_matrix(
    *,
    cfg: Mapping[str, Any],
    experiment_id: str,
    tasks: list[str] | None = None,
    set_names: list[str] | None = None,
    source_root: Path | None = None,
    backend: Callable[..., Mapping[str, Mapping[str, Any]]] | None = None,
) -> dict:
    """Prepare and execute training/eval jobs for one generated experiment."""

    exp_id = str(experiment_id or "").strip().lower()
    if not exp_id:
        raise ValueError("experiment_id is required")

    if source_root is None:
        resolved, _info = resolve_results_root(cfg, cfg.get("tokenizer") or {})
        source_root = resolved
    source_root = Path(source_root)

    if tasks is None:
        tasks = _discover_tasks(source_root)
    if not tasks:
        raise RuntimeError(f"No tasks found under source_root={source_root}")

    if set_names is None:
        set_names = _discover_sets(source_root, tasks[0], experiment_id=exp_id)
    if not set_names:
        raise RuntimeError(
            f"No set directories found under source_root={source_root} for task={tasks[0]} experiment={exp_id}"
        )

    train_variants = _resolve_train_variants(
        cfg=cfg,
        source_root=source_root,
        experiment_id=exp_id,
        tasks=tasks,
        set_names=set_names,
    )
    jobs = build_run_matrix(
        cfg,
        experiment_id=exp_id,
        tasks=tasks,
        set_names=set_names,
        train_variants=train_variants,
    )
    if not jobs:
        return {
            "experiment_id": exp_id,
            "source_root": str(source_root),
            "n_jobs": 0,
            "n_ok": 0,
            "n_failed": 0,
            "n_skipped": 0,
            "jobs": [],
        }

    base_results_root = Path(str(cfg.get("results_root", "results")))
    profiles_dir = str((cfg.get("training") or {}).get("profiles_dir") or "profiles")
    hf_runtime = _hf_runtime_cfg(cfg)
    tokenization_runtime = _tokenization_runtime_cfg(cfg, experiment_id=exp_id)
    model_limits_runtime = _model_limits_runtime_cfg(cfg, experiment_id=exp_id)
    runtime_cfg = _experiment_runtime_cfg(cfg, experiment_id=exp_id)
    save_cfg = _save_model_cfg(cfg, experiment_id=exp_id)
    asset_manifest: dict | None = None
    asset_manifest_path: Path | None = None
    preload_stats: dict[str, Any] | None = None
    if hf_runtime["preload_assets"]:
        preload_stats = _preload_assets_for_jobs(jobs)
        asset_manifest = _asset_manifest_from_preload(preload_stats, jobs=jobs)
        asset_manifest_path = _resolve_asset_manifest_path(
            base_results_root=base_results_root,
            profiles_dir=profiles_dir,
            hf_runtime=hf_runtime,
        )
        if hf_runtime["require_asset_manifest_match"] and asset_manifest_path.exists():
            expected_obj = json.loads(asset_manifest_path.read_text(encoding="utf-8"))
            mismatches = _validate_asset_manifest_match(expected=expected_obj, observed=asset_manifest)
            if mismatches:
                raise RuntimeError(
                    "HF asset checksum manifest mismatch: "
                    + "; ".join(mismatches)
                    + f". manifest={asset_manifest_path}"
                )
        if hf_runtime["write_asset_manifest"] and asset_manifest_path is not None:
            write_json(asset_manifest_path, asset_manifest)
    if hf_runtime["seal_network_after_preload"]:
        _seal_hf_network()

    backend_fn = backend or _default_backend
    run_provenance = _collect_run_provenance(cfg)

    ok = 0
    failed = 0
    skipped = 0
    job_summaries: list[dict] = []
    model_limit_cache: dict[tuple[str, str, bool], int | None] = {}
    for job0 in jobs:
        job = _job_with_local_only(job0) if hf_runtime["seal_network_after_preload"] else dict(job0)
        output_profile_root = base_results_root / profiles_dir / str(job["profile_id"])
        run_dir = run_dir_for_job(output_profile_root, job)
        prepared_dir: Path | None = None
        job_source_root: Path | None = None
        signature = ""
        try:
            tokenizer, tok_info = load_tokenizer(job["tokenizer"])
            _enforce_model_max_length_guard(
                job=job,
                hf_runtime=hf_runtime,
                model_limits_runtime=model_limits_runtime,
                cache=model_limit_cache,
            )
            job_source_root = _resolve_source_root_for_job(
                preferred_source_root=source_root,
                base_results_root=base_results_root,
                profiles_dir=profiles_dir,
                job=job,
                experiment_id=exp_id,
            )
            prep = prepare_job_datasets(
                source_root=job_source_root,
                output_profile_root=output_profile_root,
                job=job,
                tokenizer=tokenizer,
                tokenizer_info=tok_info,
                experiment_id=exp_id,
                fail_on_truncation=bool(tokenization_runtime.get("fail_on_truncation", True)),
            )

            prepared_dir = Path(prep["prepared_dir"])
            prepared_manifest = json.loads((prepared_dir / "manifest.json").read_text(encoding="utf-8"))
            run_dir.mkdir(parents=True, exist_ok=True)
            run_spec = _build_run_spec(
                experiment_id=exp_id,
                job=job,
                prepared_manifest=prepared_manifest,
                cfg=cfg,
                provenance=run_provenance,
            )
            signature = str(run_spec["signature"])
            if runtime_cfg["resume"] and (not runtime_cfg["force"]):
                if _run_is_resume_compatible(run_dir=run_dir, run_spec=run_spec):
                    skipped += 1
                    task_experiment_root = output_profile_root / str(job["task"]) / exp_id
                    asset_refs = _job_asset_refs(
                        job=job,
                        asset_manifest=asset_manifest,
                        asset_manifest_path=asset_manifest_path,
                    )
                    job_summaries.append(
                        {
                            "status": "skipped",
                            "job": {
                                "task": str(job["task"]),
                                "set": str(job["set"]),
                                "seed": int(job["seed"]),
                                "model_id": str(job["model_id"]),
                                "train_variant": str(job["train_variant"]),
                                "eval_split": str(job.get("eval_split") or "test"),
                            },
                            "run_dir": str(run_dir),
                            "prepared_dir": str(prepared_dir),
                            "profile_root": str(output_profile_root),
                            "task_experiment_root": str(task_experiment_root),
                            "source_root": str(job_source_root),
                            "hf_assets": asset_refs,
                        }
                    )
                    continue

            prev_spec = _load_json(run_dir / "run_spec.json")
            prev_sig = str((prev_spec or {}).get("signature") or "")
            archive_reason = "rerun"
            if runtime_cfg["force"]:
                archive_reason = "force_rerun"
            elif prev_sig and prev_sig != signature:
                archive_reason = "signature_changed"
            elif runtime_cfg["resume"] and (not runtime_cfg["force"]):
                archive_reason = "resume_disabled_or_incomplete"
            archive_dir = _archive_existing_run_artifacts(
                run_dir=run_dir,
                reason=archive_reason,
                previous_signature=prev_sig or None,
            )

            write_json(run_dir / "run_spec.json", run_spec)
            _write_run_state(run_dir=run_dir, status="running", signature=signature)
            save_model = _should_save_model_for_job(job=job, cfg=cfg, save_cfg=save_cfg, experiment_id=exp_id)
            model_dir = run_dir / "model"
            backend_job = dict(job)
            backend_job["save_model"] = bool(save_model)
            backend_job["save_model_dir"] = str(model_dir)
            metrics = backend_fn(
                prepared_dir=prepared_dir,
                prepared_manifest=prepared_manifest,
                job=backend_job,
                cfg=dict(cfg),
            )
            _validate_backend_metrics(metrics, job["eval_sets"])

            task_experiment_root = output_profile_root / str(job["task"]) / exp_id
            run_obj = {
                "experiment_id": exp_id,
                "task": str(job["task"]),
                "set": str(job["set"]),
                "seed": int(job["seed"]),
                "model_id": str(job["model_id"]),
                "profile_id": str(job["profile_id"]),
                "train_variant": str(job["train_variant"]),
                "eval_split": str(job.get("eval_split") or "test"),
                "eval_sets": list(job["eval_sets"]),
                "source_root": str(job_source_root),
                "prepared": prep,
                "metrics": _normalize_metrics_payload(metrics),
            }
            asset_refs = _job_asset_refs(
                job=job,
                asset_manifest=asset_manifest,
                asset_manifest_path=asset_manifest_path,
            )
            if asset_refs is not None:
                run_obj["hf_assets"] = asset_refs
            artifact_manifest = _model_artifacts_manifest(save_model=save_model, model_dir=model_dir, policy_mode=save_cfg["mode"])
            write_json(run_dir / "model_artifacts.json", artifact_manifest)
            run_obj["model_artifacts"] = {
                "saved": bool(artifact_manifest.get("saved", False)),
                "mode": str(artifact_manifest.get("mode") or ""),
                "n_files": int(artifact_manifest.get("n_files", 0)),
                "total_bytes": int(artifact_manifest.get("total_bytes", 0)),
                "manifest_path": str(run_dir / "model_artifacts.json"),
            }
            write_json(run_dir / "metrics.json", run_obj)
            _write_run_state(
                run_dir=run_dir,
                status="completed",
                signature=signature,
                extra={"metrics_sha256": _sha256_obj(run_obj)},
            )
            ok += 1
            job_summaries.append(
                {
                    "status": "completed",
                    "job": {
                        "task": str(job["task"]),
                        "set": str(job["set"]),
                        "seed": int(job["seed"]),
                        "model_id": str(job["model_id"]),
                        "train_variant": str(job["train_variant"]),
                        "eval_split": str(job.get("eval_split") or "test"),
                    },
                    "run_dir": str(run_dir),
                    "prepared_dir": str(prepared_dir),
                    "profile_root": str(output_profile_root),
                    "task_experiment_root": str(task_experiment_root),
                    "source_root": str(job_source_root),
                    "hf_assets": asset_refs,
                    "model_artifacts": run_obj["model_artifacts"],
                    "archived_previous_run_dir": str(archive_dir) if archive_dir is not None else None,
                }
            )
        except Exception as e:
            failed += 1
            try:
                run_dir.mkdir(parents=True, exist_ok=True)
                _write_run_state(
                    run_dir=run_dir,
                    status="failed",
                    signature=signature,
                    extra={
                        "error_type": type(e).__name__,
                        "error": str(e),
                    },
                )
            except Exception:
                pass
            raise

    summary = {
        "experiment_id": exp_id,
        "source_root": str(source_root),
        "n_jobs": len(jobs),
        "n_ok": ok,
        "n_failed": failed,
        "n_skipped": skipped,
        "runtime": runtime_cfg,
        "tokenization_runtime": tokenization_runtime,
        "model_limits_runtime": model_limits_runtime,
        "save_model": save_cfg,
        "hf_runtime": hf_runtime,
        "hf_preload": preload_stats,
        "hf_asset_manifest_path": str(asset_manifest_path) if asset_manifest_path is not None else None,
        "provenance": run_provenance,
        "jobs": job_summaries,
    }
    return summary
