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
import os
import hashlib
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from bio_tea.datasets.io import sha256_file, write_json
from bio_tea.datasets.results_layout import resolve_results_root
from bio_tea.datasets.tokenizer_loader import load_tokenizer

from .exp1_matrix import build_exp1_run_matrix, run_dir_for_job
from .hf_backend import run_hf_backend
from .prepared_cache import prepare_job_datasets


REQUIRED_METRICS = ("loss", "precision", "recall", "f1")
RUN_STATUS_TAGS: dict[str, str] = {
    "running": "RUN_RUNNING",
    "completed": "RUN_COMPLETED",
    "failed": "RUN_FAILED",
}


def _status_tag_path(run_dir: Path, status: str) -> Path | None:
    tag_name = RUN_STATUS_TAGS.get(str(status))
    if not tag_name:
        return None
    return run_dir / tag_name


def _sync_status_tag(*, run_dir: Path, status: str) -> None:
    for tag_name in RUN_STATUS_TAGS.values():
        tp = run_dir / tag_name
        if tp.exists():
            try:
                tp.unlink()
            except Exception:
                pass
    tp = _status_tag_path(run_dir, str(status))
    if tp is None:
        return
    tp.write_text(datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8")


def _archive_reason_slug(reason: str) -> str:
    s = str(reason or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s or "rerun"


def _archive_existing_run_artifacts(
    *,
    run_dir: Path,
    reason: str,
    previous_signature: str | None = None,
) -> Path | None:
    if not run_dir.exists():
        return None
    entries = [p for p in sorted(run_dir.iterdir(), key=lambda p: p.name) if p.name != "attempts"]
    if not entries:
        return None

    attempts_root = run_dir / "attempts"
    attempts_root.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    reason_slug = _archive_reason_slug(reason)
    archive_dir = attempts_root / f"{ts}--{reason_slug}"
    idx = 1
    while archive_dir.exists():
        archive_dir = attempts_root / f"{ts}--{reason_slug}-{idx:02d}"
        idx += 1
    archive_dir.mkdir(parents=True, exist_ok=False)

    moved: list[str] = []
    for p in entries:
        dst = archive_dir / p.name
        shutil.move(str(p), str(dst))
        moved.append(p.name)
    write_json(
        archive_dir / "archive_manifest.json",
        {
            "schema_version": 1,
            "archived_at": datetime.now(timezone.utc).isoformat(),
            "reason": str(reason),
            "reason_slug": reason_slug,
            "previous_signature": str(previous_signature or ""),
            "source_run_dir": str(run_dir),
            "moved_entries": moved,
        },
    )
    (archive_dir / "RUN_ARCHIVED").write_text(datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8")
    return archive_dir


def _sha256_obj(obj: object) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _discover_tasks(source_root: Path) -> list[str]:
    tasks: list[str] = []
    if not source_root.exists():
        return tasks
    for p in sorted(source_root.iterdir()):
        if p.is_dir():
            tasks.append(p.name)
    return tasks


def _discover_sets(source_root: Path, task: str) -> list[str]:
    task_dir = source_root / task
    if not task_dir.exists():
        return []
    out: list[str] = []
    for p in sorted(task_dir.iterdir()):
        if not p.is_dir():
            continue
        if not p.name.startswith("set"):
            continue
        if (p / "train").exists():
            out.append(p.name)
    return out


def _validate_backend_metrics(metrics: Mapping[str, Mapping[str, Any]], eval_sets: Sequence[str]) -> None:
    for eval_name in eval_sets:
        if eval_name not in metrics:
            raise ValueError(f"backend metrics missing eval set: {eval_name}")
        row = metrics[eval_name]
        for k in REQUIRED_METRICS:
            if k not in row:
                raise ValueError(f"backend metrics missing required metric '{k}' for eval set '{eval_name}'")
            _ = float(row[k])


def _default_backend(**_kwargs):
    return run_hf_backend(**_kwargs)


def _exp1_runtime_cfg(cfg: Mapping[str, Any]) -> dict[str, bool]:
    exp1_cfg = dict(((cfg.get("training") or {}).get("exp1") or {}))
    return {
        "resume": bool(exp1_cfg.get("resume", True)),
        "force": bool(exp1_cfg.get("force", False)),
    }


def _save_model_cfg(cfg: Mapping[str, Any]) -> dict[str, Any]:
    exp1_cfg = dict(((cfg.get("training") or {}).get("exp1") or {}))
    sm = exp1_cfg.get("save_model")
    if not isinstance(sm, Mapping):
        sm = {}
    mode = str(sm.get("mode", "none")).strip().lower() or "none"
    if mode not in {"none", "all", "selected"}:
        raise ValueError("training.exp1.save_model.mode must be one of: none, all, selected")
    selected = dict(sm.get("selected") or {})
    return {
        "mode": mode,
        "selected": selected,
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


def _job_matches_selected_save_policy(*, job: Mapping[str, Any], cfg: Mapping[str, Any], selected: Mapping[str, Any]) -> bool:
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
            hparams = dict((((cfg.get("training") or {}).get("exp1") or {}).get("hparams") or {}))
        try:
            cur_epochs = int(hparams.get("epochs", 1))
        except Exception:
            cur_epochs = 1
        if cur_epochs not in epochs:
            return False
    return True


def _should_save_model_for_job(*, job: Mapping[str, Any], cfg: Mapping[str, Any], save_cfg: Mapping[str, Any]) -> bool:
    mode = str(save_cfg.get("mode", "none"))
    if mode == "none":
        return False
    if mode == "all":
        return True
    selected = dict(save_cfg.get("selected") or {})
    return _job_matches_selected_save_policy(job=job, cfg=cfg, selected=selected)


def _hf_runtime_cfg(cfg: Mapping[str, Any]) -> dict[str, Any]:
    hf_cfg = dict(((cfg.get("training") or {}).get("hf") or {}))
    return {
        "preload_assets": bool(hf_cfg.get("preload_assets", False)),
        "seal_network_after_preload": bool(hf_cfg.get("seal_network_after_preload", False)),
        "local_files_only": bool(hf_cfg.get("local_files_only", False)),
        "asset_manifest_path": str(hf_cfg.get("asset_manifest_path") or "").strip(),
        "require_asset_manifest_match": bool(hf_cfg.get("require_asset_manifest_match", False)),
        "write_asset_manifest": bool(hf_cfg.get("write_asset_manifest", True)),
    }


def _tokenization_runtime_cfg(cfg: Mapping[str, Any], *, experiment_id: str = "exp1") -> dict[str, Any]:
    tr = dict(cfg.get("training") or {})
    tok_cfg = dict(tr.get("tokenization") or {})
    exp_cfg = dict(tr.get(str(experiment_id) or "") or {})
    if "fail_on_truncation" in exp_cfg:
        fail_on_truncation = bool(exp_cfg.get("fail_on_truncation"))
    else:
        fail_on_truncation = bool(tok_cfg.get("fail_on_truncation", True))
    return {
        "fail_on_truncation": bool(fail_on_truncation),
    }


def _model_limits_runtime_cfg(cfg: Mapping[str, Any], *, experiment_id: str = "exp1") -> dict[str, Any]:
    tr = dict(cfg.get("training") or {})
    ml_cfg = dict(tr.get("model_limits") or {})
    exp_cfg = dict(tr.get(str(experiment_id) or "") or {})
    if "check_model_max_length" in exp_cfg:
        enforce = bool(exp_cfg.get("check_model_max_length"))
    else:
        enforce = bool(ml_cfg.get("check_model_max_length", False))
    if "fail_on_unresolved_model_limit" in exp_cfg:
        fail_on_unresolved = bool(exp_cfg.get("fail_on_unresolved_model_limit"))
    else:
        fail_on_unresolved = bool(ml_cfg.get("fail_on_unresolved_model_limit", False))
    return {
        "check_model_max_length": bool(enforce),
        "fail_on_unresolved_model_limit": bool(fail_on_unresolved),
    }


def _resolve_model_max_length_from_config(*, job: Mapping[str, Any], hf_runtime: Mapping[str, Any]) -> int | None:
    model_name = str(job.get("model_name_or_path") or "").strip()
    if not model_name:
        return None

    model_local_files_only = bool(job.get("model_local_files_only", hf_runtime.get("local_files_only", False)))
    model_revision_raw = job.get("model_revision", hf_runtime.get("revision"))
    model_revision = str(model_revision_raw).strip() if model_revision_raw is not None else ""
    model_revision = model_revision or None

    try:
        from transformers import AutoConfig
    except Exception as e:  # pragma: no cover
        raise RuntimeError("transformers is required for model max-length guard checks") from e

    kwargs: dict[str, Any] = {}
    if model_local_files_only:
        kwargs["local_files_only"] = True
    if model_revision is not None:
        kwargs["revision"] = model_revision
    cfg = AutoConfig.from_pretrained(model_name, **kwargs)

    keys = [
        "max_position_embeddings",
        "n_positions",
        "max_seq_len",
        "max_sequence_length",
        "seq_length",
    ]
    for k in keys:
        v = getattr(cfg, k, None)
        if isinstance(v, int) and int(v) > 0:
            return int(v)

    try:
        d = dict(cfg.to_dict() or {})
    except Exception:
        d = {}
    for k in keys:
        v = d.get(k)
        if isinstance(v, int) and int(v) > 0:
            return int(v)
    return None


def _enforce_model_max_length_guard(
    *,
    job: Mapping[str, Any],
    hf_runtime: Mapping[str, Any],
    model_limits_runtime: Mapping[str, Any],
    cache: dict[tuple[str, str, bool], int | None],
) -> None:
    check_enabled = bool(model_limits_runtime.get("check_model_max_length", False))
    if job.get("check_model_max_length") is not None:
        check_enabled = bool(job.get("check_model_max_length"))
    if not check_enabled:
        return

    model_name = str(job.get("model_name_or_path") or "").strip()
    model_local_files_only = bool(job.get("model_local_files_only", hf_runtime.get("local_files_only", False)))
    model_revision_raw = job.get("model_revision", hf_runtime.get("revision"))
    model_revision = str(model_revision_raw).strip() if model_revision_raw is not None else ""
    key = (model_name, model_revision, model_local_files_only)

    if key not in cache:
        cache[key] = _resolve_model_max_length_from_config(job=job, hf_runtime=hf_runtime)
    model_limit = cache[key]

    if model_limit is None:
        if bool(model_limits_runtime.get("fail_on_unresolved_model_limit", False)):
            raise RuntimeError(
                "Unable to resolve model max sequence length from config for "
                f"{model_name or '<missing-model>'}. Disable with "
                "`training.model_limits.check_model_max_length: false` or "
                "`training.<exp>.check_model_max_length: false`."
            )
        return

    configured = int(job.get("max_length", 0))
    if configured > int(model_limit):
        raise ValueError(
            "Configured max_length exceeds model max sequence length: "
            f"model={model_name} configured={configured} model_limit={int(model_limit)}. "
            "Fix max_length or disable check via "
            "`training.model_limits.check_model_max_length: false`."
        )


def _canonical_revision(v: Any) -> str:
    s = str(v).strip() if v is not None else ""
    return s or "main"


def _asset_key(repo_id: str, revision: str) -> str:
    return f"{repo_id}@{revision}"


def _collect_required_assets(jobs: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    assets: dict[str, dict[str, Any]] = {}
    for j in jobs:
        tok_cfg = dict(j.get("tokenizer") or {})
        tok_name = str(tok_cfg.get("name_or_path") or "").strip()
        tok_rev = _canonical_revision(tok_cfg.get("revision"))
        if tok_name:
            k = _asset_key(tok_name, tok_rev)
            cur = assets.setdefault(
                k,
                {
                    "repo_id": tok_name,
                    "revision": tok_rev,
                    "roles": set(),
                },
            )
            cur["roles"].add("tokenizer")

        model_name = str(j.get("model_name_or_path") or "").strip()
        model_rev = _canonical_revision(j.get("model_revision"))
        if model_name:
            k = _asset_key(model_name, model_rev)
            cur = assets.setdefault(
                k,
                {
                    "repo_id": model_name,
                    "revision": model_rev,
                    "roles": set(),
                },
            )
            cur["roles"].add("model")
    return assets


def _checksum_snapshot_tree(snapshot_dir: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for fp in sorted(snapshot_dir.rglob("*")):
        if not fp.is_file():
            continue
        rel = str(fp.relative_to(snapshot_dir))
        sz = int(fp.stat().st_size)
        files.append(
            {
                "path": rel,
                "size": sz,
                "sha256": sha256_file(fp),
            }
        )

    import hashlib

    h = hashlib.sha256()
    for row in files:
        h.update(str(row["path"]).encode("utf-8"))
        h.update(b"\t")
        h.update(str(row["size"]).encode("utf-8"))
        h.update(b"\t")
        h.update(str(row["sha256"]).encode("utf-8"))
        h.update(b"\n")
    return {
        "file_count": len(files),
        "total_bytes": int(sum(int(r["size"]) for r in files)),
        "root_sha256": h.hexdigest(),
        "files": files,
    }


def _preload_assets_for_jobs(jobs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    try:
        from transformers import AutoModel
        from huggingface_hub import snapshot_download
    except Exception as e:  # pragma: no cover
        raise RuntimeError("transformers + huggingface_hub are required for preload_assets mode") from e

    seen_toks: set[tuple[str, str | None]] = set()
    seen_models: set[tuple[str, str | None]] = set()

    for j in jobs:
        tok_cfg = dict(j.get("tokenizer") or {})
        tok_name = str(tok_cfg.get("name_or_path") or "").strip()
        tok_rev_raw = tok_cfg.get("revision")
        tok_rev = str(tok_rev_raw).strip() if tok_rev_raw is not None else ""
        tok_rev = tok_rev or None
        tok_key = (tok_name, tok_rev)
        if tok_name and tok_key not in seen_toks:
            load_tokenizer(tok_cfg)
            seen_toks.add(tok_key)

        model_name = str(j.get("model_name_or_path") or "").strip()
        model_rev_raw = j.get("model_revision")
        model_rev = str(model_rev_raw).strip() if model_rev_raw is not None else ""
        model_rev = model_rev or None
        model_key = (model_name, model_rev)
        if model_name and model_key not in seen_models:
            mk: dict[str, Any] = {}
            if model_rev is not None:
                mk["revision"] = model_rev
            if bool(j.get("model_local_files_only", False)):
                mk["local_files_only"] = True
            AutoModel.from_pretrained(model_name, **mk)
            seen_models.add(model_key)

    assets = _collect_required_assets(jobs)
    assets_out: dict[str, dict[str, Any]] = {}
    for k in sorted(assets.keys()):
        spec = assets[k]
        repo_id = str(spec["repo_id"])
        revision = str(spec["revision"])
        local_dir = Path(
            snapshot_download(
                repo_id=repo_id,
                revision=revision,
                local_files_only=True,
            )
        )
        tree = _checksum_snapshot_tree(local_dir)
        assets_out[k] = {
            "repo_id": repo_id,
            "revision": revision,
            "roles": sorted(list(spec["roles"])),
            "snapshot_dir": str(local_dir),
            **tree,
        }

    return {
        "tokenizers": int(len(seen_toks)),
        "models": int(len(seen_models)),
        "assets": assets_out,
    }


def _seal_hf_network() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"


def _job_with_local_only(job: Mapping[str, Any]) -> dict[str, Any]:
    j = dict(job)
    tok = dict(j.get("tokenizer") or {})
    tok["local_files_only"] = True
    j["tokenizer"] = tok
    j["model_local_files_only"] = True
    return j


def _resolve_asset_manifest_path(*, base_results_root: Path, profiles_dir: str, hf_runtime: Mapping[str, Any]) -> Path:
    explicit = str(hf_runtime.get("asset_manifest_path") or "").strip()
    if explicit:
        p = Path(explicit)
        return p if p.is_absolute() else (base_results_root / p)
    return base_results_root / profiles_dir / "hf_asset_manifest.json"


def _asset_manifest_from_preload(preload: Mapping[str, Any], *, jobs: Sequence[Mapping[str, Any]]) -> dict:
    assets = dict(preload.get("assets") or {})
    required = _collect_required_assets(jobs)
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_assets": int(len(assets)),
        "assets": assets,
        "required_assets": {
            k: {
                "repo_id": str(v.get("repo_id") or ""),
                "revision": str(v.get("revision") or ""),
                "roles": sorted(list(v.get("roles") or [])),
            }
            for k, v in sorted(required.items(), key=lambda kv: kv[0])
        },
    }


def _validate_asset_manifest_match(*, expected: Mapping[str, Any], observed: Mapping[str, Any]) -> list[str]:
    exp_assets = dict(expected.get("assets") or {})
    obs_assets = dict(observed.get("assets") or {})
    mismatches: list[str] = []
    for key, obs in sorted(obs_assets.items(), key=lambda kv: kv[0]):
        exp = exp_assets.get(key)
        if not isinstance(exp, Mapping):
            mismatches.append(f"missing_asset:{key}")
            continue
        exp_hash = str(exp.get("root_sha256") or "")
        obs_hash = str(obs.get("root_sha256") or "")
        if exp_hash != obs_hash:
            mismatches.append(f"hash_mismatch:{key}:expected={exp_hash}:observed={obs_hash}")
    return mismatches


def _job_asset_refs(
    *,
    job: Mapping[str, Any],
    asset_manifest: Mapping[str, Any] | None,
    asset_manifest_path: Path | None,
) -> dict[str, Any] | None:
    if not isinstance(asset_manifest, Mapping):
        return None
    assets = dict(asset_manifest.get("assets") or {})
    tok_cfg = dict(job.get("tokenizer") or {})
    tok_name = str(tok_cfg.get("name_or_path") or "").strip()
    tok_rev = _canonical_revision(tok_cfg.get("revision"))
    model_name = str(job.get("model_name_or_path") or "").strip()
    model_rev = _canonical_revision(job.get("model_revision"))
    tok_key = _asset_key(tok_name, tok_rev) if tok_name else ""
    model_key = _asset_key(model_name, model_rev) if model_name else ""
    tok_hash = str((assets.get(tok_key) or {}).get("root_sha256") or "")
    model_hash = str((assets.get(model_key) or {}).get("root_sha256") or "")
    out = {
        "asset_manifest_path": str(asset_manifest_path) if asset_manifest_path is not None else None,
        "tokenizer_asset_key": tok_key or None,
        "model_asset_key": model_key or None,
        "tokenizer_root_sha256": tok_hash or None,
        "model_root_sha256": model_hash or None,
    }
    return out


def _has_eval_artifacts(
    *,
    source_root: Path,
    task: str,
    set_name: str,
    eval_name: str,
    eval_split: str,
) -> bool:
    split_name = str(eval_split or "test")
    set_scoped_set = source_root / task / "exp1" / str(set_name) / split_name / f"{eval_name}.set"
    set_scoped_meta = source_root / task / "exp1" / str(set_name) / split_name / f"{eval_name}.meta.jsonl"
    if set_scoped_set.exists() and set_scoped_meta.exists():
        return True

    root_level_set = source_root / task / "exp1" / split_name / f"{eval_name}.set"
    root_level_meta = source_root / task / "exp1" / split_name / f"{eval_name}.meta.jsonl"
    if root_level_set.exists() and root_level_meta.exists():
        return True

    if str(eval_name) == "unaugmented":
        fb_set_scoped_set = source_root / task / str(set_name) / "splits" / split_name / "unaugmented.set"
        fb_set_scoped_meta = source_root / task / str(set_name) / "splits" / split_name / "unaugmented.meta.jsonl"
        if fb_set_scoped_set.exists() and fb_set_scoped_meta.exists():
            return True

        fb_set = source_root / task / split_name / "unaugmented.set"
        fb_meta = source_root / task / split_name / "unaugmented.meta.jsonl"
        return fb_set.exists() and fb_meta.exists()
    return False


def _train_artifact_candidates(*, source_root: Path, task: str, set_name: str, variant: str) -> list[tuple[Path, Path]]:
    return [
        (
            source_root / task / set_name / "train" / f"{variant}.set",
            source_root / task / set_name / "train" / f"{variant}.meta.jsonl",
        ),
        (
            source_root / task / "exp1" / set_name / "train" / f"{variant}.set",
            source_root / task / "exp1" / set_name / "train" / f"{variant}.meta.jsonl",
        ),
        (
            source_root / task / "exp1" / set_name / "train" / variant / "train.set",
            source_root / task / "exp1" / set_name / "train" / variant / "train.meta.jsonl",
        ),
    ]


def _has_train_artifacts(*, source_root: Path, task: str, set_name: str, variant: str) -> bool:
    for set_path, meta_path in _train_artifact_candidates(
        source_root=source_root,
        task=task,
        set_name=set_name,
        variant=variant,
    ):
        if set_path.exists() and meta_path.exists():
            return True
    return False


def _source_root_has_job_artifacts(source_root: Path, job: Mapping[str, Any]) -> bool:
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
    ):
        return False

    for eval_name in eval_sets:
        if not _has_eval_artifacts(
            source_root=source_root,
            task=task,
            set_name=set_name,
            eval_name=eval_name,
            eval_split=eval_split,
        ):
            return False
    return True


def validate_exp1_source_root(
    *,
    cfg: Mapping[str, Any],
    tasks: list[str],
    set_names: list[str],
    source_root: Path,
) -> dict[str, Any]:
    src = Path(source_root)
    profiles_dir = str((((cfg.get("training") or {}).get("profiles_dir")) or "profiles"))
    jobs = build_exp1_run_matrix(cfg, tasks=list(tasks), set_names=list(set_names))
    missing_jobs: list[dict[str, Any]] = []
    seen_keys: set[tuple[Any, ...]] = set()
    checked_jobs = 0

    for job in jobs:
        dedupe_key = (
            str(job.get("task") or ""),
            str(job.get("set") or ""),
            str(job.get("model_id") or ""),
            str(job.get("profile_id") or ""),
            str(job.get("train_variant") or ""),
            str(job.get("eval_split") or "test"),
            tuple(str(x) for x in (job.get("eval_sets") or [])),
        )
        if dedupe_key in seen_keys:
            continue
        seen_keys.add(dedupe_key)
        checked_jobs += 1
        try:
            _resolve_source_root_for_job(
                preferred_source_root=src,
                base_results_root=src,
                profiles_dir=profiles_dir,
                job=job,
            )
            continue
        except FileNotFoundError as e:
            error_text = str(e)
        if _source_root_has_job_artifacts(src, job):
            continue
        missing_jobs.append(
            {
                "task": str(job.get("task") or ""),
                "set": str(job.get("set") or ""),
                "model_id": str(job.get("model_id") or ""),
                "train_variant": str(job.get("train_variant") or ""),
                "eval_split": str(job.get("eval_split") or "test"),
                "eval_sets": [str(x) for x in (job.get("eval_sets") or [])],
                "error": error_text,
                "train_candidates": [
                    {"set": str(set_path), "meta": str(meta_path)}
                    for set_path, meta_path in _train_artifact_candidates(
                        source_root=src,
                        task=str(job.get("task") or ""),
                        set_name=str(job.get("set") or ""),
                        variant=str(job.get("train_variant") or ""),
                    )
                ],
            }
        )

    return {
        "ok": not bool(missing_jobs),
        "source_root": str(src),
        "n_jobs": len(jobs),
        "n_checked_jobs": checked_jobs,
        "n_missing_jobs": len(missing_jobs),
        "missing_jobs": missing_jobs,
    }


def _resolve_source_root_for_job(
    *,
    preferred_source_root: Path,
    base_results_root: Path,
    profiles_dir: str,
    job: Mapping[str, Any],
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
        if _source_root_has_job_artifacts(cp, job):
            return cp
        return None

    direct = _try(Path(preferred_source_root))
    if direct is not None:
        return direct

    preferred_prof = _try(Path(preferred_source_root) / profiles_dir / str(job.get("profile_id") or ""))
    if preferred_prof is not None:
        return preferred_prof

    prof = _try(base_results_root / profiles_dir / str(job.get("profile_id") or ""))
    if prof is not None:
        return prof

    matches: list[Path] = []
    preferred_profiles_root = Path(preferred_source_root) / profiles_dir
    if preferred_profiles_root.exists():
        for p in sorted(preferred_profiles_root.iterdir()):
            if not p.is_dir() or _already_tried(p):
                continue
            if _source_root_has_job_artifacts(p, job):
                matches.append(p)

    profiles_root = base_results_root / profiles_dir
    if profiles_root.exists():
        for p in sorted(profiles_root.iterdir()):
            if not p.is_dir() or _already_tried(p):
                continue
            if _source_root_has_job_artifacts(p, job):
                matches.append(p)

    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError(
            "Ambiguous source roots for Exp1 job; matched multiple profile roots: "
            + ", ".join(str(p) for p in matches)
            + ". Use --source-root to select one explicitly."
        )

    task = str(job.get("task") or "")
    set_name = str(job.get("set") or "")
    variant = str(job.get("train_variant") or "")
    eval_split = str(job.get("eval_split") or "test")
    raise FileNotFoundError(
        f"Missing train/eval artifacts for task={task} set={set_name} variant={variant} eval_split={eval_split}. "
        f"Tried source roots: {', '.join(str(p) for p in tried)}"
    )


def _normalize_json_scalar(v: Any) -> Any:
    if isinstance(v, (int, float)):
        return float(v)
    return v


def _normalize_metrics_payload(metrics: Mapping[str, Mapping[str, Any]]) -> dict:
    out: dict[str, dict[str, Any]] = {}
    for eval_name, row in metrics.items():
        if not isinstance(row, Mapping):
            continue
        cur: dict[str, Any] = {}
        for k, v in row.items():
            if isinstance(v, (int, float)):
                cur[str(k)] = float(v)
            elif isinstance(v, list):
                vals: list[Any] = []
                for item in v:
                    if isinstance(item, Mapping):
                        vals.append({str(kk): _normalize_json_scalar(vv) for kk, vv in item.items()})
                    else:
                        vals.append(_normalize_json_scalar(item))
                cur[str(k)] = vals
            elif isinstance(v, Mapping):
                cur[str(k)] = {str(kk): _normalize_json_scalar(vv) for kk, vv in v.items()}
            else:
                cur[str(k)] = v
        out[str(eval_name)] = cur
    return out


def _run_cmd_stdout(args: list[str], *, cwd: Path) -> str | None:
    try:
        cp = subprocess.run(
            args,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except Exception:
        return None
    if cp.returncode != 0:
        return None
    out = str(cp.stdout or "").strip()
    return out or None


def _discover_git_root_candidates() -> list[Path]:
    candidates: list[Path] = []
    seen: set[str] = set()
    for base in [Path.cwd(), Path(__file__).resolve()]:
        cur = base if base.is_dir() else base.parent
        for p in [cur, *cur.parents]:
            key = str(p)
            if key in seen:
                continue
            seen.add(key)
            if (p / ".git").exists():
                candidates.append(p)
    return candidates


def _git_provenance() -> dict[str, Any]:
    for root in _discover_git_root_candidates():
        commit = _run_cmd_stdout(["git", "rev-parse", "HEAD"], cwd=root)
        if not commit:
            continue
        branch = _run_cmd_stdout(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=root)
        status = _run_cmd_stdout(["git", "status", "--porcelain"], cwd=root)
        return {
            "available": True,
            "repo_root": str(root),
            "commit": str(commit),
            "branch": str(branch) if branch else None,
            "dirty": bool(str(status or "").strip()),
        }
    return {
        "available": False,
        "repo_root": None,
        "commit": None,
        "branch": None,
        "dirty": None,
    }


def _package_versions() -> dict[str, str]:
    try:
        from importlib import metadata as ilm
    except Exception:  # pragma: no cover
        return {}

    out: dict[str, str] = {}
    for name in [
        "bio-tea",
        "bio-tea-runner",
        "transformers",
        "datasets",
        "torch",
        "numpy",
        "evaluate",
        "seqeval",
        "accelerate",
    ]:
        try:
            out[str(name)] = str(ilm.version(name))
        except Exception:
            continue
    return out


def _runtime_provenance() -> dict[str, Any]:
    return {
        "python": {
            "version": str(sys.version.split()[0]),
            "implementation": str(platform.python_implementation()),
        },
        "platform": {
            "system": str(platform.system()),
            "release": str(platform.release()),
            "machine": str(platform.machine()),
        },
        "packages": _package_versions(),
        "git": _git_provenance(),
    }


def _collect_run_provenance(cfg: Mapping[str, Any]) -> dict[str, Any]:
    cfg_json = json.loads(json.dumps(cfg, sort_keys=True, default=str))
    return {
        "config_sha256": _sha256_obj(cfg_json),
        "runtime": _runtime_provenance(),
    }


def _build_run_spec(
    *,
    job: Mapping[str, Any],
    prepared_manifest: Mapping[str, Any],
    cfg: Mapping[str, Any],
    provenance: Mapping[str, Any] | None = None,
) -> dict:
    exp1_cfg = dict(((cfg.get("training") or {}).get("exp1") or {}))
    tok_runtime = _tokenization_runtime_cfg(cfg)
    model_limits_runtime = _model_limits_runtime_cfg(cfg)
    hf_cfg = dict(((cfg.get("training") or {}).get("hf") or {}))
    payload = {
        "schema_version": 1,
        "provenance": dict(provenance or {}),
        "job": {
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
            "hparams": dict(job.get("hparams") or exp1_cfg.get("hparams") or {}),
            "save_model": dict((exp1_cfg.get("save_model") or {})),
            "tokenization": dict(tok_runtime),
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


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return dict(json.loads(path.read_text(encoding="utf-8")) or {})
    except Exception:
        return None


def _run_is_resume_compatible(*, run_dir: Path, run_spec: Mapping[str, Any]) -> bool:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return False

    prev_spec = _load_json(run_dir / "run_spec.json")
    if not isinstance(prev_spec, dict):
        return False
    prev_sig = str(prev_spec.get("signature") or "")
    cur_sig = str(run_spec.get("signature") or "")
    if not prev_sig or prev_sig != cur_sig:
        return False

    prev_state = _load_json(run_dir / "run_state.json")
    if isinstance(prev_state, dict):
        st = str(prev_state.get("status") or "")
        if st and st != "completed":
            return False
    return True


def _write_run_state(*, run_dir: Path, status: str, signature: str, extra: Mapping[str, Any] | None = None) -> None:
    obj = {
        "schema_version": 1,
        "status": str(status),
        "signature": str(signature),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if isinstance(extra, Mapping):
        obj.update({str(k): v for k, v in extra.items()})
    write_json(run_dir / "run_state.json", obj)
    _sync_status_tag(run_dir=run_dir, status=str(status))


def _model_artifacts_manifest(*, save_model: bool, model_dir: Path, policy_mode: str) -> dict[str, Any]:
    if not save_model:
        return {
            "schema_version": 1,
            "saved": False,
            "mode": str(policy_mode),
            "model_dir": str(model_dir),
            "n_files": 0,
            "total_bytes": 0,
            "files": [],
        }
    if not model_dir.exists():
        return {
            "schema_version": 1,
            "saved": False,
            "mode": str(policy_mode),
            "model_dir": str(model_dir),
            "reason": "missing_model_dir",
            "n_files": 0,
            "total_bytes": 0,
            "files": [],
        }

    files: list[dict[str, Any]] = []
    for fp in sorted(model_dir.rglob("*")):
        if not fp.is_file():
            continue
        rel = str(fp.relative_to(model_dir))
        sz = int(fp.stat().st_size)
        files.append(
            {
                "path": rel,
                "size": sz,
                "sha256": sha256_file(fp),
            }
        )
    return {
        "schema_version": 1,
        "saved": True,
        "mode": str(policy_mode),
        "model_dir": str(model_dir),
        "n_files": len(files),
        "total_bytes": int(sum(int(x["size"]) for x in files)),
        "files": files,
    }


def run_exp1_matrix(
    *,
    cfg: Mapping[str, Any],
    tasks: list[str] | None = None,
    set_names: list[str] | None = None,
    source_root: Path | None = None,
    backend: Callable[..., Mapping[str, Mapping[str, Any]]] | None = None,
) -> dict:
    """Prepare and execute Exp1 training/eval jobs."""

    if source_root is None:
        resolved, _info = resolve_results_root(cfg, cfg.get("tokenizer") or {})
        source_root = resolved
    source_root = Path(source_root)

    if tasks is None:
        tasks = _discover_tasks(source_root)
    if not tasks:
        raise RuntimeError(f"No tasks found under source_root={source_root}")

    if set_names is None:
        set_names = _discover_sets(source_root, tasks[0])
    if not set_names:
        raise RuntimeError(f"No set directories found under source_root={source_root} for task={tasks[0]}")

    jobs = build_exp1_run_matrix(cfg, tasks=tasks, set_names=set_names)
    if not jobs:
        return {
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
    tokenization_runtime = _tokenization_runtime_cfg(cfg)
    model_limits_runtime = _model_limits_runtime_cfg(cfg)
    exp1_runtime = _exp1_runtime_cfg(cfg)
    save_cfg = _save_model_cfg(cfg)
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
            )
            prep = prepare_job_datasets(
                source_root=job_source_root,
                output_profile_root=output_profile_root,
                job=job,
                tokenizer=tokenizer,
                tokenizer_info=tok_info,
                fail_on_truncation=bool(tokenization_runtime.get("fail_on_truncation", True)),
            )

            prepared_dir = Path(prep["prepared_dir"])
            prepared_manifest = json.loads((prepared_dir / "manifest.json").read_text(encoding="utf-8"))
            run_dir.mkdir(parents=True, exist_ok=True)
            run_spec = _build_run_spec(
                job=job,
                prepared_manifest=prepared_manifest,
                cfg=cfg,
                provenance=run_provenance,
            )
            signature = str(run_spec["signature"])
            if exp1_runtime["resume"] and (not exp1_runtime["force"]):
                if _run_is_resume_compatible(run_dir=run_dir, run_spec=run_spec):
                    skipped += 1
                    task_exp1_root = output_profile_root / str(job["task"]) / "exp1"
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
                            "task_exp1_root": str(task_exp1_root),
                            "source_root": str(job_source_root),
                            "hf_assets": asset_refs,
                        }
                    )
                    continue

            prev_spec = _load_json(run_dir / "run_spec.json")
            prev_sig = str((prev_spec or {}).get("signature") or "")
            archive_reason = "rerun"
            if exp1_runtime["force"]:
                archive_reason = "force_rerun"
            elif prev_sig and prev_sig != signature:
                archive_reason = "signature_changed"
            elif exp1_runtime["resume"] and (not exp1_runtime["force"]):
                archive_reason = "resume_disabled_or_incomplete"
            archive_dir = _archive_existing_run_artifacts(
                run_dir=run_dir,
                reason=archive_reason,
                previous_signature=prev_sig or None,
            )

            write_json(run_dir / "run_spec.json", run_spec)
            _write_run_state(run_dir=run_dir, status="running", signature=signature)
            save_model = _should_save_model_for_job(job=job, cfg=cfg, save_cfg=save_cfg)
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

            task_exp1_root = output_profile_root / str(job["task"]) / "exp1"
            run_obj = {
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
                    "task_exp1_root": str(task_exp1_root),
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
        "source_root": str(source_root),
        "n_jobs": len(jobs),
        "n_ok": ok,
        "n_failed": failed,
        "n_skipped": skipped,
        "exp1_runtime": exp1_runtime,
        "save_model": save_cfg,
        "hf_runtime": hf_runtime,
        "tokenization_runtime": tokenization_runtime,
        "model_limits_runtime": model_limits_runtime,
        "hf_preload": preload_stats,
        "hf_asset_manifest_path": str(asset_manifest_path) if asset_manifest_path is not None else None,
        "provenance": run_provenance,
        "jobs": job_summaries,
    }
    return summary
