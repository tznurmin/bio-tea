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

import re
from pathlib import Path
from typing import Any, Mapping

from .param_resolution import (
    resolve_eval_sets_for_model,
    resolve_eval_split_for_model,
    resolve_hparams_for_job,
)


def _slug(v: object, *, fallback: str = "na", max_len: int = 80) -> str:
    s = str(v or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    if not s:
        s = fallback
    if len(s) > max_len:
        s = s[:max_len].rstrip("-")
    return s or fallback


def _as_list(v: object) -> list:
    if v is None:
        return []
    if isinstance(v, list):
        return v
    if isinstance(v, tuple):
        return list(v)
    return [v]


def normalize_training_models(cfg: Mapping[str, Any]) -> list[dict]:
    """Normalize training model configs for Exp1 runner."""

    tok_default = dict(cfg.get("tokenizer") or {})
    train_cfg = cfg.get("training") or {}
    hf_cfg = dict(train_cfg.get("hf") or {})
    hf_local_files_only = bool(hf_cfg.get("local_files_only", False))
    hf_revision_raw = hf_cfg.get("revision")
    hf_revision = str(hf_revision_raw).strip() if hf_revision_raw is not None else ""
    hf_revision = hf_revision or None
    raw_models = list(_as_list(train_cfg.get("models")))
    if not raw_models:
        fallback_model = str(
            train_cfg.get("default_model_name_or_path")
            or tok_default.get("name_or_path")
            or ""
        ).strip()
        if not fallback_model:
            raise ValueError(
                "No model source configured: provide training.models or set "
                "training.default_model_name_or_path/tokenizer.name_or_path"
            )
        raw_models = [
            {
                "id": _slug(fallback_model, fallback="model-1"),
                "model_name_or_path": fallback_model,
                "tokenizer": dict(tok_default),
                "max_length": int(tok_default.get("model_max_length") or 510),
            }
        ]

    out: list[dict] = []
    for i, m in enumerate(raw_models):
        if not isinstance(m, Mapping):
            raise ValueError(f"training.models[{i}] must be an object")

        model_name = str(m.get("model_name_or_path") or m.get("name_or_path") or "").strip()
        if not model_name:
            raise ValueError(f"training.models[{i}].model_name_or_path is required")

        model_id = str(m.get("id") or "").strip() or _slug(model_name, fallback=f"model-{i+1}")

        model_tok_cfg = dict(m.get("tokenizer") or {})
        tok_cfg = dict(tok_default)
        tok_cfg.update(model_tok_cfg)
        if not str(tok_cfg.get("name_or_path") or "").strip():
            tok_cfg["name_or_path"] = model_name

        # Do not carry casing-policy defaults across different tokenizer families
        # unless explicitly configured for the model tokenizer entry.
        default_tok_name = str(tok_default.get("name_or_path") or "").strip()
        resolved_tok_name = str(tok_cfg.get("name_or_path") or "").strip()
        if default_tok_name and resolved_tok_name and (resolved_tok_name != default_tok_name):
            for key in ("do_lower_case", "require_case_sensitive", "require_cased"):
                if key not in model_tok_cfg:
                    tok_cfg.pop(key, None)

        max_len = int(m.get("max_length") or tok_cfg.get("model_max_length") or 510)
        tok_cfg["model_max_length"] = max_len
        model_local_files_only = bool(m.get("local_files_only", hf_local_files_only))
        model_revision_raw = m.get("revision", hf_revision)
        model_revision = str(model_revision_raw).strip() if model_revision_raw is not None else ""
        model_revision = model_revision or None
        check_model_max_length_raw = m.get("check_model_max_length")
        check_model_max_length = (
            None if check_model_max_length_raw is None else bool(check_model_max_length_raw)
        )

        profile_id = (
            f"model-{_slug(model_name, fallback='model')}"
            f"--tok-{_slug(tok_cfg.get('name_or_path'), fallback='tokenizer')}"
            f"--ml-{max_len}"
        )

        out.append(
            {
                "id": model_id,
                "model_name_or_path": model_name,
                "tokenizer": tok_cfg,
                "max_length": max_len,
                "profile_id": profile_id,
                "model_local_files_only": model_local_files_only,
                "model_revision": model_revision,
                "check_model_max_length": check_model_max_length,
                "model_config": dict(m),
            }
        )
    return out


def build_exp1_run_matrix(
    cfg: Mapping[str, Any],
    *,
    tasks: list[str],
    set_names: list[str],
) -> list[dict]:
    """Build deterministic Exp1 run jobs from config."""

    train_cfg = cfg.get("training") or {}
    exp1_cfg = train_cfg.get("exp1") or {}
    if not bool(exp1_cfg.get("enabled", True)):
        return []

    seeds = [int(x) for x in _as_list(exp1_cfg.get("seeds") or [0])]
    if not seeds:
        seeds = [0]
    variants = [str(v) for v in _as_list(exp1_cfg.get("train_variants") or ["none", "all", "combined"])]
    default_eval_sets = [str(v) for v in _as_list(exp1_cfg.get("eval_sets") or ["unaugmented", "augmented_exclusive"])]
    models = normalize_training_models(cfg)

    jobs: list[dict] = []
    for task in tasks:
        for set_name in set_names:
            for model in models:
                model_cfg = dict(model.get("model_config") or {})
                eval_sets = resolve_eval_sets_for_model(
                    cfg=cfg,
                    experiment_id="exp1",
                    model_cfg=model_cfg,
                    default_eval_sets=list(default_eval_sets),
                )
                eval_split = resolve_eval_split_for_model(
                    cfg=cfg,
                    experiment_id="exp1",
                    model_cfg=model_cfg,
                    default_split="test",
                )
                for seed in seeds:
                    for variant in variants:
                        resolved_hparams = resolve_hparams_for_job(
                            cfg=cfg,
                            experiment_id="exp1",
                            model_cfg=model_cfg,
                            train_variant=str(variant),
                        )
                        jobs.append(
                            {
                                "task": str(task),
                                "set": str(set_name),
                                "model_id": str(model["id"]),
                                "model_name_or_path": str(model["model_name_or_path"]),
                                "tokenizer": dict(model["tokenizer"]),
                                "max_length": int(model["max_length"]),
                                "profile_id": str(model["profile_id"]),
                                "model_local_files_only": bool(model.get("model_local_files_only", False)),
                                "model_revision": model.get("model_revision"),
                                "check_model_max_length": model.get("check_model_max_length"),
                                "seed": int(seed),
                                "train_variant": str(variant),
                                "eval_split": eval_split,
                                "eval_sets": list(eval_sets),
                                "hparams": dict(resolved_hparams),
                            }
                        )
    return jobs


def prepared_dir_for_job(results_root: Path, job: Mapping[str, Any]) -> Path:
    return (
        Path(results_root)
        / str(job["task"])
        / "exp1"
        / "prepared"
        / str(job["model_id"])
        / str(job["set"])
        / str(job["train_variant"])
    )


def run_dir_for_job(results_root: Path, job: Mapping[str, Any]) -> Path:
    return (
        Path(results_root)
        / str(job["task"])
        / "exp1"
        / "runs"
        / str(job["model_id"])
        / str(job["set"])
        / f"seed_{int(job['seed'])}"
        / str(job["train_variant"])
    )
