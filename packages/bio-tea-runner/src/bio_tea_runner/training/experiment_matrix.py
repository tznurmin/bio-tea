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

from pathlib import Path
from typing import Any, Mapping

from .exp1_matrix import normalize_training_models
from .param_resolution import (
    resolve_eval_sets_for_model,
    resolve_eval_split_for_model,
    resolve_hparams_for_job,
)


DEFAULT_TRAIN_VARIANTS: dict[str, list[str]] = {
    "exp1": ["none", "all", "combined"],
    "exp2": ["none", "species", "strains", "all"],
}

DEFAULT_EVAL_SETS: list[str] = ["unaugmented", "augmented_exclusive"]


def _as_list(v: object) -> list:
    if v is None:
        return []
    if isinstance(v, list):
        return v
    if isinstance(v, tuple):
        return list(v)
    return [v]


def build_run_matrix(
    cfg: Mapping[str, Any],
    *,
    experiment_id: str,
    tasks: list[str],
    set_names: list[str],
    train_variants: list[str],
) -> list[dict]:
    """Build deterministic run jobs for one experiment."""

    exp_id = str(experiment_id or "").strip().lower()
    if not exp_id:
        raise ValueError("experiment_id is required")

    train_cfg = cfg.get("training") or {}
    exp_cfg = train_cfg.get(exp_id) or {}
    if not bool(exp_cfg.get("enabled", True)):
        return []

    seeds = [int(x) for x in _as_list(exp_cfg.get("seeds") or [0])]
    if not seeds:
        seeds = [0]
    default_eval_sets = [str(v) for v in _as_list(exp_cfg.get("eval_sets") or list(DEFAULT_EVAL_SETS))]
    models = normalize_training_models(cfg)

    jobs: list[dict] = []
    for task in tasks:
        for set_name in set_names:
            for model in models:
                model_cfg = dict(model.get("model_config") or {})
                eval_sets = resolve_eval_sets_for_model(
                    cfg=cfg,
                    experiment_id=exp_id,
                    model_cfg=model_cfg,
                    default_eval_sets=list(default_eval_sets),
                )
                eval_split = resolve_eval_split_for_model(
                    cfg=cfg,
                    experiment_id=exp_id,
                    model_cfg=model_cfg,
                    default_split="test",
                )
                for seed in seeds:
                    for variant in train_variants:
                        resolved_hparams = resolve_hparams_for_job(
                            cfg=cfg,
                            experiment_id=exp_id,
                            model_cfg=model_cfg,
                            train_variant=str(variant),
                        )
                        jobs.append(
                            {
                                "experiment_id": exp_id,
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
    exp_id = str(job.get("experiment_id") or "exp1")
    return (
        Path(results_root)
        / str(job["task"])
        / exp_id
        / "prepared"
        / str(job["model_id"])
        / str(job["set"])
        / str(job["train_variant"])
    )


def run_dir_for_job(results_root: Path, job: Mapping[str, Any]) -> Path:
    exp_id = str(job.get("experiment_id") or "exp1")
    return (
        Path(results_root)
        / str(job["task"])
        / exp_id
        / "runs"
        / str(job["model_id"])
        / str(job["set"])
        / f"seed_{int(job['seed'])}"
        / str(job["train_variant"])
    )
