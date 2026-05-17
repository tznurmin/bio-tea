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

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from bio_tea.datasets.io import write_json, write_jsonl

from .data_prep import build_label_maps, load_canonical_rows, prepare_tokenized_rows
from .exp1_matrix import prepared_dir_for_job
from .experiment_matrix import prepared_dir_for_job as prepared_dir_for_job_generic


def _sha256_obj(obj: object) -> str:
    data = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _source_set_paths(
    *,
    source_root: Path,
    task: str,
    set_name: str,
    variant: str,
    experiment_id: str = "exp1",
) -> tuple[Path, Path]:
    exp_id = str(experiment_id or "exp1")
    candidates: list[tuple[Path, Path]] = []

    # Task/set train layout.
    if exp_id == "exp1":
        candidates.append(
            (
                source_root / task / set_name / "train" / f"{variant}.set",
                source_root / task / set_name / "train" / f"{variant}.meta.jsonl",
            )
        )

    # Flat variant files under task/<exp>/set/train.
    candidates.append(
        (
            source_root / task / exp_id / set_name / "train" / f"{variant}.set",
            source_root / task / exp_id / set_name / "train" / f"{variant}.meta.jsonl",
        )
    )
    # Directory variant layout under task/<exp>/set/train/<variant>/train.set.
    candidates.append(
        (
            source_root / task / exp_id / set_name / "train" / variant / "train.set",
            source_root / task / exp_id / set_name / "train" / variant / "train.meta.jsonl",
        )
    )

    for set_path, meta_path in candidates:
        if set_path.exists() and meta_path.exists():
            return set_path, meta_path
    return candidates[0]


def _source_eval_paths(
    *,
    source_root: Path,
    task: str,
    set_name: str,
    eval_name: str,
    experiment_id: str = "exp1",
    eval_split: str = "test",
) -> tuple[Path, Path]:
    exp_id = str(experiment_id or "exp1")
    split_name = str(eval_split or "test")
    candidates: list[tuple[Path, Path]] = [
        (
            source_root / task / exp_id / str(set_name) / split_name / f"{eval_name}.set",
            source_root / task / exp_id / str(set_name) / split_name / f"{eval_name}.meta.jsonl",
        ),
        (
            source_root / task / exp_id / split_name / f"{eval_name}.set",
            source_root / task / exp_id / split_name / f"{eval_name}.meta.jsonl",
        ),
    ]
    for set_path, meta_path in candidates:
        if set_path.exists() and meta_path.exists():
            return set_path, meta_path

    # Fallback for unaugmented to base split artifacts.
    if str(eval_name) == "unaugmented":
        fb_set_set = source_root / task / str(set_name) / "splits" / split_name / "unaugmented.set"
        fb_meta_set = source_root / task / str(set_name) / "splits" / split_name / "unaugmented.meta.jsonl"
        if fb_set_set.exists() and fb_meta_set.exists():
            return fb_set_set, fb_meta_set
        fb_set = source_root / task / split_name / "unaugmented.set"
        fb_meta = source_root / task / split_name / "unaugmented.meta.jsonl"
        if fb_set.exists() and fb_meta.exists():
            return fb_set, fb_meta

    fallback_set, fallback_meta = candidates[0]
    raise FileNotFoundError(f"Missing eval set artifacts for eval_name={eval_name}: {fallback_set} / {fallback_meta}")


def _prepared_fingerprint(
    *,
    job: Mapping[str, Any],
    tokenizer_info: Mapping[str, Any] | None,
    label2id: Mapping[str, int],
    train_rows: list[dict],
    eval_rows: Mapping[str, list[dict]],
) -> str:
    payload = {
        "job": {
            "experiment_id": str(job.get("experiment_id") or "exp1"),
            "task": str(job.get("task")),
            "set": str(job.get("set")),
            "model_id": str(job.get("model_id")),
            "seed": int(job.get("seed", 0)),
            "train_variant": str(job.get("train_variant")),
            "eval_split": str(job.get("eval_split") or "test"),
            "max_length": int(job.get("max_length", 0)),
        },
        "tokenizer_info": dict(tokenizer_info or {}),
        "label2id": dict(sorted(((str(k), int(v)) for k, v in label2id.items()), key=lambda kv: kv[0])),
        "train_rows": train_rows,
        "eval_rows": {k: eval_rows[k] for k in sorted(eval_rows.keys())},
    }
    return _sha256_obj(payload)


def prepare_job_datasets(
    *,
    source_root: Path,
    output_profile_root: Path,
    job: Mapping[str, Any],
    tokenizer: Any,
    tokenizer_info: Mapping[str, Any] | None = None,
    experiment_id: str = "exp1",
    fail_on_truncation: bool = False,
    lowercase_tokens: bool | None = None,
) -> dict:
    """Prepare tokenized train/eval artifacts for one Exp1 job."""

    exp_id = str(experiment_id or "exp1")
    task = str(job["task"])
    set_name = str(job["set"])
    train_variant = str(job["train_variant"])
    eval_split = str(job.get("eval_split") or "test")
    eval_sets = [str(x) for x in (job.get("eval_sets") or ["unaugmented", "augmented_exclusive"])]
    max_length = int(job.get("max_length", 510))

    train_set, train_meta = _source_set_paths(
        source_root=Path(source_root),
        task=task,
        set_name=set_name,
        variant=train_variant,
        experiment_id=exp_id,
    )
    if not train_set.exists() or not train_meta.exists():
        raise FileNotFoundError(f"Missing train artifacts: {train_set} / {train_meta}")

    train_rows = load_canonical_rows(train_set, train_meta)

    eval_canonical: dict[str, list[dict]] = {}
    for name in eval_sets:
        ep_set, ep_meta = _source_eval_paths(
            source_root=Path(source_root),
            task=task,
            set_name=set_name,
            eval_name=name,
            experiment_id=exp_id,
            eval_split=eval_split,
        )
        eval_canonical[name] = load_canonical_rows(ep_set, ep_meta)

    # Build label map from train + eval rows to avoid unknown-label alignment failures.
    all_for_labels = list(train_rows)
    for rows in eval_canonical.values():
        all_for_labels.extend(rows)
    label2id, id2label = build_label_maps(all_for_labels)

    # Keep original curated token casing by default and rely on tokenizer runtime
    # behavior (HF tokenizer/model pair) unless explicit override is provided.
    resolved_lowercase_tokens = bool(lowercase_tokens) if lowercase_tokens is not None else False

    train_tok = prepare_tokenized_rows(
        rows=train_rows,
        tokenizer=tokenizer,
        label2id=label2id,
        max_length=max_length,
        fail_on_truncation=bool(fail_on_truncation),
        lowercase_tokens=resolved_lowercase_tokens,
    )
    eval_tok: dict[str, list[dict]] = {}
    for name, rows in eval_canonical.items():
        eval_tok[name] = prepare_tokenized_rows(
            rows=rows,
            tokenizer=tokenizer,
            label2id=label2id,
            max_length=max_length,
            fail_on_truncation=bool(fail_on_truncation),
            lowercase_tokens=resolved_lowercase_tokens,
        )

    if exp_id == "exp1":
        out_dir = prepared_dir_for_job(Path(output_profile_root), job)
    else:
        job2 = dict(job)
        job2["experiment_id"] = exp_id
        out_dir = prepared_dir_for_job_generic(Path(output_profile_root), job2)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "train.tokenized.jsonl", train_tok)
    for name, rows in eval_tok.items():
        write_jsonl(out_dir / f"eval.{name}.tokenized.jsonl", rows)

    fp = _prepared_fingerprint(
        job=job,
        tokenizer_info=tokenizer_info,
        label2id=label2id,
        train_rows=train_tok,
        eval_rows=eval_tok,
    )
    manifest = {
        "experiment_id": exp_id,
        "task": task,
        "set": set_name,
        "model_id": str(job.get("model_id")),
        "seed": int(job.get("seed", 0)),
        "train_variant": train_variant,
        "eval_split": eval_split,
        "eval_sets": eval_sets,
        "max_length": max_length,
        "tokenizer": dict(tokenizer_info or {}),
        "fail_on_truncation": bool(fail_on_truncation),
        "lowercase_tokens": resolved_lowercase_tokens,
        "label2id": dict(label2id),
        "id2label": {str(k): v for k, v in id2label.items()},
        "n_train_examples": len(train_tok),
        "n_eval_examples": {k: len(v) for k, v in eval_tok.items()},
        "fingerprint": fp,
    }
    write_json(out_dir / "manifest.json", manifest)

    return {
        "prepared_dir": str(out_dir),
        "fingerprint": fp,
        "n_train_examples": len(train_tok),
        "n_eval_examples": {k: len(v) for k, v in eval_tok.items()},
    }
