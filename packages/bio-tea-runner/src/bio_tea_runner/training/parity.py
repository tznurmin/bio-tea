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

from copy import deepcopy
from typing import Any, Callable, Mapping, Sequence


PARITY_METRICS = ("loss", "precision", "recall", "f1")


def _canonical_hparams(
    cfg: Mapping[str, Any],
    *,
    experiment_id: str,
    job: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    exp_id = str(experiment_id or "exp1")
    hparams = dict((((cfg.get("training") or {}).get(exp_id) or {}).get("hparams") or {})
    )
    hparams.update(dict((job or {}).get("hparams") or {}))
    return {
        "epochs": int(hparams.get("epochs", 1)),
        "learning_rate": float(hparams.get("learning_rate", 5e-5)),
        "per_device_train_batch_size": int(hparams.get("per_device_train_batch_size", 8)),
        "per_device_eval_batch_size": int(hparams.get("per_device_eval_batch_size", 8)),
    }


def build_parity_signature(
    *,
    prepared_manifest: Mapping[str, Any],
    job: Mapping[str, Any],
    cfg: Mapping[str, Any],
    experiment_id: str = "exp1",
) -> dict[str, Any]:
    exp_id = str(experiment_id or "exp1")
    return {
        "experiment_id": exp_id,
        "prepared_fingerprint": str(prepared_manifest.get("fingerprint") or ""),
        "label2id": dict(prepared_manifest.get("label2id") or {}),
        "eval_sets": [str(x) for x in list(prepared_manifest.get("eval_sets") or [])],
        "model_name_or_path": str(job.get("model_name_or_path") or ""),
        "model_revision": str(job.get("model_revision") or ""),
        "seed": int(job.get("seed", 0)),
        "max_length": int(job.get("max_length", 0)),
        "hparams": _canonical_hparams(cfg, experiment_id=exp_id, job=job),
    }


def contract_mismatches(sig_a: Mapping[str, Any], sig_b: Mapping[str, Any]) -> list[str]:
    mismatches: list[str] = []
    keys = sorted(set(str(k) for k in sig_a.keys()) | set(str(k) for k in sig_b.keys()))
    for k in keys:
        a = sig_a.get(k)
        b = sig_b.get(k)
        if a != b:
            mismatches.append(str(k))
    return mismatches


def compare_metric_payloads(
    metrics_a: Mapping[str, Mapping[str, Any]],
    metrics_b: Mapping[str, Mapping[str, Any]],
    *,
    abs_tol: float = 1e-6,
) -> dict[str, Any]:
    mismatches: list[str] = []
    eval_keys = sorted(set(str(k) for k in metrics_a.keys()) | set(str(k) for k in metrics_b.keys()))
    for eval_name in eval_keys:
        row_a = metrics_a.get(eval_name)
        row_b = metrics_b.get(eval_name)
        if not isinstance(row_a, Mapping) or not isinstance(row_b, Mapping):
            mismatches.append(f"missing_eval:{eval_name}")
            continue
        for mk in PARITY_METRICS:
            if mk not in row_a or mk not in row_b:
                mismatches.append(f"missing_metric:{eval_name}:{mk}")
                continue
            try:
                va = float(row_a[mk])
                vb = float(row_b[mk])
            except Exception:
                mismatches.append(f"non_numeric_metric:{eval_name}:{mk}")
                continue
            if abs(va - vb) > float(abs_tol):
                mismatches.append(f"value_mismatch:{eval_name}:{mk}:{va}:{vb}")
    return {
        "ok": len(mismatches) == 0,
        "mismatches": mismatches,
    }


def run_hf_parity_harness(
    *,
    prepared_dir: Any,
    prepared_manifest: Mapping[str, Any],
    job: Mapping[str, Any],
    cfg: Mapping[str, Any],
    runner_a: Callable[..., Mapping[str, Mapping[str, Any]]],
    runner_b: Callable[..., Mapping[str, Mapping[str, Any]]] | None = None,
    job_b: Mapping[str, Any] | None = None,
    cfg_b: Mapping[str, Any] | None = None,
    experiment_id: str = "exp1",
    abs_tol: float = 1e-6,
) -> dict[str, Any]:
    """Run two HF-compatible backends under parity contract checks.

    Contract requires identical core configuration (seed/model/dataset/hparams/etc.).
    """

    exp_id = str(experiment_id or "exp1")
    cfg_right = dict(cfg_b or cfg)
    job_right = dict(job_b or job)
    runner_right = runner_b or runner_a

    sig_a = build_parity_signature(
        prepared_manifest=prepared_manifest,
        job=job,
        cfg=cfg,
        experiment_id=exp_id,
    )
    sig_b = build_parity_signature(
        prepared_manifest=prepared_manifest,
        job=job_right,
        cfg=cfg_right,
        experiment_id=exp_id,
    )
    mismatches = contract_mismatches(sig_a, sig_b)
    if mismatches:
        raise ValueError(
            "parity contract mismatch: " + ", ".join(mismatches)
        )

    # Run both backends on identical prepared artifacts.
    m_a = runner_a(
        prepared_dir=prepared_dir,
        prepared_manifest=deepcopy(dict(prepared_manifest)),
        job=deepcopy(dict(job)),
        cfg=deepcopy(dict(cfg)),
    )
    m_b = runner_right(
        prepared_dir=prepared_dir,
        prepared_manifest=deepcopy(dict(prepared_manifest)),
        job=deepcopy(dict(job_right)),
        cfg=deepcopy(dict(cfg_right)),
    )
    metrics_cmp = compare_metric_payloads(m_a, m_b, abs_tol=float(abs_tol))
    return {
        "contract_ok": True,
        "contract_signature": sig_a,
        "metrics_ok": bool(metrics_cmp["ok"]),
        "metric_mismatches": list(metrics_cmp["mismatches"]),
        "metrics_a": m_a,
        "metrics_b": m_b,
        "abs_tol": float(abs_tol),
    }
