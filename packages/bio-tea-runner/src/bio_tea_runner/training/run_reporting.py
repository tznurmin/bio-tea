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
import math
import random
from numbers import Integral, Real
from pathlib import Path
from statistics import NormalDist, mean, median, stdev
from typing import Any, Mapping, Sequence

from bio_tea.datasets.reporting import load_callable, normalize_ci_config


METRIC_KEYS = ("loss", "precision", "recall", "f1")


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("cannot compute percentile of empty values")
    if q <= 0:
        return float(sorted_values[0])
    if q >= 1:
        return float(sorted_values[-1])
    n = len(sorted_values)
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_values[lo])
    w = pos - lo
    return float((1.0 - w) * sorted_values[lo] + w * sorted_values[hi])


def _point_fn(name: str):
    nm = str(name or "mean").strip().lower()
    if nm == "mean":
        return lambda xs: float(mean(xs))
    if nm == "median":
        return lambda xs: float(median(xs))
    raise ValueError("point_estimator must be one of: mean, median")


def _require_metric_number(value: Any, *, field: str) -> float:
    if value is None:
        raise ValueError(f"{field} must not be null")
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field} must be numeric, got {type(value).__name__}")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{field} must be finite")
    metric_name = str(field).split(".")[-1]
    if metric_name in {"precision", "recall", "f1"} and not (0.0 <= out <= 1.0):
        raise ValueError(f"{field} must be in [0, 1], got {out}")
    return out


def _require_seed_value(value: Any) -> int:
    if value is None:
        raise ValueError("seed must not be null")
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"seed must be an integer, got {type(value).__name__}")
    return int(value)


def _require_set_value(value: Any) -> str:
    if value is None:
        raise ValueError("set must not be null")
    if not isinstance(value, str):
        raise TypeError(f"set must be text, got {type(value).__name__}")
    return value


def _representative_row_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    # Median reports are intended to select one deterministic representative run
    # tuple rather than mixing per-metric medians across different runs.
    return (
        _require_metric_number(row.get("f1"), field="f1"),
        _require_metric_number(row.get("precision"), field="precision"),
        _require_metric_number(row.get("recall"), field="recall"),
        _require_metric_number(row.get("loss"), field="loss"),
        _require_set_value(row.get("set")),
        _require_seed_value(row.get("seed")),
    )


def _select_representative_row(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ordered = sorted((dict(r) for r in rows), key=_representative_row_sort_key)
    if not ordered:
        raise ValueError("cannot select representative row from empty group")
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return dict(ordered[mid])

    lower = dict(ordered[mid - 1])
    upper = dict(ordered[mid])
    return {
        "_synthetic": True,
        "_source_rows": [lower, upper],
        "loss": float(median([_require_metric_number(lower.get("loss"), field="loss"), _require_metric_number(upper.get("loss"), field="loss")])),
        "precision": float(median([_require_metric_number(lower.get("precision"), field="precision"), _require_metric_number(upper.get("precision"), field="precision")])),
        "recall": float(median([_require_metric_number(lower.get("recall"), field="recall"), _require_metric_number(upper.get("recall"), field="recall")])),
        "f1": float(median([_require_metric_number(lower.get("f1"), field="f1"), _require_metric_number(upper.get("f1"), field="f1")])),
    }


def _ci_from_bootstrap_samples(
    *,
    point: float,
    boots: Sequence[float],
    cfg: Mapping[str, Any],
) -> dict[str, list[float]]:
    alpha = 1.0 - float(cfg["confidence_level"])
    out: dict[str, list[float]] = {}
    if not boots:
        return out
    sorted_boots = sorted(float(x) for x in boots)
    for method in cfg.get("methods") or []:
        mm = str(method)
        if mm == "percentile":
            lo = _percentile(sorted_boots, alpha / 2.0)
            hi = _percentile(sorted_boots, 1.0 - alpha / 2.0)
            out[mm] = [float(lo), float(hi)]
        elif mm == "basic":
            q_lo = _percentile(sorted_boots, alpha / 2.0)
            q_hi = _percentile(sorted_boots, 1.0 - alpha / 2.0)
            out[mm] = [float(2.0 * point - q_hi), float(2.0 * point - q_lo)]
        else:
            raise ValueError(f"Unsupported CI method for representative-run metrics: {mm}")
    return out


def _bootstrap_representative_row_ci(
    rows: Sequence[Mapping[str, Any]],
    *,
    cfg: Mapping[str, Any],
) -> dict[str, dict[str, list[float]]]:
    vals = [dict(x) for x in rows]
    if not vals:
        return {}
    rng = random.Random(int(cfg["seed"]))
    n = len(vals)
    point_row = _select_representative_row(vals)
    boot_metric_values: dict[str, list[float]] = {mk: [] for mk in METRIC_KEYS}

    for _ in range(int(cfg["n_bootstrap"])):
        samp = [vals[rng.randrange(n)] for _ in range(n)]
        rep = _select_representative_row(samp)
        for mk in METRIC_KEYS:
            boot_metric_values[mk].append(float(rep[mk]))

    return {
        mk: _ci_from_bootstrap_samples(
            point=float(point_row[mk]),
            boots=boot_metric_values[mk],
            cfg=cfg,
        )
        for mk in METRIC_KEYS
    }


def _bootstrap_ci(values: Sequence[float], *, cfg: Mapping[str, Any], point_estimator: str) -> dict[str, list[float]]:
    vals = [float(x) for x in values]
    if not vals:
        return {}
    pfn = _point_fn(point_estimator)
    point = float(pfn(vals))
    alpha = 1.0 - float(cfg["confidence_level"])
    rng = random.Random(int(cfg["seed"]))
    n = len(vals)

    boots: list[float] = []
    for _ in range(int(cfg["n_bootstrap"])):
        samp = [vals[rng.randrange(n)] for _ in range(n)]
        boots.append(float(pfn(samp)))
    boots.sort()

    out: dict[str, list[float]] = {}
    for method in cfg.get("methods") or []:
        mm = str(method)
        if mm == "percentile":
            lo = _percentile(boots, alpha / 2.0)
            hi = _percentile(boots, 1.0 - alpha / 2.0)
            out[mm] = [float(lo), float(hi)]
        elif mm == "basic":
            q_lo = _percentile(boots, alpha / 2.0)
            q_hi = _percentile(boots, 1.0 - alpha / 2.0)
            out[mm] = [float(2.0 * point - q_hi), float(2.0 * point - q_lo)]
        else:
            raise ValueError(f"Unsupported CI method for scalar run metrics: {mm}")
    return out


def _clt_ci(values: Sequence[float], *, cfg: Mapping[str, Any]) -> dict[str, list[float]]:
    vals = [float(x) for x in values]
    if not vals:
        return {}
    mu = float(mean(vals))
    if len(vals) > 1:
        se = float(stdev(vals)) / math.sqrt(float(len(vals)))
    else:
        se = 0.0
    alpha = 1.0 - float(cfg["confidence_level"])
    z = float(NormalDist().inv_cdf(1.0 - alpha / 2.0))
    return {"normal": [float(mu - z * se), float(mu + z * se)]}


def _custom_ci(values: Sequence[float], *, cfg: Mapping[str, Any], point_estimator: str) -> dict[str, list[float]]:
    spec = cfg.get("custom_callable")
    if not spec:
        raise ValueError("strategy=custom requires ci_cfg.custom_callable")
    fn = load_callable(str(spec))
    pfn = _point_fn(point_estimator)
    point = float(pfn([float(x) for x in values])) if values else 0.0
    out = fn(point=point, values=[float(x) for x in values], config=dict(cfg))
    if not isinstance(out, Mapping):
        raise ValueError("custom callable for run metrics must return mapping of method -> [lo, hi]")
    norm: dict[str, list[float]] = {}
    for k, v in out.items():
        if not isinstance(v, (list, tuple)) or len(v) != 2:
            raise ValueError(f"custom CI result must be [lo, hi] pairs, got {k}={v}")
        norm[str(k)] = [float(v[0]), float(v[1])]
    return norm


def _ci_for_values(values: Sequence[float], *, cfg: Mapping[str, Any], point_estimator: str) -> dict[str, list[float]]:
    strat = str(cfg.get("strategy") or "bootstrap")
    if strat == "bootstrap":
        return _bootstrap_ci(values, cfg=cfg, point_estimator=point_estimator)
    if strat == "clt":
        return _clt_ci(values, cfg=cfg)
    if strat == "custom":
        return _custom_ci(values, cfg=cfg, point_estimator=point_estimator)
    raise ValueError(f"Unknown strategy: {strat}")


def compute_scalar_metric_ci(
    values: Sequence[float],
    *,
    ci_cfg: Mapping[str, Any] | None = None,
    point_estimator: str = "median",
) -> dict[str, list[float]]:
    cfg = normalize_ci_config(ci_cfg)
    return _ci_for_values(values, cfg=cfg, point_estimator=point_estimator)


def _load_metric_rows_from_summary(summary: Mapping[str, Any]) -> tuple[list[dict], int]:
    out: list[dict] = []
    n_runs = 0
    for j in list(summary.get("jobs") or []):
        run_dir = Path(str(j.get("run_dir") or "")).expanduser()
        if not run_dir.exists():
            continue
        mp = run_dir / "metrics.json"
        if not mp.exists():
            continue
        n_runs += 1
        obj = json.loads(mp.read_text(encoding="utf-8"))
        base = {
            "task": str(obj.get("task") or ""),
            "set": _require_set_value(obj.get("set")),
            "seed": _require_seed_value(obj.get("seed")),
            "model_id": str(obj.get("model_id") or ""),
            "train_variant": str(obj.get("train_variant") or ""),
            "eval_split": str(obj.get("eval_split") or "test"),
        }
        metrics = obj.get("metrics") or {}
        if not isinstance(metrics, Mapping):
            continue
        for eval_set, mm in metrics.items():
            if not isinstance(mm, Mapping):
                continue
            effective_epoch = None
            for em in list(mm.get("per_epoch") or []):
                if not isinstance(em, Mapping):
                    continue
                try:
                    ep = int(round(float(em.get("epoch"))))
                except Exception:
                    continue
                if ep <= 0:
                    continue
                if effective_epoch is None or ep > effective_epoch:
                    effective_epoch = ep
            row = {**base, "eval_set": str(eval_set), "effective_epoch": effective_epoch}
            for mk in METRIC_KEYS:
                if mk not in mm:
                    raise ValueError(f"{eval_set} missing required metric '{mk}'")
                row[mk] = _require_metric_number(mm[mk], field=f"{eval_set}.{mk}")
            out.append(row)
    return out, n_runs


def build_exp1_report_from_summary(
    summary: Mapping[str, Any],
    *,
    ci_cfg: Mapping[str, Any] | None = None,
    point_estimator: str = "mean",
) -> dict:
    # `mean` returns scalar aggregates. `median` selects a coherent
    # representative run tuple to keep table rows internally consistent.
    rows, n_runs = _load_metric_rows_from_summary(summary)
    cfg = normalize_ci_config(ci_cfg)
    groups: dict[tuple[str, str, str, str, str], list[dict]] = {}
    for r in rows:
        key = (r["task"], r["model_id"], r["train_variant"], r["eval_split"], r["eval_set"])
        groups.setdefault(key, []).append(r)

    pfn = _point_fn(point_estimator)
    out_groups: list[dict] = []
    for key in sorted(groups.keys()):
        task, model_id, train_variant, eval_split, eval_set = key
        items = groups[key]
        if str(point_estimator).strip().lower() == "median":
            # Median is reported as a coherent representative run tuple.
            rep_row = _select_representative_row(items)
            point = {mk: float(rep_row[mk]) for mk in METRIC_KEYS}
            if str(cfg.get("strategy") or "bootstrap") == "bootstrap":
                ci = _bootstrap_representative_row_ci(items, cfg=cfg)
            else:
                ci = {
                    mk: _ci_for_values([float(x[mk]) for x in items], cfg=cfg, point_estimator=point_estimator)
                    for mk in METRIC_KEYS
                }
        else:
            point = {
                mk: float(pfn([float(x[mk]) for x in items]))
                for mk in METRIC_KEYS
            }
            ci = {
                mk: _ci_for_values([float(x[mk]) for x in items], cfg=cfg, point_estimator=point_estimator)
                for mk in METRIC_KEYS
            }
        effective_epochs = sorted({int(v) for v in [x.get("effective_epoch") for x in items] if isinstance(v, int)})
        out_groups.append(
            {
                "task": task,
                "model_id": model_id,
                "train_variant": train_variant,
                "eval_split": eval_split,
                "eval_set": eval_set,
                "effective_epoch": (effective_epochs[0] if len(effective_epochs) == 1 else None),
                "n_runs": len(items),
                "point": point,
                "ci": ci,
            }
        )

    return {
        "n_runs": int(n_runs),
        "n_metric_rows": len(rows),
        "n_groups": len(out_groups),
        "point_estimator": str(point_estimator),
        "ci_config": cfg,
        "groups": out_groups,
    }
