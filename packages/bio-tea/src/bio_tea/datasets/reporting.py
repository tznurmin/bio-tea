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

import importlib
import math
import random
from statistics import NormalDist, mean, stdev
from typing import Callable, Mapping, Sequence


DEFAULT_CI_CONFIG = {
    "strategy": "bootstrap",  # bootstrap | clt | custom
    "confidence_level": 0.95,
    "n_bootstrap": 50000,
    "seed": 0,
    "methods": ["percentile"],
    "custom_callable": None,  # "pkg.module:function_name"
}


def load_callable(spec: str) -> Callable:
    s = str(spec or "").strip()
    if ":" not in s:
        raise ValueError("custom callable must use format 'module.submodule:function'")
    mod_name, attr_path = s.split(":", 1)
    mod_name = mod_name.strip()
    attr_path = attr_path.strip()
    if not mod_name or not attr_path:
        raise ValueError("custom callable must use format 'module.submodule:function'")
    mod = importlib.import_module(mod_name)
    obj = mod
    for part in attr_path.split("."):
        obj = getattr(obj, part)
    if not callable(obj):
        raise ValueError(f"Custom callable is not callable: {spec}")
    return obj


def normalize_ci_config(cfg: Mapping[str, object] | None) -> dict:
    out = dict(DEFAULT_CI_CONFIG)
    if isinstance(cfg, Mapping):
        if "strategy" in cfg:
            out["strategy"] = str(cfg.get("strategy") or "bootstrap")
        if "confidence_level" in cfg:
            out["confidence_level"] = float(cfg.get("confidence_level"))
        if "n_bootstrap" in cfg:
            out["n_bootstrap"] = int(cfg.get("n_bootstrap"))
        if "seed" in cfg:
            out["seed"] = int(cfg.get("seed"))
        if "methods" in cfg:
            methods = list(cfg.get("methods") or [])
            out["methods"] = [str(m) for m in methods]
        if "custom_callable" in cfg:
            out["custom_callable"] = cfg.get("custom_callable")

    strategy = str(out.get("strategy") or "bootstrap").strip().lower()
    if strategy not in {"bootstrap", "clt", "custom"}:
        raise ValueError("strategy must be one of: bootstrap, clt, custom")
    cl = float(out["confidence_level"])
    if cl <= 0.0 or cl >= 1.0:
        raise ValueError("confidence_level must be in (0, 1)")
    n_bootstrap = int(out["n_bootstrap"])
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be > 0")
    methods = list(out["methods"] or [])
    if strategy == "bootstrap" and not methods:
        raise ValueError("at least one CI method is required")
    if strategy == "custom" and not out.get("custom_callable"):
        # A callable can still be provided via evaluate_binary_classification(custom_ci_strategy=...),
        # so this is not a hard error here.
        pass
    if strategy != "bootstrap":
        # Keep shape stable; methods are ignored outside bootstrap strategy.
        methods = []
    out["strategy"] = strategy
    out["methods"] = methods
    out["confidence_level"] = cl
    out["n_bootstrap"] = n_bootstrap
    out["seed"] = int(out["seed"])
    return out


def _binary_point_metrics(y_true: Sequence[int], y_pred: Sequence[int], losses: Sequence[float] | None = None) -> dict:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have same length")
    if not y_true:
        raise ValueError("y_true/y_pred must be non-empty")
    if losses is not None and len(losses) != len(y_true):
        raise ValueError("losses must be None or same length as y_true")

    tp = 0
    fp = 0
    fn = 0
    for t, p in zip(y_true, y_pred):
        ti = int(t)
        pi = int(p)
        if pi == 1 and ti == 1:
            tp += 1
        elif pi == 1 and ti != 1:
            fp += 1
        elif pi != 1 and ti == 1:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    out = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }
    if losses is not None:
        out["loss"] = float(mean(float(x) for x in losses))
    return out


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("cannot compute percentile of empty list")
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


def _metric_names(losses: Sequence[float] | None) -> list[str]:
    metrics = ["precision", "recall", "f1"]
    if losses is not None:
        metrics.append("loss")
    return metrics


def _validate_ci_metrics(ci_metrics: Mapping[str, Mapping[str, Sequence[float]]], metrics: Sequence[str]) -> dict[str, dict[str, list[float]]]:
    out: dict[str, dict[str, list[float]]] = {}
    for m in metrics:
        mm = ci_metrics.get(m)
        if not isinstance(mm, Mapping) or not mm:
            raise ValueError(f"Missing CI entries for metric: {m}")
        out[m] = {}
        for method, vals in mm.items():
            if not isinstance(vals, (list, tuple)) or len(vals) != 2:
                raise ValueError(f"CI entry must be a 2-element list/tuple for metric={m}, method={method}")
            lo = float(vals[0])
            hi = float(vals[1])
            out[m][str(method)] = [lo, hi]
    return out


def _ci_from_bootstrap(
    *,
    point_estimate: float,
    bootstrap_values: Sequence[float],
    method: str,
    alpha: float,
    custom_methods: Mapping[str, Callable[[float, Sequence[float], float], tuple[float, float]]] | None,
) -> tuple[float, float]:
    vals = sorted(float(v) for v in bootstrap_values)
    if method == "percentile":
        lo = _percentile(vals, alpha / 2.0)
        hi = _percentile(vals, 1.0 - alpha / 2.0)
        return float(lo), float(hi)
    if method == "basic":
        q_lo = _percentile(vals, alpha / 2.0)
        q_hi = _percentile(vals, 1.0 - alpha / 2.0)
        lo = 2.0 * point_estimate - q_hi
        hi = 2.0 * point_estimate - q_lo
        return float(lo), float(hi)
    if custom_methods and method in custom_methods:
        lo, hi = custom_methods[method](float(point_estimate), vals, float(alpha))
        return float(lo), float(hi)
    raise ValueError(f"Unknown CI method: {method}")


def _bootstrap_ci_metrics(
    *,
    y_true: Sequence[int],
    y_pred: Sequence[int],
    losses: Sequence[float] | None,
    cfg: Mapping[str, object],
    point: Mapping[str, float],
    custom_ci_methods: Mapping[str, Callable[[float, Sequence[float], float], tuple[float, float]]] | None,
) -> dict[str, dict[str, list[float]]]:
    rng = random.Random(int(cfg["seed"]))
    n = len(y_true)
    n_bootstrap = int(cfg["n_bootstrap"])
    alpha = 1.0 - float(cfg["confidence_level"])

    metrics = _metric_names(losses)
    boots: dict[str, list[float]] = {m: [] for m in metrics}
    for _ in range(n_bootstrap):
        idx = [rng.randrange(n) for _ in range(n)]
        bt_true = [int(y_true[i]) for i in idx]
        bt_pred = [int(y_pred[i]) for i in idx]
        bt_losses = [float(losses[i]) for i in idx] if losses is not None else None
        cur = _binary_point_metrics(bt_true, bt_pred, bt_losses)
        for m in metrics:
            boots[m].append(float(cur[m]))

    ci_metrics: dict[str, dict[str, list[float]]] = {}
    for m in metrics:
        point_est = float(point[m])
        ci_metrics[m] = {}
        for method in cfg["methods"]:
            lo, hi = _ci_from_bootstrap(
                point_estimate=point_est,
                bootstrap_values=boots[m],
                method=str(method),
                alpha=alpha,
                custom_methods=custom_ci_methods,
            )
            ci_metrics[m][str(method)] = [float(lo), float(hi)]
    return ci_metrics


def _clt_ci_metrics(
    *,
    y_true: Sequence[int],
    y_pred: Sequence[int],
    losses: Sequence[float] | None,
    cfg: Mapping[str, object],
    point: Mapping[str, float],
) -> dict[str, dict[str, list[float]]]:
    metrics = _metric_names(losses)
    alpha = 1.0 - float(cfg["confidence_level"])
    z = float(NormalDist().inv_cdf(1.0 - alpha / 2.0))
    n = len(y_true)

    n_pred_pos = sum(1 for p in y_pred if int(p) == 1)
    n_true_pos = sum(1 for t in y_true if int(t) == 1)
    p = float(point["precision"])
    r = float(point["recall"])
    f1 = float(point["f1"])

    var_p = (p * (1.0 - p) / n_pred_pos) if n_pred_pos > 0 else 0.0
    var_r = (r * (1.0 - r) / n_true_pos) if n_true_pos > 0 else 0.0
    se_p = math.sqrt(max(var_p, 0.0))
    se_r = math.sqrt(max(var_r, 0.0))

    if (p + r) > 0:
        dfdp = 2.0 * (r**2) / ((p + r) ** 2)
        dfdr = 2.0 * (p**2) / ((p + r) ** 2)
        var_f1 = (dfdp**2) * var_p + (dfdr**2) * var_r
        se_f1 = math.sqrt(max(var_f1, 0.0))
    else:
        se_f1 = 0.0

    def _clamp01(x: float) -> float:
        return max(0.0, min(1.0, float(x)))

    ci_metrics: dict[str, dict[str, list[float]]] = {
        "precision": {"normal": [_clamp01(p - z * se_p), _clamp01(p + z * se_p)]},
        "recall": {"normal": [_clamp01(r - z * se_r), _clamp01(r + z * se_r)]},
        "f1": {"normal": [_clamp01(f1 - z * se_f1), _clamp01(f1 + z * se_f1)]},
    }

    if "loss" in metrics and losses is not None:
        if n > 1:
            se_loss = float(stdev(float(x) for x in losses)) / math.sqrt(float(n))
        else:
            se_loss = 0.0
        mu = float(point["loss"])
        ci_metrics["loss"] = {"normal": [float(mu - z * se_loss), float(mu + z * se_loss)]}

    return ci_metrics


def _custom_ci_metrics(
    *,
    y_true: Sequence[int],
    y_pred: Sequence[int],
    losses: Sequence[float] | None,
    cfg: Mapping[str, object],
    point: Mapping[str, float],
    custom_ci_strategy: Callable | None,
) -> dict[str, dict[str, list[float]]]:
    fn = custom_ci_strategy
    if fn is None:
        spec = cfg.get("custom_callable")
        if not spec:
            raise ValueError("strategy=custom requires custom_ci_strategy or ci_cfg.custom_callable")
        fn = load_callable(str(spec))

    try:
        out = fn(
            point=dict(point),
            y_true=[int(x) for x in y_true],
            y_pred=[int(x) for x in y_pred],
            losses=[float(x) for x in losses] if losses is not None else None,
            config=dict(cfg),
        )
    except TypeError:
        # Fallback for callables that use positional arguments.
        out = fn(dict(point), [int(x) for x in y_true], [int(x) for x in y_pred], losses, dict(cfg))

    if isinstance(out, Mapping) and isinstance(out.get("metrics"), Mapping):
        out = out.get("metrics")
    if not isinstance(out, Mapping):
        raise ValueError("custom CI callable must return a mapping (or {'metrics': mapping})")
    return _validate_ci_metrics(out, _metric_names(losses))


def evaluate_binary_classification(
    *,
    y_true: Sequence[int],
    y_pred: Sequence[int],
    losses: Sequence[float] | None = None,
    ci_cfg: Mapping[str, object] | None = None,
    custom_ci_methods: Mapping[str, Callable[[float, Sequence[float], float], tuple[float, float]]] | None = None,
    custom_ci_strategy: Callable | None = None,
) -> dict:
    cfg = normalize_ci_config(ci_cfg)
    point = _binary_point_metrics(y_true, y_pred, losses)
    strategy = str(cfg.get("strategy") or "bootstrap")
    if strategy == "bootstrap":
        ci_metrics = _bootstrap_ci_metrics(
            y_true=y_true,
            y_pred=y_pred,
            losses=losses,
            cfg=cfg,
            point=point,
            custom_ci_methods=custom_ci_methods,
        )
    elif strategy == "clt":
        ci_metrics = _clt_ci_metrics(
            y_true=y_true,
            y_pred=y_pred,
            losses=losses,
            cfg=cfg,
            point=point,
        )
    elif strategy == "custom":
        ci_metrics = _custom_ci_metrics(
            y_true=y_true,
            y_pred=y_pred,
            losses=losses,
            cfg=cfg,
            point=point,
            custom_ci_strategy=custom_ci_strategy,
        )
    else:
        raise ValueError(f"Unknown CI strategy: {strategy}")

    return {
        "point": point,
        "ci": {
            "config": cfg,
            "metrics": ci_metrics,
        },
    }
