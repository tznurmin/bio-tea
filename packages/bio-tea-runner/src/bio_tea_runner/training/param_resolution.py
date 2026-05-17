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

import fnmatch
from typing import Any, Mapping


DEFAULT_HPARAMS: dict[str, Any] = {
    "epochs": 8,
    "learning_rate": 2e-5,
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 8,
    "report_each_epoch": False,
}


def _as_map(v: Any) -> dict[str, Any]:
    if isinstance(v, Mapping):
        return {str(k): v[k] for k in v.keys()}
    return {}


def _resolve_cast(name: str, value: Any) -> Any:
    if name in {"epochs", "per_device_train_batch_size", "per_device_eval_batch_size"}:
        iv = int(value)
        if iv <= 0:
            raise ValueError(f"{name} must be > 0")
        return iv
    if name == "learning_rate":
        fv = float(value)
        if fv <= 0:
            raise ValueError("learning_rate must be > 0")
        return fv
    if name == "report_each_epoch":
        return bool(value)
    return value


def _first_present(key: str, sources: list[Mapping[str, Any]], default: Any) -> Any:
    for src in sources:
        if key in src:
            return src[key]
    return default


def _model_experiment_cfg(model_cfg: Mapping[str, Any], *, experiment_id: str) -> dict[str, Any]:
    exp_id = str(experiment_id or "exp1")
    out: dict[str, Any] = {}
    exp_block = _as_map(_as_map(model_cfg).get("experiments")).get(exp_id)
    if isinstance(exp_block, Mapping):
        out.update(_as_map(exp_block))
    direct_block = _as_map(model_cfg).get(exp_id)
    if isinstance(direct_block, Mapping):
        out.update(_as_map(direct_block))
    return out


def resolve_hparams_for_model(
    *,
    cfg: Mapping[str, Any],
    experiment_id: str,
    model_cfg: Mapping[str, Any] | None,
) -> dict[str, Any]:
    exp_id = str(experiment_id or "exp1")
    train_cfg = _as_map(cfg.get("training"))
    exp_cfg = _as_map(train_cfg.get(exp_id))
    global_hparams = _as_map(exp_cfg.get("hparams"))
    model_map = _as_map(model_cfg)
    model_hparams = _as_map(model_map.get("hparams"))
    model_exp_cfg = _model_experiment_cfg(model_map, experiment_id=exp_id)
    model_exp_hparams = _as_map(model_exp_cfg.get("hparams"))
    sources: list[Mapping[str, Any]] = [model_exp_hparams, model_hparams, global_hparams]

    out: dict[str, Any] = {}
    for key, default in DEFAULT_HPARAMS.items():
        out[key] = _resolve_cast(key, _first_present(key, sources, default))

    # Additional keys follow precedence: model(exp) -> model(global) -> experiment(global).
    extras: set[str] = set()
    for src in sources:
        extras.update(src.keys())
    for key in sorted(extras):
        if key in out:
            continue
        out[key] = _first_present(key, sources, None)
    return out


def _resolve_variant_hparams_table(
    table: Mapping[str, Any] | None,
    *,
    train_variant: str,
) -> dict[str, Any]:
    tbl = _as_map(table)
    out = _as_map(tbl.get("default"))
    variant = str(train_variant or "").strip()
    if not variant:
        return out

    exact = tbl.get(variant)
    if isinstance(exact, Mapping):
        out.update(_as_map(exact))
        return out

    matches: list[tuple[int, str, dict[str, Any]]] = []
    for raw_key, raw_val in tbl.items():
        key = str(raw_key)
        if key in {"default", variant}:
            continue
        if not isinstance(raw_val, Mapping):
            continue
        if not any(ch in key for ch in "*?[]"):
            continue
        if fnmatch.fnmatch(variant, key):
            matches.append((len(key), key, _as_map(raw_val)))
    if matches:
        matches.sort(key=lambda t: (-t[0], t[1]))
        out.update(matches[0][2])
    return out


def resolve_hparams_for_job(
    *,
    cfg: Mapping[str, Any],
    experiment_id: str,
    model_cfg: Mapping[str, Any] | None,
    train_variant: str,
) -> dict[str, Any]:
    """Resolve hparams with optional train-variant-specific overrides."""

    out = resolve_hparams_for_model(
        cfg=cfg,
        experiment_id=experiment_id,
        model_cfg=model_cfg,
    )

    exp_id = str(experiment_id or "exp1")
    train_cfg = _as_map(cfg.get("training"))
    exp_cfg = _as_map(train_cfg.get(exp_id))
    model_map = _as_map(model_cfg)
    model_exp_cfg = _model_experiment_cfg(model_map, experiment_id=exp_id)

    src_model_exp = _resolve_variant_hparams_table(
        _as_map(model_exp_cfg).get("variant_hparams"),
        train_variant=str(train_variant),
    )
    src_model = _resolve_variant_hparams_table(
        _as_map(model_map).get("variant_hparams"),
        train_variant=str(train_variant),
    )
    src_exp = _resolve_variant_hparams_table(
        _as_map(exp_cfg).get("variant_hparams"),
        train_variant=str(train_variant),
    )
    override_sources: list[Mapping[str, Any]] = [src_model_exp, src_model, src_exp]

    extras: set[str] = set()
    for src in override_sources:
        extras.update(src.keys())
    keys = list(DEFAULT_HPARAMS.keys()) + sorted([k for k in extras if k not in DEFAULT_HPARAMS])
    for key in keys:
        for src in override_sources:
            if key not in src:
                continue
            val = src[key]
            out[key] = _resolve_cast(key, val) if key in DEFAULT_HPARAMS else val
            break
    return out


def resolve_eval_split_for_model(
    *,
    cfg: Mapping[str, Any],
    experiment_id: str,
    model_cfg: Mapping[str, Any] | None,
    default_split: str = "test",
) -> str:
    exp_id = str(experiment_id or "exp1")
    train_cfg = _as_map(cfg.get("training"))
    exp_cfg = _as_map(train_cfg.get(exp_id))
    model_map = _as_map(model_cfg)
    model_exp_cfg = _model_experiment_cfg(model_map, experiment_id=exp_id)
    raw = _first_present(
        "eval_split",
        [model_exp_cfg, model_map, exp_cfg, train_cfg],
        default_split,
    )
    s = str(raw or "").strip()
    return s or str(default_split)


def resolve_eval_sets_for_model(
    *,
    cfg: Mapping[str, Any],
    experiment_id: str,
    model_cfg: Mapping[str, Any] | None,
    default_eval_sets: list[str],
) -> list[str]:
    exp_id = str(experiment_id or "exp1")
    train_cfg = _as_map(cfg.get("training"))
    exp_cfg = _as_map(train_cfg.get(exp_id))
    model_map = _as_map(model_cfg)
    model_exp_cfg = _model_experiment_cfg(model_map, experiment_id=exp_id)
    raw = _first_present(
        "eval_sets",
        [model_exp_cfg, model_map, exp_cfg],
        list(default_eval_sets),
    )
    if isinstance(raw, (list, tuple)):
        out = [str(x) for x in raw if str(x).strip()]
        if out:
            return out
    if raw is None:
        return list(default_eval_sets)
    s = str(raw).strip()
    return [s] if s else list(default_eval_sets)
