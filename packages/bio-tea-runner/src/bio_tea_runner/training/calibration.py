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
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Mapping


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _csv(v: str | None) -> list[str]:
    if not v:
        return []
    return [x.strip() for x in str(v).split(",") if x.strip()]


def _profiles_root(results_root: Path) -> Path:
    p = results_root / "profiles"
    return p if p.exists() else results_root


def _discover_exp_roots(profiles_root: Path, experiment_id: str) -> list[Path]:
    exp = str(experiment_id or "").strip().strip("/")
    if not exp:
        raise ValueError("experiment_id is required")
    pattern = f"*/{exp}" if "/" in exp else f"*/*/{exp}"
    return sorted([p for p in profiles_root.glob(pattern) if p.is_dir()])


def _profile_and_task(profiles_root: Path, exp_root: Path) -> tuple[str, str]:
    rel = exp_root.relative_to(profiles_root)
    parts = rel.parts
    profile_id = str(parts[0]) if len(parts) >= 1 else ""
    task = str(parts[1]) if len(parts) >= 2 else ""
    return profile_id, task


def _metrics_paths(exp_root: Path) -> list[Path]:
    return sorted(exp_root.glob("runs/*/*/seed_*/*/metrics.json"))


def _epoch_int(v: Any) -> int | None:
    try:
        ep = int(round(float(v)))
    except Exception:
        return None
    if ep <= 0:
        return None
    return ep


def load_per_epoch_rows(
    *,
    results_root: Path,
    experiment_id: str = "exp1",
    tasks: list[str] | None = None,
    model_ids: list[str] | None = None,
    train_variants: list[str] | None = None,
    eval_split: str | None = None,
) -> list[dict[str, Any]]:
    task_filter = set(tasks or [])
    model_filter = set(model_ids or [])
    variant_filter = set(train_variants or [])
    split_filter = str(eval_split or "").strip()
    rows: list[dict[str, Any]] = []
    pr = _profiles_root(Path(results_root))
    for exp_root in _discover_exp_roots(pr, experiment_id):
        profile_id, task = _profile_and_task(pr, exp_root)
        if task_filter and task not in task_filter:
            continue
        for mp in _metrics_paths(exp_root):
            obj = dict(json.loads(mp.read_text(encoding="utf-8")) or {})
            model_id = str(obj.get("model_id") or "")
            if model_filter and model_id not in model_filter:
                continue
            variant = str(obj.get("train_variant") or "")
            if variant_filter and variant not in variant_filter:
                continue
            split = str(obj.get("eval_split") or "test")
            if split_filter and split != split_filter:
                continue
            base = {
                "profile_id": profile_id,
                "task": str(obj.get("task") or task or ""),
                "set": str(obj.get("set") or ""),
                "seed": int(obj.get("seed", 0)),
                "model_id": model_id,
                "train_variant": variant,
                "eval_split": split,
            }
            metrics = obj.get("metrics") or {}
            if not isinstance(metrics, Mapping):
                continue
            for eval_name, mm in metrics.items():
                if not isinstance(mm, Mapping):
                    continue
                for em in list(mm.get("per_epoch") or []):
                    if not isinstance(em, Mapping):
                        continue
                    ep = _epoch_int(em.get("epoch"))
                    if ep is None:
                        continue
                    try:
                        rows.append(
                            {
                                **base,
                                "eval_set": str(eval_name),
                                "epoch": ep,
                                "loss": float(em["loss"]),
                                "precision": float(em["precision"]),
                                "recall": float(em["recall"]),
                                "f1": float(em["f1"]),
                            }
                        )
                    except Exception:
                        continue
    return rows


def _unit_score(
    *,
    by_eval_set: Mapping[str, dict[str, Any]],
    objective: str,
) -> float | None:
    obj = str(objective or "balanced_f1")
    if obj == "augmented_exclusive_f1":
        r = by_eval_set.get("augmented_exclusive")
        return None if r is None else float(r["f1"])
    if obj == "unaugmented_f1":
        r = by_eval_set.get("unaugmented")
        return None if r is None else float(r["f1"])
    if obj == "balanced_loss":
        ra = by_eval_set.get("augmented_exclusive")
        ru = by_eval_set.get("unaugmented")
        if ra is None or ru is None:
            return None
        return float((float(ra["loss"]) + float(ru["loss"])) / 2.0)
    # default balanced_f1
    ra = by_eval_set.get("augmented_exclusive")
    ru = by_eval_set.get("unaugmented")
    if ra is None or ru is None:
        return None
    return float((float(ra["f1"]) + float(ru["f1"])) / 2.0)


def _objective_mode(objective: str) -> str:
    return "min" if str(objective) == "balanced_loss" else "max"


def summarize_calibration_for_model(
    *,
    rows: list[Mapping[str, Any]],
    objective: str,
    min_epoch: int,
    max_epoch: int | None = None,
    exclude_collapsed: bool = False,
    collapse_epsilon: float = 0.0,
) -> dict[str, Any]:
    units_by_epoch: dict[int, dict[tuple[str, str, int, str], dict[str, dict[str, Any]]]] = {}
    for row in rows:
        ep = int(row["epoch"])
        key = (
            str(row.get("task") or ""),
            str(row.get("set") or ""),
            int(row.get("seed", 0)),
            str(row.get("train_variant") or ""),
        )
        units_by_epoch.setdefault(ep, {}).setdefault(key, {})[str(row.get("eval_set") or "")] = dict(row)

    by_epoch: list[dict[str, Any]] = []
    mode = _objective_mode(objective)
    for ep in sorted(units_by_epoch.keys()):
        if ep < int(min_epoch):
            continue
        if max_epoch is not None and ep > int(max_epoch):
            continue
        unit_map = units_by_epoch[ep]
        total = 0
        used = 0
        collapsed = 0
        scores: list[float] = []
        for _, by_eval in unit_map.items():
            s = _unit_score(by_eval_set=by_eval, objective=objective)
            if s is None:
                continue
            total += 1
            if mode == "max" and float(s) <= float(collapse_epsilon):
                collapsed += 1
                if bool(exclude_collapsed):
                    continue
            used += 1
            scores.append(float(s))
        row = {
            "epoch": ep,
            "n_units_total": int(total),
            "n_units_used": int(used),
            "n_collapsed": int(collapsed),
            "collapse_rate": (float(collapsed) / float(total)) if total > 0 else 0.0,
            "score_median": (float(median(scores)) if scores else None),
        }
        by_epoch.append(row)

    best_epoch = None
    best_score = None
    for r in by_epoch:
        sc = r.get("score_median")
        if sc is None:
            continue
        ep = int(r["epoch"])
        if best_epoch is None:
            best_epoch = ep
            best_score = float(sc)
            continue
        if mode == "max":
            if float(sc) > float(best_score) or (float(sc) == float(best_score) and ep < int(best_epoch)):
                best_epoch = ep
                best_score = float(sc)
        else:
            if float(sc) < float(best_score) or (float(sc) == float(best_score) and ep < int(best_epoch)):
                best_epoch = ep
                best_score = float(sc)

    return {
        "objective": str(objective),
        "mode": mode,
        "min_epoch": int(min_epoch),
        "max_epoch": (None if max_epoch is None else int(max_epoch)),
        "exclude_collapsed": bool(exclude_collapsed),
        "collapse_epsilon": float(collapse_epsilon),
        "by_epoch": by_epoch,
        "best_epoch": best_epoch,
        "best_score": best_score,
    }


def apply_epoch_overrides_to_config(
    *,
    cfg: Mapping[str, Any],
    experiment_id: str,
    model_epochs: Mapping[str, int],
) -> dict[str, Any]:
    # Calibration writes one epoch override per model id. Callers that need
    # per-task or per-train-variant epochs must apply that logic outside this
    # helper before collapsing to a single editable config.
    out = deepcopy(dict(cfg))
    tr = dict(out.get("training") or {})
    models = list(tr.get("models") or [])
    exp_id = str(experiment_id)
    model_epochs_map = {str(k): int(v) for k, v in dict(model_epochs).items()}
    out_models: list[dict[str, Any]] = []
    for m in models:
        mm = deepcopy(dict(m))
        mid = str(mm.get("id") or "")
        if mid in model_epochs_map:
            eb = dict(mm.get(exp_id) or {})
            hp = dict(eb.get("hparams") or {})
            hp["epochs"] = int(model_epochs_map[mid])
            eb["hparams"] = hp
            mm[exp_id] = eb
        out_models.append(mm)
    tr["models"] = out_models
    out["training"] = tr
    return out


def write_calibrated_config_and_manifest(
    *,
    source_config_path: Path,
    source_obj: Mapping[str, Any],
    calibrated_obj: Mapping[str, Any],
    output_config_path: Path,
    manifest_path: Path,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    src_hash = _sha256_file(Path(source_config_path))
    ext = output_config_path.suffix.lower()
    if ext in {".yaml", ".yml"}:
        try:
            import yaml
        except Exception as e:  # pragma: no cover
            raise RuntimeError("PyYAML is required to write YAML calibration config") from e
        payload = yaml.safe_dump(dict(calibrated_obj), sort_keys=False)
    else:
        payload = json.dumps(dict(calibrated_obj), indent=2, sort_keys=True) + "\n"
    output_config_path.parent.mkdir(parents=True, exist_ok=True)
    output_config_path.write_text(payload, encoding="utf-8")
    out_hash = _sha256_bytes(payload.encode("utf-8"))
    mf = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_config_path": str(source_config_path),
        "source_config_sha256": src_hash,
        "output_config_path": str(output_config_path),
        "output_config_sha256": out_hash,
        "metadata": dict(metadata or {}),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(mf, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return mf


__all__ = [
    "load_per_epoch_rows",
    "summarize_calibration_for_model",
    "apply_epoch_overrides_to_config",
    "write_calibrated_config_and_manifest",
]
