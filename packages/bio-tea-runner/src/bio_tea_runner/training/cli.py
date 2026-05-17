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

import argparse
import csv
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any

from bio_tea.datasets.reporting import normalize_ci_config

from .run_reporting import build_exp1_report_from_summary, compute_scalar_metric_ci


def _csv(v: str | None) -> list[str]:
    if not v:
        return []
    return [x.strip() for x in str(v).split(",") if x.strip()]


def _csv_int(v: str | None) -> list[int]:
    out: list[int] = []
    for s in _csv(v):
        try:
            out.append(int(s))
        except Exception:
            continue
    return out


def _load_config(path: Path) -> dict:
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except Exception as e:  # pragma: no cover
            raise RuntimeError("PyYAML is required to load YAML config") from e
        return dict(yaml.safe_load(path.read_text(encoding="utf-8")) or {})
    return dict(json.loads(path.read_text(encoding="utf-8")) or {})


def _json_arg(v: str | None) -> dict | None:
    if not v:
        return None
    s = str(v).strip()
    if not s:
        return None
    if s.startswith("{"):
        return dict(json.loads(s))
    p = Path(s)
    if not p.exists():
        raise RuntimeError(f"report-ci path does not exist: {p}")
    return _load_config(p)


def _profiles_root(results_root: Path) -> Path:
    candidate = results_root / "profiles"
    return candidate if candidate.exists() else results_root


def _discover_exp_roots(profiles_root: Path, experiment_id: str) -> list[Path]:
    exp = str(experiment_id or "").strip().strip("/")
    if not exp:
        raise ValueError("experiment_id is required")
    # Common form: task/exp1 ; convenience form: exp1 (all tasks).
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


def _run_dirs(exp_root: Path) -> list[Path]:
    return sorted([p for p in exp_root.glob("runs/*/*/seed_*/*") if p.is_dir()])


def _point_fn(name: str):
    nm = str(name or "median").strip().lower()
    if nm == "mean":
        return lambda xs: float(mean(xs))
    if nm == "median":
        return lambda xs: float(median(xs))
    raise ValueError("point estimator must be one of: mean, median")


def _parse_epoch(v: str | None) -> str | int:
    s = str(v or "final").strip().lower()
    if s in {"", "final"}:
        return "final"
    if s == "all":
        return "all"
    try:
        ep = int(s)
    except Exception as e:  # pragma: no cover
        raise ValueError("--epoch must be 'final', 'all', or a positive integer") from e
    if ep <= 0:
        raise ValueError("--epoch integer must be >= 1")
    return ep


def _as_epoch_int(v: Any) -> int | None:
    try:
        return int(round(float(v)))
    except Exception:
        return None


def _infer_effective_epoch(metrics_row: dict[str, Any]) -> int | None:
    if not isinstance(metrics_row, dict):
        return None
    best: int | None = None
    for em in list(metrics_row.get("per_epoch") or []):
        if not isinstance(em, dict):
            continue
        ep_int = _as_epoch_int(em.get("epoch"))
        if ep_int is None:
            continue
        if best is None or ep_int > best:
            best = ep_int
    return best


def _load_rows_from_run_metrics(exp_root: Path, *, epoch: str | int) -> list[dict]:
    rows: list[dict] = []
    for mp in _metrics_paths(exp_root):
        obj = dict(json.loads(mp.read_text(encoding="utf-8")))
        run_dir = mp.parent
        run_state_path = run_dir / "run_state.json"
        run_state: dict[str, Any] = {}
        if run_state_path.exists():
            try:
                run_state = dict(json.loads(run_state_path.read_text(encoding="utf-8")) or {})
            except Exception:
                run_state = {}
        base = {
            "task": str(obj.get("task") or ""),
            "set": str(obj.get("set") or ""),
            "seed": int(obj.get("seed", 0)),
            "model_id": str(obj.get("model_id") or ""),
            "train_variant": str(obj.get("train_variant") or ""),
            "eval_split": str(obj.get("eval_split") or "test"),
            "run_dir": str(run_dir),
            "run_status": str(run_state.get("status") or ""),
            "run_updated_at": str(run_state.get("updated_at") or ""),
            "run_signature": str(run_state.get("signature") or ""),
        }
        metrics = obj.get("metrics") or {}
        if not isinstance(metrics, dict):
            continue
        for eval_set, mm in metrics.items():
            if not isinstance(mm, dict):
                continue
            effective_epoch = _infer_effective_epoch(mm)

            def _append_from(src: dict, epoch_val: str | int):
                try:
                    rows.append(
                        {
                            **base,
                            "effective_epoch": effective_epoch,
                            "eval_set": str(eval_set),
                            "epoch": epoch_val,
                            "loss": float(src["loss"]),
                            "precision": float(src["precision"]),
                            "recall": float(src["recall"]),
                            "f1": float(src["f1"]),
                        }
                    )
                except Exception:
                    return

            if epoch == "final":
                _append_from(mm, "final")
                continue

            per_epoch = list(mm.get("per_epoch") or [])
            if epoch == "all":
                for em in per_epoch:
                    if not isinstance(em, dict):
                        continue
                    ep_int = _as_epoch_int(em.get("epoch"))
                    if ep_int is None:
                        continue
                    _append_from(em, ep_int)
                continue

            # specific integer epoch
            target = int(epoch)
            hit = None
            for em in per_epoch:
                if not isinstance(em, dict):
                    continue
                ep_int = _as_epoch_int(em.get("epoch"))
                if ep_int == target:
                    hit = em
            if isinstance(hit, dict):
                _append_from(hit, target)
    return rows


def _group_effective_epoch(items: list[dict[str, Any]]) -> int | None:
    vals = sorted({int(v) for v in [x.get("effective_epoch") for x in items] if isinstance(v, int)})
    if len(vals) == 1:
        return int(vals[0])
    return None


def _build_report_from_rows(rows: list[dict], *, point_estimator: str, ci_cfg: dict | None) -> dict:
    pfn = _point_fn(point_estimator)
    cfg = normalize_ci_config(ci_cfg)
    groups: dict[tuple[str, str, str, str, str, Any], list[dict]] = {}
    for r in rows:
        key = (
            str(r.get("task") or ""),
            str(r.get("model_id") or ""),
            str(r.get("train_variant") or ""),
            str(r.get("eval_split") or "test"),
            str(r.get("eval_set") or ""),
            r.get("epoch", "final"),
        )
        groups.setdefault(key, []).append(r)

    out_groups: list[dict] = []
    for key in sorted(groups.keys(), key=lambda k: (k[0], k[1], k[2], k[3], k[4], str(k[5]))):
        task, model_id, train_variant, eval_split, eval_set, epoch = key
        items = groups[key]
        point = {
            mk: float(pfn([float(x[mk]) for x in items]))
            for mk in ("loss", "precision", "recall", "f1")
        }
        ci = {
            mk: compute_scalar_metric_ci(
                [float(x[mk]) for x in items],
                ci_cfg=cfg,
                point_estimator=point_estimator,
            )
            for mk in ("loss", "precision", "recall", "f1")
        }
        out_groups.append(
            {
                "task": task,
                "model_id": model_id,
                "train_variant": train_variant,
                "eval_split": eval_split,
                "eval_set": eval_set,
                "epoch": epoch,
                "effective_epoch": _group_effective_epoch(items),
                "n_runs": len(items),
                "point": point,
                "ci": ci,
            }
        )
    return {
        "n_runs": len({(r["task"], r["set"], r["seed"], r["model_id"], r["train_variant"]) for r in rows}),
        "n_metric_rows": len(rows),
        "n_groups": len(out_groups),
        "groups": out_groups,
    }


def _load_report_or_build(
    *,
    exp_root: Path,
    ci_cfg: dict | None,
    point_estimator: str,
    epoch: str | int,
    force_recompute: bool,
) -> tuple[dict, str]:
    # Optional epoch views always derive from per-run metrics files.
    if epoch != "final":
        rows = _load_rows_from_run_metrics(exp_root, epoch=epoch)
        if rows:
            report = _build_report_from_rows(rows, point_estimator=point_estimator, ci_cfg=ci_cfg)
            return report, f"built_from_live_runs_epoch_{epoch}"
        raise FileNotFoundError(f"Missing per-epoch metrics under {exp_root} for epoch={epoch}")

    # Final metrics are aggregated directly from run metrics so
    # --report-point and --report-ci are applied consistently.
    mps = _metrics_paths(exp_root)
    if mps:
        jobs = [{"run_dir": str(p.parent)} for p in mps]
        summary = {"jobs": jobs}
        report = build_exp1_report_from_summary(summary, ci_cfg=ci_cfg, point_estimator=point_estimator)
        return report, "built_from_live_runs"

    # Optionally read cached report only when recompute is not requested.
    run_report = exp_root / "run_report.json"
    if (not force_recompute) and run_report.exists():
        return dict(json.loads(run_report.read_text(encoding="utf-8"))), str(run_report)

    run_summary = exp_root / "run_summary.json"
    if run_summary.exists():
        summary = dict(json.loads(run_summary.read_text(encoding="utf-8")))
        report = build_exp1_report_from_summary(summary, ci_cfg=ci_cfg, point_estimator=point_estimator)
        return report, "built_from_run_summary"

    raise FileNotFoundError(f"Missing run report and run summary under {exp_root}")


def _filter_group(group: dict, *, model_ids: set[str], variants: set[str], eval_sets: set[str]) -> bool:
    if model_ids and str(group.get("model_id") or "") not in model_ids:
        return False
    if variants and str(group.get("train_variant") or "") not in variants:
        return False
    if eval_sets and str(group.get("eval_set") or "") not in eval_sets:
        return False
    return True


def _metric_scale_value(metric_name: str, value: float, *, scale: str) -> float:
    if str(scale) == "percent" and metric_name in {"precision", "recall", "f1"}:
        return float(value) * 100.0
    return float(value)


def _ci_bounds_for_metric(group: dict[str, Any], metric_name: str, *, ci_method: str | None) -> tuple[float, float] | None:
    ci = dict((group.get("ci") or {}).get(metric_name) or {})
    if not ci:
        return None
    method = str(ci_method or "").strip()
    if method:
        pair = ci.get(method)
        if isinstance(pair, (list, tuple)) and len(pair) == 2:
            return (float(pair[0]), float(pair[1]))
        return None
    for pair in ci.values():
        if isinstance(pair, (list, tuple)) and len(pair) == 2:
            return (float(pair[0]), float(pair[1]))
    return None


def _format_metric_value(
    group: dict[str, Any],
    metric_name: str,
    *,
    show_ci: bool,
    ci_method: str | None,
    scale: str,
) -> str:
    point = float(dict(group.get("point") or {}).get(metric_name, 0.0))
    point = _metric_scale_value(metric_name, point, scale=scale)
    if not show_ci:
        return f"{point:.6f}"
    pair = _ci_bounds_for_metric(group, metric_name, ci_method=ci_method)
    if pair is None:
        return f"{point:.6f}"
    lo = _metric_scale_value(metric_name, float(pair[0]), scale=scale)
    hi = _metric_scale_value(metric_name, float(pair[1]), scale=scale)
    return f"{point:.6f} ({lo:.6f}/{hi:.6f})"


def _display_epoch(group: dict[str, Any], *, mode: str) -> str | int:
    if str(mode) == "effective":
        effective_epoch = group.get("effective_epoch")
        if isinstance(effective_epoch, int):
            return effective_epoch
    return group.get("epoch", "final")


def _print_table(
    groups: list[dict],
    *,
    show_ci: bool,
    ci_method: str | None,
    epoch_display: str,
    metric_scale: str,
) -> None:
    if not groups:
        print("No matching groups.")
        return
    hdr = "profile task model variant eval_split eval_set epoch n_runs loss precision recall f1"
    print(hdr)
    for g in groups:
        ep = _display_epoch(g, mode=epoch_display)
        print(
            f"{g['profile_id']} {g['task']} {g['model_id']} {g['train_variant']} "
            f"{g.get('eval_split', 'test')} {g['eval_set']} "
            f"{ep} "
            f"{int(g.get('n_runs', 0))} "
            f"{_format_metric_value(g, 'loss', show_ci=show_ci, ci_method=ci_method, scale=metric_scale)} "
            f"{_format_metric_value(g, 'precision', show_ci=show_ci, ci_method=ci_method, scale=metric_scale)} "
            f"{_format_metric_value(g, 'recall', show_ci=show_ci, ci_method=ci_method, scale=metric_scale)} "
            f"{_format_metric_value(g, 'f1', show_ci=show_ci, ci_method=ci_method, scale=metric_scale)}"
        )


def _group_csv_row(
    group: dict[str, Any],
    *,
    ci_method: str | None,
    epoch_display: str,
    metric_scale: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "profile_id": str(group.get("profile_id") or ""),
        "task": str(group.get("task") or ""),
        "model_id": str(group.get("model_id") or ""),
        "train_variant": str(group.get("train_variant") or ""),
        "eval_split": str(group.get("eval_split") or "test"),
        "eval_set": str(group.get("eval_set") or ""),
        "epoch": _display_epoch(group, mode=epoch_display),
        "n_runs": int(group.get("n_runs", 0) or 0),
    }
    for metric_name in ("loss", "precision", "recall", "f1"):
        point = float(dict(group.get("point") or {}).get(metric_name, 0.0))
        row[metric_name] = _metric_scale_value(metric_name, point, scale=metric_scale)
        pair = _ci_bounds_for_metric(group, metric_name, ci_method=ci_method)
        if pair is None:
            row[f"{metric_name}_ci_low"] = ""
            row[f"{metric_name}_ci_high"] = ""
        else:
            row[f"{metric_name}_ci_low"] = _metric_scale_value(metric_name, float(pair[0]), scale=metric_scale)
            row[f"{metric_name}_ci_high"] = _metric_scale_value(metric_name, float(pair[1]), scale=metric_scale)
    return row


def _run_csv_row(row: dict[str, Any], *, metric_scale: str) -> dict[str, Any]:
    out: dict[str, Any] = {
        "profile_id": str(row.get("profile_id") or ""),
        "task": str(row.get("task") or ""),
        "model_id": str(row.get("model_id") or ""),
        "train_variant": str(row.get("train_variant") or ""),
        "set": str(row.get("set") or ""),
        "seed": _safe_int(row.get("seed")),
        "eval_split": str(row.get("eval_split") or "test"),
        "eval_set": str(row.get("eval_set") or ""),
        "epoch": row.get("epoch", "final"),
        "effective_epoch": row.get("effective_epoch", ""),
        "run_status": str(row.get("run_status") or ""),
        "run_updated_at": str(row.get("run_updated_at") or ""),
    }
    for metric_name in ("loss", "precision", "recall", "f1"):
        val = float(row.get(metric_name, 0.0) or 0.0)
        out[metric_name] = _metric_scale_value(metric_name, val, scale=metric_scale)
    return out


def _write_csv(
    path: Path,
    *,
    groups: list[dict[str, Any]],
    run_rows: list[dict[str, Any]],
    include_runs: bool,
    ci_method: str | None,
    epoch_display: str,
    metric_scale: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if include_runs:
        fieldnames = [
            "profile_id",
            "task",
            "model_id",
            "train_variant",
            "set",
            "seed",
            "eval_split",
            "eval_set",
            "epoch",
            "effective_epoch",
            "run_status",
            "run_updated_at",
            "loss",
            "precision",
            "recall",
            "f1",
        ]
        rows = [_run_csv_row(r, metric_scale=metric_scale) for r in run_rows]
    else:
        fieldnames = [
            "profile_id",
            "task",
            "model_id",
            "train_variant",
            "eval_split",
            "eval_set",
            "epoch",
            "n_runs",
            "loss",
            "loss_ci_low",
            "loss_ci_high",
            "precision",
            "precision_ci_low",
            "precision_ci_high",
            "recall",
            "recall_ci_low",
            "recall_ci_high",
            "f1",
            "f1_ci_low",
            "f1_ci_high",
        ]
        rows = [
            _group_csv_row(
                g,
                ci_method=ci_method,
                epoch_display=epoch_display,
                metric_scale=metric_scale,
            )
            for g in groups
        ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _profile_label(profile_id: str, *, display: str) -> str:
    s = str(profile_id or "")
    if str(display) == "full":
        return s
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:8]


def _format_time(v: Any, *, display: str) -> str:
    s = str(v or "").strip()
    if not s:
        return ""
    if str(display) == "iso":
        return s
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return f"{dt.hour:02d}:{dt.minute:02d} {dt.day}/{dt.month}/{str(dt.year)[2:]}"
    except Exception:
        return s


def _time_sort_key(v: Any) -> tuple[int, str]:
    s = str(v or "").strip()
    if not s:
        return (1, "")
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return (0, dt.isoformat())
    except Exception:
        return (1, s)


def _epoch_sort_key(v: Any) -> tuple[int, int | str]:
    s = str(v or "").strip()
    if not s:
        return (2, "")
    if s.lower() == "final":
        return (1, "final")
    try:
        return (0, int(round(float(s))))
    except Exception:
        return (2, s)


def _run_row_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        _time_sort_key(row.get("run_updated_at")),
        str(row.get("profile_id") or ""),
        str(row.get("task") or ""),
        str(row.get("model_id") or ""),
        str(row.get("train_variant") or ""),
        str(row.get("set") or ""),
        int(_safe_int(row.get("seed")) or 0),
        str(row.get("eval_split") or ""),
        str(row.get("eval_set") or ""),
        _epoch_sort_key(row.get("epoch")),
    )


def _parse_time_bound(value: str | None, *, end_inclusive: bool) -> datetime | None:
    s = str(value or "").strip()
    if not s:
        return None
    if "T" in s or " " in s:
        try:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except Exception as e:
            raise ValueError(f"Invalid datetime bound: {value}") from e
    try:
        d = datetime.strptime(s, "%Y-%m-%d")
    except Exception as e:
        raise ValueError(f"Invalid date bound (expected YYYY-MM-DD or ISO datetime): {value}") from e
    if end_inclusive:
        d = d + timedelta(days=1)
    return d.replace(tzinfo=timezone.utc)


def _parse_row_updated_at(row: dict[str, Any]) -> datetime | None:
    s = str(row.get("run_updated_at") or "").strip()
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def _run_row_in_time_range(row: dict[str, Any], *, since: datetime | None, until: datetime | None) -> bool:
    ts = _parse_row_updated_at(row)
    if ts is None:
        return False
    if since is not None and ts < since:
        return False
    if until is not None and ts >= until:
        return False
    return True


def _aggregate_groups_from_run_rows(
    rows: list[dict],
    *,
    point_estimator: str,
    ci_cfg: dict | None,
) -> list[dict[str, Any]]:
    pfn = _point_fn(point_estimator)
    cfg = normalize_ci_config(ci_cfg)
    groups: dict[tuple[str, str, str, str, str, str, Any], list[dict]] = {}
    for r in rows:
        key = (
            str(r.get("profile_id") or ""),
            str(r.get("task") or ""),
            str(r.get("model_id") or ""),
            str(r.get("train_variant") or ""),
            str(r.get("eval_split") or "test"),
            str(r.get("eval_set") or ""),
            r.get("epoch", "final"),
        )
        groups.setdefault(key, []).append(r)

    out: list[dict[str, Any]] = []
    for key in sorted(groups.keys(), key=lambda x: (x[0], x[1], x[2], x[3], x[4], x[5], str(x[6]))):
        profile_id, task, model_id, train_variant, eval_split, eval_set, epoch = key
        items = groups[key]
        point = {
            mk: float(pfn([float(x.get(mk, 0.0)) for x in items]))
            for mk in ("loss", "precision", "recall", "f1")
        }
        ci = {
            mk: compute_scalar_metric_ci(
                [float(x.get(mk, 0.0)) for x in items],
                ci_cfg=cfg,
                point_estimator=point_estimator,
            )
            for mk in ("loss", "precision", "recall", "f1")
        }
        out.append(
            {
                "profile_id": profile_id,
                "task": task,
                "model_id": model_id,
                "train_variant": train_variant,
                "eval_split": eval_split,
                "eval_set": eval_set,
                "epoch": epoch,
                "effective_epoch": _group_effective_epoch(items),
                "n_runs": len(items),
                "point": point,
                "ci": ci,
            }
        )
    return out


def _print_run_rows_table(
    rows: list[dict],
    *,
    profile_display: str,
    time_display: str,
    show_status: bool,
) -> None:
    if not rows:
        print("No matching run rows.")
        return
    hdr = "profile task model variant set seed eval_split eval_set epoch loss precision recall f1 time"
    if bool(show_status):
        hdr += " status"
    print(hdr)
    for r in rows:
        ep = r.get("epoch", "final")
        prof = _profile_label(str(r.get("profile_id") or ""), display=profile_display)
        tstamp = _format_time(r.get("run_updated_at"), display=time_display)
        status = str(r.get("run_status") or "")
        line = (
            f"{prof} {r.get('task', '')} {r.get('model_id', '')} "
            f"{r.get('train_variant', '')} {r.get('set', '')} {int(r.get('seed', 0))} "
            f"{r.get('eval_split', 'test')} {r.get('eval_set', '')} {ep} "
            f"{float(r.get('loss', 0.0)):.6f} "
            f"{float(r.get('precision', 0.0)):.6f} "
            f"{float(r.get('recall', 0.0)):.6f} "
            f"{float(r.get('f1', 0.0)):.6f} "
            f"{tstamp}"
        )
        if bool(show_status):
            line += f" {status}"
        print(
            line
        )


def _report_row_matches_filters(row: dict, *, model_ids: set[str], variants: set[str], eval_sets: set[str]) -> bool:
    if model_ids and str(row.get("model_id") or "") not in model_ids:
        return False
    if variants and str(row.get("train_variant") or "") not in variants:
        return False
    if eval_sets and str(row.get("eval_set") or "") not in eval_sets:
        return False
    return True


def _build_paired_delta_bundle(
    rows: list[dict],
    *,
    variant_a: str,
    variant_b: str,
    point_estimator: str,
) -> dict[str, Any]:
    va = str(variant_a or "").strip()
    vb = str(variant_b or "").strip()
    if not va or not vb:
        return {
            "enabled": True,
            "variant_a": va,
            "variant_b": vb,
            "n_pairs": 0,
            "n_groups": 0,
            "groups": [],
        }

    buckets: dict[tuple[str, str, int, str, str, Any, str, str], dict[str, dict]] = {}
    for r in rows:
        seed = _safe_int(r.get("seed"))
        if seed is None:
            continue
        key = (
            str(r.get("profile_id") or ""),
            str(r.get("task") or ""),
            int(seed),
            str(r.get("set") or ""),
            str(r.get("model_id") or ""),
            r.get("epoch", "final"),
            str(r.get("eval_split") or "test"),
            str(r.get("eval_set") or ""),
        )
        buckets.setdefault(key, {})[str(r.get("train_variant") or "")] = r

    pairs: list[dict[str, Any]] = []
    for key in sorted(buckets.keys(), key=lambda x: (x[0], x[1], x[4], x[3], x[2], str(x[5]), x[6], x[7])):
        variants = buckets[key]
        if va not in variants or vb not in variants:
            continue
        ra = variants[va]
        rb = variants[vb]
        point: dict[str, float] = {}
        ok = True
        for mk in ("loss", "precision", "recall", "f1"):
            a = _safe_float(ra.get(mk))
            b = _safe_float(rb.get(mk))
            if a is None or b is None:
                ok = False
                break
            point[mk] = float(a) - float(b)
        if not ok:
            continue
        profile_id, task, seed, set_name, model_id, epoch, eval_split, eval_set = key
        pairs.append(
            {
                "profile_id": profile_id,
                "task": task,
                "set": set_name,
                "seed": int(seed),
                "model_id": model_id,
                "eval_split": eval_split,
                "eval_set": eval_set,
                "epoch": epoch,
                "pair": [va, vb],
                "delta": point,
            }
        )

    pfn = _point_fn(point_estimator)
    groups: dict[tuple[str, str, str, Any, str, str], list[dict[str, Any]]] = {}
    for row in pairs:
        key = (
            str(row.get("profile_id") or ""),
            str(row.get("task") or ""),
            str(row.get("model_id") or ""),
            row.get("epoch", "final"),
            str(row.get("eval_split") or "test"),
            str(row.get("eval_set") or ""),
        )
        groups.setdefault(key, []).append(row)

    out_groups: list[dict[str, Any]] = []
    for key in sorted(groups.keys(), key=lambda x: (x[0], x[1], x[2], str(x[3]), x[4], x[5])):
        profile_id, task, model_id, epoch, eval_split, eval_set = key
        items = groups[key]
        point = {
            mk: float(pfn([float(x["delta"][mk]) for x in items]))
            for mk in ("loss", "precision", "recall", "f1")
        }
        out_groups.append(
            {
                "profile_id": profile_id,
                "task": task,
                "model_id": model_id,
                "epoch": epoch,
                "eval_split": eval_split,
                "eval_set": eval_set,
                "n_pairs": len(items),
                "point": point,
            }
        )

    return {
        "enabled": True,
        "variant_a": va,
        "variant_b": vb,
        "n_pairs": len(pairs),
        "n_groups": len(out_groups),
        "groups": out_groups,
    }


def _sha256_obj(obj: object) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_json_file(path: Path) -> tuple[dict | None, str | None]:
    if not path.exists():
        return None, f"missing_file:{path.name}"
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None, f"invalid_json:{path.name}"
    if not isinstance(obj, dict):
        return None, f"invalid_json_type:{path.name}"
    return dict(obj), None


def _safe_float(v: Any) -> float | None:
    try:
        return float(v)
    except Exception:
        return None


def _safe_int(v: Any) -> int | None:
    try:
        return int(v)
    except Exception:
        return None


def _extract_run_identity(*, metrics: dict | None, run_spec: dict | None) -> dict[str, Any]:
    spec_job = dict((((run_spec or {}).get("payload") or {}).get("job") or {}))
    task = str((metrics or {}).get("task") or spec_job.get("task") or "")
    set_name = str((metrics or {}).get("set") or spec_job.get("set") or "")
    model_id = str((metrics or {}).get("model_id") or spec_job.get("model_id") or "")
    variant = str((metrics or {}).get("train_variant") or spec_job.get("train_variant") or "")
    seed = _safe_int((metrics or {}).get("seed"))
    if seed is None:
        seed = _safe_int(spec_job.get("seed"))
    return {
        "task": task,
        "set": set_name,
        "seed": seed,
        "model_id": model_id,
        "train_variant": variant,
    }


def _resolve_manifest_path(raw_path: Any, run_dir: Path) -> Path:
    p = Path(str(raw_path)).expanduser()
    if p.is_absolute():
        return p
    return (run_dir / p).resolve()


def _validate_model_artifacts(*, run_dir: Path, metrics: dict | None) -> list[str]:
    errors: list[str] = []
    metrics_obj = dict((metrics or {}).get("model_artifacts") or {})
    manifest_path_raw = metrics_obj.get("manifest_path")
    if not manifest_path_raw:
        manifest_path = run_dir / "model_artifacts.json"
    else:
        manifest_path = _resolve_manifest_path(manifest_path_raw, run_dir)

    manifest, err = _load_json_file(manifest_path)
    if err is not None:
        errors.append(f"{err}:{manifest_path}")
        return errors

    files = list((manifest or {}).get("files") or [])
    n_files = _safe_int((manifest or {}).get("n_files"))
    if n_files is None or n_files != len(files):
        errors.append(
            "model_artifacts_n_files_mismatch:"
            f"manifest_n_files={manifest.get('n_files')}:actual={len(files)}"
        )
    total_bytes = _safe_int((manifest or {}).get("total_bytes"))
    computed_total = 0
    for row in files:
        if not isinstance(row, dict):
            continue
        sz = _safe_int(row.get("size"))
        if sz is None:
            continue
        computed_total += int(sz)
    if total_bytes is None or total_bytes != computed_total:
        errors.append(
            "model_artifacts_total_bytes_mismatch:"
            f"manifest_total_bytes={manifest.get('total_bytes')}:actual={computed_total}"
        )

    if metrics_obj:
        for k in ("saved", "mode", "n_files", "total_bytes"):
            if k in metrics_obj and metrics_obj.get(k) != manifest.get(k):
                errors.append(
                    "model_artifacts_summary_mismatch:"
                    f"{k}:metrics={metrics_obj.get(k)}:manifest={manifest.get(k)}"
                )
    return errors


def validate_exp1_run_dir(*, run_dir: Path, collapse_epsilon: float = 0.0) -> dict[str, Any]:
    errors: list[str] = []
    collapsed_eval_sets: list[dict[str, Any]] = []

    run_spec, err = _load_json_file(run_dir / "run_spec.json")
    if err is not None:
        errors.append(err)
    run_state, err = _load_json_file(run_dir / "run_state.json")
    if err is not None:
        errors.append(err)
    metrics, err = _load_json_file(run_dir / "metrics.json")
    if err is not None:
        errors.append(err)

    ident = _extract_run_identity(metrics=metrics, run_spec=run_spec)

    spec_sig = str((run_spec or {}).get("signature") or "")
    state_sig = str((run_state or {}).get("signature") or "")
    if spec_sig and state_sig and spec_sig != state_sig:
        errors.append(f"signature_mismatch:run_spec={spec_sig}:run_state={state_sig}")

    state_status = str((run_state or {}).get("status") or "")
    if state_status and state_status != "completed":
        errors.append(f"unexpected_run_state_status:{state_status}")

    metrics_expected_sha = str((run_state or {}).get("metrics_sha256") or "")
    if metrics_expected_sha and isinstance(metrics, dict):
        observed_sha = _sha256_obj(metrics)
        if observed_sha != metrics_expected_sha:
            errors.append(
                "metrics_sha256_mismatch:"
                f"run_state={metrics_expected_sha}:observed={observed_sha}"
            )

    spec_job = dict((((run_spec or {}).get("payload") or {}).get("job") or {}))
    if spec_job and isinstance(metrics, dict):
        for k in ("task", "set", "model_id", "train_variant"):
            spec_v = str(spec_job.get(k) or "")
            met_v = str(metrics.get(k) or "")
            if spec_v and met_v and spec_v != met_v:
                errors.append(f"job_field_mismatch:{k}:run_spec={spec_v}:metrics={met_v}")
        spec_seed = _safe_int(spec_job.get("seed"))
        met_seed = _safe_int(metrics.get("seed"))
        if spec_seed is not None and met_seed is not None and spec_seed != met_seed:
            errors.append(f"job_field_mismatch:seed:run_spec={spec_seed}:metrics={met_seed}")

    metrics_rows = (metrics or {}).get("metrics")
    if not isinstance(metrics_rows, dict):
        if isinstance(metrics, dict):
            errors.append("invalid_metrics_payload:metrics_missing_or_not_object")
        metrics_rows = {}

    expected_eval_sets = [str(x) for x in list(spec_job.get("eval_sets") or []) if str(x).strip()]
    for eval_name in expected_eval_sets:
        if eval_name not in metrics_rows:
            errors.append(f"missing_eval_set:{eval_name}")

    required = ("loss", "precision", "recall", "f1")
    for eval_name, row in metrics_rows.items():
        if not isinstance(row, dict):
            errors.append(f"invalid_eval_metrics_payload:{eval_name}")
            continue
        vals: dict[str, float] = {}
        for mk in required:
            if mk not in row:
                errors.append(f"missing_metric:{eval_name}:{mk}")
                continue
            fv = _safe_float(row.get(mk))
            if fv is None:
                errors.append(f"non_numeric_metric:{eval_name}:{mk}:{row.get(mk)}")
                continue
            vals[mk] = fv
        if len(vals) != len(required):
            continue
        if (
            abs(float(vals["precision"])) <= float(collapse_epsilon)
            and abs(float(vals["recall"])) <= float(collapse_epsilon)
            and abs(float(vals["f1"])) <= float(collapse_epsilon)
        ):
            collapsed_eval_sets.append(
                {
                    "eval_set": str(eval_name),
                    "precision": float(vals["precision"]),
                    "recall": float(vals["recall"]),
                    "f1": float(vals["f1"]),
                }
            )

    if isinstance(metrics, dict):
        errors.extend(_validate_model_artifacts(run_dir=run_dir, metrics=metrics))

    return {
        "run_dir": str(run_dir),
        **ident,
        "errors": errors,
        "collapsed_eval_sets": collapsed_eval_sets,
        "ok": (len(errors) == 0),
    }


def _run_matches_filters(
    row: dict[str, Any],
    *,
    model_ids: set[str],
    train_variants: set[str],
    sets: set[str],
    seeds: set[int],
) -> bool:
    if model_ids and str(row.get("model_id") or "") not in model_ids:
        return False
    if train_variants and str(row.get("train_variant") or "") not in train_variants:
        return False
    if sets and str(row.get("set") or "") not in sets:
        return False
    seed = _safe_int(row.get("seed"))
    if seeds and seed not in seeds:
        return False
    return True


def exp1_report_main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Aggregate Exp1 run reports by experiment ID across model profiles.",
    )
    ap.add_argument("--results-root", default="results", type=str)
    ap.add_argument(
        "--experiment-id",
        required=True,
        type=str,
        help="Experiment path under profile root, for example 'pathogens/exp1' or 'exp1'",
    )
    ap.add_argument("--profile-ids", default=None, type=str, help="Optional comma-separated profile id filter")
    ap.add_argument("--tasks", default=None, type=str, help="Optional comma-separated task filter")
    ap.add_argument("--model-ids", default=None, type=str, help="Optional comma-separated model id filter")
    ap.add_argument("--train-variants", default=None, type=str, help="Optional comma-separated train variant filter")
    ap.add_argument("--eval-sets", default=None, type=str, help="Optional comma-separated eval set filter")
    ap.add_argument(
        "--epoch",
        default="final",
        type=str,
        help="Which metrics to aggregate: final | all | <epoch-int> (for per-epoch metrics)",
    )
    ap.add_argument("--report-ci", default=None, type=str, help="Optional CI config for fallback summary build")
    ap.add_argument(
        "--report-point",
        default="median",
        choices=["mean", "median"],
        type=str,
        help="Point estimator for report aggregation",
    )
    ap.add_argument(
        "--force-recompute",
        action="store_true",
        help="Ignore cached run_report.json and rebuild from run metrics/summary",
    )
    ap.add_argument(
        "--paired-deltas",
        action="store_true",
        help="Include optional paired all-vs-none deltas matched by profile/task/set/seed/model/eval/epoch",
    )
    ap.add_argument("--paired-variant-a", default="all", type=str, help="Variant A for paired delta (A-B)")
    ap.add_argument("--paired-variant-b", default="none", type=str, help="Variant B for paired delta (A-B)")
    ap.add_argument(
        "--since",
        default=None,
        type=str,
        help="Optional lower time bound (inclusive) for run rows: YYYY-MM-DD or ISO datetime",
    )
    ap.add_argument(
        "--until",
        default=None,
        type=str,
        help="Optional upper time bound (inclusive for date form) for run rows: YYYY-MM-DD or ISO datetime",
    )
    ap.add_argument("--output", default=None, type=str, help="Optional output JSON path")
    ap.add_argument("--csv-output", default=None, type=str, help="Optional output CSV path")
    ap.add_argument("--table", action="store_true", help="Print concise table to stdout")
    ap.add_argument("--show-ci", action="store_true", help="Include confidence intervals in --table output")
    ap.add_argument(
        "--table-ci-method",
        default=None,
        type=str,
        help="Optional CI method to display in --table (defaults to first available method)",
    )
    ap.add_argument(
        "--table-epoch-display",
        default="report",
        choices=["report", "effective"],
        type=str,
        help="Epoch value to display in --table: report label or effective final epoch when available",
    )
    ap.add_argument(
        "--metric-scale",
        default="fraction",
        choices=["fraction", "percent"],
        type=str,
        help="Display precision/recall/F1 as fractions or percentages in --table",
    )
    ap.add_argument(
        "--table-runs",
        action="store_true",
        help="Print per-run rows (includes set/seed/time) instead of aggregated groups",
    )
    ap.add_argument(
        "--profile-display",
        default="short",
        choices=["short", "full"],
        type=str,
        help="Profile ID display mode for --table-runs: short hash or full identifier",
    )
    ap.add_argument(
        "--time-display",
        default="compact",
        choices=["compact", "iso"],
        type=str,
        help="Timestamp display mode for --table-runs",
    )
    ap.add_argument(
        "--show-status",
        action="store_true",
        help="Include run status column in --table-runs output",
    )
    args = ap.parse_args(argv)

    results_root = Path(args.results_root)
    profiles_root = _profiles_root(results_root)
    ci_cfg = _json_arg(args.report_ci)
    epoch = _parse_epoch(args.epoch)

    profile_filter = set(_csv(args.profile_ids))
    task_filter = set(_csv(args.tasks))
    model_filter = set(_csv(args.model_ids))
    variant_filter = set(_csv(args.train_variants))
    eval_filter = set(_csv(args.eval_sets))
    since_bound = _parse_time_bound(args.since, end_inclusive=False)
    until_bound = _parse_time_bound(args.until, end_inclusive=True)
    if since_bound is not None and until_bound is not None and since_bound >= until_bound:
        raise ValueError("--since must be earlier than --until")

    exp_roots = _discover_exp_roots(profiles_root, args.experiment_id)
    experiments: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    errors: list[str] = []

    for exp_root in exp_roots:
        profile_id, task = _profile_and_task(profiles_root, exp_root)
        if profile_filter and profile_id not in profile_filter:
            continue
        if task_filter and task not in task_filter:
            continue
        try:
            report, source = _load_report_or_build(
                exp_root=exp_root,
                ci_cfg=ci_cfg,
                point_estimator=str(args.report_point),
                epoch=epoch,
                force_recompute=bool(args.force_recompute),
            )
        except FileNotFoundError as e:
            errors.append(str(e))
            continue

        report_groups = list(report.get("groups") or [])
        kept = 0
        for g in report_groups:
            if not isinstance(g, dict):
                continue
            if not _filter_group(
                g,
                model_ids=model_filter,
                variants=variant_filter,
                eval_sets=eval_filter,
            ):
                continue
            row = dict(g)
            row["profile_id"] = profile_id
            row["task"] = task or str(g.get("task") or "")
            row["experiment_id"] = str(args.experiment_id)
            row["exp_root"] = str(exp_root)
            if "epoch" not in row:
                row["epoch"] = "final"
            groups.append(row)
            kept += 1

        for r0 in _load_rows_from_run_metrics(exp_root, epoch=epoch):
            if not _report_row_matches_filters(
                r0,
                model_ids=model_filter,
                variants=variant_filter,
                eval_sets=eval_filter,
            ):
                continue
            rr = dict(r0)
            rr["profile_id"] = profile_id
            rr["task"] = task or str(r0.get("task") or "")
            rr["experiment_id"] = str(args.experiment_id)
            rr["exp_root"] = str(exp_root)
            if "epoch" not in rr:
                rr["epoch"] = "final"
            run_rows.append(rr)

        if bool(args.paired_deltas):
            for r in _load_rows_from_run_metrics(exp_root, epoch=epoch):
                if not _report_row_matches_filters(
                    r,
                    model_ids=model_filter,
                    variants=variant_filter,
                    eval_sets=eval_filter,
                ):
                    continue
                rr = dict(r)
                rr["profile_id"] = profile_id
                rr["task"] = task or str(r.get("task") or "")
                rr["experiment_id"] = str(args.experiment_id)
                rr["exp_root"] = str(exp_root)
                if "epoch" not in rr:
                    rr["epoch"] = "final"
                paired_rows.append(rr)

        experiments.append(
            {
                "profile_id": profile_id,
                "task": task,
                "exp_root": str(exp_root),
                "report_source": source,
                "n_groups": int(kept),
            }
        )

    run_rows = sorted(run_rows, key=_run_row_sort_key)
    paired_rows = sorted(paired_rows, key=_run_row_sort_key)

    if since_bound is not None or until_bound is not None:
        run_rows = [
            r for r in run_rows
            if _run_row_in_time_range(r, since=since_bound, until=until_bound)
        ]
        paired_rows = [
            r for r in paired_rows
            if _run_row_in_time_range(r, since=since_bound, until=until_bound)
        ]
        groups = _aggregate_groups_from_run_rows(
            run_rows,
            point_estimator=str(args.report_point),
            ci_cfg=ci_cfg,
        )

    paired_obj = {
        "enabled": False,
        "variant_a": str(args.paired_variant_a),
        "variant_b": str(args.paired_variant_b),
        "n_pairs": 0,
        "n_groups": 0,
        "groups": [],
    }
    if bool(args.paired_deltas):
        paired_obj = _build_paired_delta_bundle(
            paired_rows,
            variant_a=str(args.paired_variant_a),
            variant_b=str(args.paired_variant_b),
            point_estimator=str(args.report_point),
        )

    obj = {
        "results_root": str(results_root),
        "profiles_root": str(profiles_root),
        "experiment_id": str(args.experiment_id),
        "filters": {
            "profile_ids": sorted(profile_filter),
            "tasks": sorted(task_filter),
            "model_ids": sorted(model_filter),
            "train_variants": sorted(variant_filter),
            "eval_sets": sorted(eval_filter),
            "epoch": epoch,
            "since": str(args.since or ""),
            "until": str(args.until or ""),
        },
        "n_experiments": len(experiments),
        "n_groups": len(groups),
        "n_run_rows": len(run_rows),
        "errors": errors,
        "experiments": experiments,
        "groups": groups,
        "run_rows": run_rows,
        "paired_deltas": paired_obj,
    }

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.csv_output:
        _write_csv(
            Path(args.csv_output),
            groups=groups,
            run_rows=run_rows,
            include_runs=bool(args.table_runs),
            ci_method=args.table_ci_method,
            epoch_display=str(args.table_epoch_display),
            metric_scale=str(args.metric_scale),
        )

    if args.table_runs:
        _print_run_rows_table(
            run_rows,
            profile_display=str(args.profile_display),
            time_display=str(args.time_display),
            show_status=bool(args.show_status),
        )
    elif args.table:
        _print_table(
            groups,
            show_ci=bool(args.show_ci),
            ci_method=args.table_ci_method,
            epoch_display=str(args.table_epoch_display),
            metric_scale=str(args.metric_scale),
        )
    else:
        print(json.dumps(obj, indent=2, sort_keys=True))

    # Return non-zero only when no file matched and explicit errors exist.
    if not groups and errors:
        return 2
    return 0


def exp1_validate_main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Validate Exp1 run artifacts for consistency and optional collapse detection.",
    )
    ap.add_argument("--results-root", default="results", type=str)
    ap.add_argument(
        "--experiment-id",
        required=True,
        type=str,
        help="Experiment path under profile root, for example 'pathogens/exp1' or 'exp1'",
    )
    ap.add_argument("--profile-ids", default=None, type=str, help="Optional comma-separated profile id filter")
    ap.add_argument("--tasks", default=None, type=str, help="Optional comma-separated task filter")
    ap.add_argument("--model-ids", default=None, type=str, help="Optional comma-separated model id filter")
    ap.add_argument("--train-variants", default=None, type=str, help="Optional comma-separated train variant filter")
    ap.add_argument("--sets", default=None, type=str, help="Optional comma-separated set filter")
    ap.add_argument("--seeds", default=None, type=str, help="Optional comma-separated seed filter")
    ap.add_argument(
        "--collapse-epsilon",
        default=0.0,
        type=float,
        help="Treat precision/recall/f1 values with absolute value <= epsilon as collapsed",
    )
    ap.add_argument(
        "--fail-on-collapse",
        action="store_true",
        help="Return non-zero when collapsed eval metrics are detected",
    )
    ap.add_argument("--output", default=None, type=str, help="Optional output JSON path")
    args = ap.parse_args(argv)

    results_root = Path(args.results_root)
    profiles_root = _profiles_root(results_root)
    profile_filter = set(_csv(args.profile_ids))
    task_filter = set(_csv(args.tasks))
    model_filter = set(_csv(args.model_ids))
    variant_filter = set(_csv(args.train_variants))
    set_filter = set(_csv(args.sets))
    seed_filter = set(_csv_int(args.seeds))

    exp_roots = _discover_exp_roots(profiles_root, args.experiment_id)
    experiments: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for exp_root in exp_roots:
        profile_id, task = _profile_and_task(profiles_root, exp_root)
        if profile_filter and profile_id not in profile_filter:
            continue
        if task_filter and task not in task_filter:
            continue

        exp_rows: list[dict[str, Any]] = []
        for run_dir in _run_dirs(exp_root):
            row = validate_exp1_run_dir(run_dir=run_dir, collapse_epsilon=float(args.collapse_epsilon))
            row["profile_id"] = profile_id
            row["task"] = task or str(row.get("task") or "")
            row["experiment_id"] = str(args.experiment_id)
            row["exp_root"] = str(exp_root)
            if not _run_matches_filters(
                row,
                model_ids=model_filter,
                train_variants=variant_filter,
                sets=set_filter,
                seeds=seed_filter,
            ):
                continue
            rows.append(row)
            exp_rows.append(row)

        experiments.append(
            {
                "profile_id": profile_id,
                "task": task,
                "exp_root": str(exp_root),
                "n_runs": len(exp_rows),
                "n_ok": int(sum(1 for r in exp_rows if bool(r.get("ok")))),
                "n_failed": int(sum(1 for r in exp_rows if not bool(r.get("ok")))),
                "n_collapsed": int(sum(len(list(r.get("collapsed_eval_sets") or [])) for r in exp_rows)),
            }
        )

    if not exp_roots:
        errors.append("no_experiment_roots_found")
    if not rows:
        errors.append("no_matching_runs")

    failed_runs = [r for r in rows if not bool(r.get("ok"))]
    collapsed_runs: list[dict[str, Any]] = []
    for r in rows:
        for c in list(r.get("collapsed_eval_sets") or []):
            collapsed_runs.append(
                {
                    "run_dir": str(r.get("run_dir") or ""),
                    "profile_id": str(r.get("profile_id") or ""),
                    "task": str(r.get("task") or ""),
                    "set": str(r.get("set") or ""),
                    "seed": r.get("seed"),
                    "model_id": str(r.get("model_id") or ""),
                    "train_variant": str(r.get("train_variant") or ""),
                    **dict(c),
                }
            )

    obj = {
        "results_root": str(results_root),
        "profiles_root": str(profiles_root),
        "experiment_id": str(args.experiment_id),
        "filters": {
            "profile_ids": sorted(profile_filter),
            "tasks": sorted(task_filter),
            "model_ids": sorted(model_filter),
            "train_variants": sorted(variant_filter),
            "sets": sorted(set_filter),
            "seeds": sorted(seed_filter),
        },
        "collapse_epsilon": float(args.collapse_epsilon),
        "n_experiments": len(experiments),
        "n_run_dirs": len(rows),
        "n_ok": int(sum(1 for r in rows if bool(r.get("ok")))),
        "n_failed": len(failed_runs),
        "n_collapsed": len(collapsed_runs),
        "errors": errors,
        "experiments": experiments,
        "failed_runs": [
            {
                "run_dir": str(r.get("run_dir") or ""),
                "profile_id": str(r.get("profile_id") or ""),
                "task": str(r.get("task") or ""),
                "set": str(r.get("set") or ""),
                "seed": r.get("seed"),
                "model_id": str(r.get("model_id") or ""),
                "train_variant": str(r.get("train_variant") or ""),
                "errors": list(r.get("errors") or []),
            }
            for r in failed_runs
        ],
        "collapsed_runs": collapsed_runs,
        "runs": rows,
    }

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(obj, indent=2, sort_keys=True))

    if obj["n_failed"] > 0:
        return 2
    if bool(args.fail_on_collapse) and obj["n_collapsed"] > 0:
        return 2
    if obj["n_run_dirs"] == 0:
        return 2
    return 0
