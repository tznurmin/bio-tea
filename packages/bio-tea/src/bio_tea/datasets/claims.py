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
from itertools import combinations
from pathlib import Path
from typing import Iterable

from .io import read_json, read_jsonl


def _hash_checksums(checksums: Iterable[str]) -> str:
    ordered = [str(x) for x in list(checksums)]
    return hashlib.sha256(("\n".join(ordered)).encode("utf-8")).hexdigest()


def _species_keys_from_meta(meta_rows: list[dict]) -> set[tuple[str, str]]:
    out: set[tuple[str, str]] = set()
    for row in list(meta_rows or []):
        for key in list(row.get("species_keys") or []):
            if isinstance(key, list) and len(key) == 2:
                out.add((str(key[0]), str(key[1])))
    return out


def _strain_phrases_from_meta(meta_rows: list[dict], *, field: str = "strain_phrases") -> set[str]:
    out: set[str] = set()
    for row in list(meta_rows or []):
        for phrase in list(row.get(field) or []):
            phrase_s = str(phrase).strip()
            if phrase_s:
                out.add(phrase_s)
    return out


def _checksums_from_meta(meta_rows: list[dict]) -> set[str]:
    out: set[str] = set()
    for row in list(meta_rows or []):
        checksum = str(row.get("checksum") or "").strip()
        if checksum:
            out.add(checksum)
    return out


def _append_error(report: dict, *, check: str, message: str, **context) -> None:
    report["errors"].append(
        {
            "check": str(check),
            "message": str(message),
            **context,
        }
    )


def _discover_profile_roots(results_root: Path) -> dict[str, Path]:
    profiles_dir = results_root / "profiles"
    if profiles_dir.is_dir():
        out = {
            str(p.name): p
            for p in sorted(profiles_dir.iterdir())
            if p.is_dir()
        }
        if out:
            return out
    return {"flat": results_root}


def _discover_task_roots(profile_root: Path, tasks: list[str] | None) -> dict[str, Path]:
    if tasks:
        out: dict[str, Path] = {}
        for task in tasks:
            task_root = profile_root / str(task)
            if task_root.is_dir():
                out[str(task)] = task_root
        return out

    out: dict[str, Path] = {}
    for child in sorted(profile_root.iterdir()):
        if not child.is_dir():
            continue
        if (child / "splits.json").exists():
            out[str(child.name)] = child
    return out


def _discover_set_roots(task_root: Path) -> dict[str, Path]:
    return {
        str(p.name): p
        for p in sorted(task_root.iterdir())
        if p.is_dir() and str(p.name).startswith("set") and (p / "splits.json").exists()
    }


def _discover_experiment_roots(task_root: Path, experiments: list[str] | None) -> dict[str, Path]:
    if experiments:
        names = [str(x).strip() for x in experiments if str(x).strip()]
        return {
            name: task_root / name
            for name in names
            if (task_root / name).is_dir() and (task_root / name / "manifest.json").exists()
        }
    return {
        str(p.name): p
        for p in sorted(task_root.iterdir())
        if p.is_dir() and str(p.name).startswith("exp") and (p / "manifest.json").exists()
    }


def _build_split_index(results_root: Path, tasks: list[str] | None) -> dict[str, dict[str, dict[str, dict]]]:
    index: dict[str, dict[str, dict[str, dict]]] = {}
    for profile_id, profile_root in _discover_profile_roots(results_root).items():
        task_roots = _discover_task_roots(profile_root, tasks)
        task_index: dict[str, dict[str, dict]] = {}
        for task_name, task_root in task_roots.items():
            task_index[task_name] = {
                set_name: _load_set_split_data(set_root)
                for set_name, set_root in _discover_set_roots(task_root).items()
            }
        index[profile_id] = task_index
    return index


def _variant_name_for_meta(meta_path: Path) -> str:
    if meta_path.name == "train.meta.jsonl":
        return str(meta_path.parent.name)
    return meta_path.name[: -len(".meta.jsonl")]


def _iter_train_meta_paths(train_root: Path) -> list[tuple[str, Path]]:
    if not train_root.is_dir():
        return []
    out: list[tuple[str, Path]] = []
    for meta_path in sorted(train_root.glob("*.meta.jsonl")):
        out.append((_variant_name_for_meta(meta_path), meta_path))
    for meta_path in sorted(train_root.glob("*/train.meta.jsonl")):
        out.append((_variant_name_for_meta(meta_path), meta_path))
    return out


def _load_set_split_data(set_root: Path) -> dict:
    splits = read_json(set_root / "splits.json")
    split_checksums = dict(splits.get("split_checksums") or {})
    if not split_checksums:
        split_checksums = {
            "train": list(splits.get("train_checksums") or []),
            "test": list(splits.get("test_checksums") or []),
            **{
                str(k): list(v)
                for k, v in dict(splits.get("eval_checksums") or {}).items()
            },
        }
    return {
        "splits": splits,
        "split_checksums": {
            str(name): [str(x) for x in list(values or [])]
            for name, values in split_checksums.items()
        },
    }


def _verify_lineage_sets(task_root: Path, set_roots: dict[str, Path], task_report: dict) -> dict:
    check = {
        "ok": True,
        "sets": {},
    }
    split_data_by_set: dict[str, dict] = {}
    for set_name, set_root in set_roots.items():
        payload = _load_set_split_data(set_root)
        split_data_by_set[set_name] = payload
        split_checksums = payload["split_checksums"]
        train_checksums = list(split_checksums.get("train") or [])
        test_checksums = list(split_checksums.get("test") or [])
        eval_checksums = {
            name: list(values)
            for name, values in split_checksums.items()
            if name != "train"
        }
        train_hash = _hash_checksums(train_checksums)
        test_hash = _hash_checksums(test_checksums)
        eval_hashes = {
            name: _hash_checksums(values)
            for name, values in eval_checksums.items()
        }
        check["sets"][set_name] = {
            "train_checksums_sha256": train_hash,
            "test_checksums_sha256": test_hash,
            "eval_checksums_sha256": eval_hashes,
            "n_train_checksums": len(train_checksums),
            "n_test_checksums": len(test_checksums),
            "n_eval_checksums": {k: len(v) for k, v in eval_checksums.items()},
        }

        manifest_path = set_root / "manifest.json"
        if manifest_path.exists():
            manifest = read_json(manifest_path)
            manifest_splits = dict(manifest.get("splits") or {})
            if manifest_splits.get("train_checksums_sha256") != train_hash:
                _append_error(
                    task_report,
                    check="lineage_manifest_train_hash_mismatch",
                    message="set manifest train checksum hash does not match splits.json",
                    task=str(task_root.name),
                    set=set_name,
                )
            if manifest_splits.get("test_checksums_sha256") != test_hash:
                _append_error(
                    task_report,
                    check="lineage_manifest_test_hash_mismatch",
                    message="set manifest test checksum hash does not match splits.json",
                    task=str(task_root.name),
                    set=set_name,
                )
            manifest_eval_hashes = dict(manifest_splits.get("eval_checksums_sha256") or {})
            for split_name, eval_hash in eval_hashes.items():
                if manifest_eval_hashes.get(split_name) != eval_hash:
                    _append_error(
                        task_report,
                        check="lineage_manifest_eval_hash_mismatch",
                        message="set manifest eval checksum hash does not match splits.json",
                        task=str(task_root.name),
                        set=set_name,
                        eval_split=split_name,
                    )

        train_set = set(train_checksums)
        for eval_split, eval_values in eval_checksums.items():
            overlap = sorted(list(train_set.intersection(set(eval_values))))
            if overlap:
                _append_error(
                    task_report,
                    check="lineage_train_eval_checksum_overlap",
                    message="train and eval checksum sets overlap within a lineage",
                    task=str(task_root.name),
                    set=set_name,
                    eval_split=str(eval_split),
                    overlap_count=len(overlap),
                )

    for (set_a, data_a), (set_b, data_b) in combinations(sorted(split_data_by_set.items()), 2):
        for split_name in sorted(set(data_a["split_checksums"].keys()) & set(data_b["split_checksums"].keys())):
            hash_a = _hash_checksums(data_a["split_checksums"][split_name])
            hash_b = _hash_checksums(data_b["split_checksums"][split_name])
            if hash_a == hash_b:
                _append_error(
                    task_report,
                    check="lineage_checksum_signature_collision",
                    message="two lineages use identical checksum signatures for the same split",
                    task=str(task_root.name),
                    split=split_name,
                    sets=[set_a, set_b],
                )

    check["ok"] = not any(
        err["check"].startswith("lineage_")
        for err in list(task_report.get("errors") or [])
    )
    return check


def _allowed_checksums_for_source_splits(split_data: dict, split_names: list[str]) -> set[str]:
    split_checksums = dict(split_data.get("split_checksums") or {})
    allowed: set[str] = set()
    for split_name in list(split_names or []):
        allowed |= {str(x) for x in list(split_checksums.get(str(split_name)) or [])}
    return allowed


def _verify_train_meta_paths(
    *,
    task_root: Path,
    scope_name: str,
    set_name: str,
    meta_paths: list[tuple[str, Path]],
    allowed_checksums: set[str],
    task_report: dict,
) -> dict:
    scope_report = {
        "ok": True,
        "variants": {},
    }
    for variant, meta_path in meta_paths:
        meta_rows = read_jsonl(meta_path)
        present = _checksums_from_meta(meta_rows)
        outside = sorted(list(present - allowed_checksums))
        scope_report["variants"][variant] = {
            "n_examples": len(meta_rows),
            "n_checksums": len(present),
            "n_checksums_outside_source_splits": len(outside),
        }
        if outside:
            _append_error(
                task_report,
                check="train_variant_checksums_outside_source_splits",
                message="training artifact contains checksums outside its declared source splits",
                task=str(task_root.name),
                scope=scope_name,
                set=set_name,
                variant=variant,
                n_outside=len(outside),
                meta_path=str(meta_path.relative_to(task_root.parents[1])),
            )
    scope_report["ok"] = not any(
        err["check"] == "train_variant_checksums_outside_source_splits"
        and err.get("scope") == scope_name
        and err.get("set") == set_name
        for err in list(task_report.get("errors") or [])
    )
    return scope_report


def _verify_base_train_claims(task_root: Path, set_roots: dict[str, Path], split_data_by_set: dict[str, dict], task_report: dict) -> dict:
    out = {"ok": True, "by_set": {}}
    for set_name, set_root in set_roots.items():
        train_root = set_root / "train"
        meta_paths = _iter_train_meta_paths(train_root)
        allowed = _allowed_checksums_for_source_splits(split_data_by_set[set_name], ["train"])
        out["by_set"][set_name] = _verify_train_meta_paths(
            task_root=task_root,
            scope_name="base",
            set_name=set_name,
            meta_paths=meta_paths,
            allowed_checksums=allowed,
            task_report=task_report,
        )
    out["ok"] = not any(
        err["check"] == "train_variant_checksums_outside_source_splits"
        and err.get("scope") == "base"
        for err in list(task_report.get("errors") or [])
    )
    return out


def _compute_training_union_for_experiment(
    *,
    task_root: Path,
    exp_name: str,
    exp_manifest: dict,
    set_name: str,
) -> tuple[set[tuple[str, str]], set[str], set[str], set[str]]:
    if exp_name == "exp3":
        mix_grid = dict(exp_manifest.get("mix_grid") or {})
        source_variants = [str(v).strip() for v in list(mix_grid.get("source_variants") or []) if str(v).strip()]
        meta_paths = [
            task_root / set_name / "train" / f"{variant}.meta.jsonl"
            for variant in source_variants
        ]
    else:
        meta_paths = [
            meta_path
            for _variant, meta_path in _iter_train_meta_paths(task_root / exp_name / set_name / "train")
        ]

    union_species: set[tuple[str, str]] = set()
    union_strains: set[str] = set()
    union_original: set[str] = set()
    union_scrambled: set[str] = set()
    for meta_path in meta_paths:
        if not meta_path.exists():
            continue
        meta_rows = read_jsonl(meta_path)
        union_species |= _species_keys_from_meta(meta_rows)
        union_strains |= _strain_phrases_from_meta(meta_rows, field="strain_phrases")
        union_original |= _strain_phrases_from_meta(meta_rows, field="strain_phrases_original")
        union_scrambled |= _strain_phrases_from_meta(meta_rows, field="strain_phrases_scrambled")
    return union_species, union_strains, union_original, union_scrambled


def _verify_experiment_claims(
    *,
    task_root: Path,
    set_roots: dict[str, Path],
    split_data_by_set: dict[str, dict],
    experiment_roots: dict[str, Path],
    task_report: dict,
) -> dict:
    out = {"ok": True, "by_experiment": {}}
    for exp_name, exp_root in experiment_roots.items():
        manifest = read_json(exp_root / "manifest.json")
        exp_report = {"ok": True, "by_set": {}}
        train_source_splits = [str(x) for x in list(manifest.get("train_source_splits") or ["train"])]
        eval_split = str(manifest.get("eval_split") or "test")
        for set_name in set_roots:
            allowed = _allowed_checksums_for_source_splits(split_data_by_set[set_name], train_source_splits if exp_name == "exp1" else ["train"])
            meta_paths = _iter_train_meta_paths(exp_root / set_name / "train")
            set_report = _verify_train_meta_paths(
                task_root=task_root,
                scope_name=exp_name,
                set_name=set_name,
                meta_paths=meta_paths,
                allowed_checksums=allowed,
                task_report=task_report,
            )

            exclusion_path = exp_root / set_name / eval_split / "augmented_exclusive_exclusions.json"
            eval_meta_path = exp_root / set_name / eval_split / "augmented_exclusive.meta.jsonl"
            if exclusion_path.exists() and eval_meta_path.exists():
                exclusions = read_json(exclusion_path)
                eval_meta = read_jsonl(eval_meta_path)
                disallowed_species = {
                    (str(key[0]), str(key[1]))
                    for key in list(exclusions.get("species_keys_disallowed") or [])
                    if isinstance(key, list) and len(key) == 2
                }
                disallowed_strains = {
                    str(phrase).strip()
                    for phrase in list(exclusions.get("strain_phrases_disallowed") or [])
                    if str(phrase).strip()
                }
                actual_species, actual_strains, actual_original, actual_scrambled = _compute_training_union_for_experiment(
                    task_root=task_root,
                    exp_name=exp_name,
                    exp_manifest=manifest,
                    set_name=set_name,
                )

                if disallowed_species != actual_species or disallowed_strains != actual_strains:
                    _append_error(
                        task_report,
                        check="augmented_exclusive_exclusion_artifact_mismatch",
                        message="augmented-exclusive exclusion artifact does not match the training-union metadata",
                        task=str(task_root.name),
                        experiment=exp_name,
                        set=set_name,
                    )
                artifact_original = {
                    str(phrase).strip()
                    for phrase in list(exclusions.get("train_strain_phrases_original") or [])
                    if str(phrase).strip()
                }
                artifact_scrambled = {
                    str(phrase).strip()
                    for phrase in list(exclusions.get("train_strain_phrases_scrambled") or [])
                    if str(phrase).strip()
                }
                if artifact_original != actual_original:
                    _append_error(
                        task_report,
                        check="augmented_exclusive_original_strain_artifact_mismatch",
                        message="original strain phrase exclusion artifact does not match training metadata",
                        task=str(task_root.name),
                        experiment=exp_name,
                        set=set_name,
                    )
                if artifact_scrambled != actual_scrambled:
                    _append_error(
                        task_report,
                        check="augmented_exclusive_scrambled_strain_artifact_mismatch",
                        message="scrambled strain phrase exclusion artifact does not match training metadata",
                        task=str(task_root.name),
                        experiment=exp_name,
                        set=set_name,
                    )

                species_overlap = 0
                strain_overlap = 0
                for row in eval_meta:
                    row_species = _species_keys_from_meta([row])
                    row_strains = _strain_phrases_from_meta([row], field="strain_phrases")
                    if row_species.intersection(disallowed_species):
                        species_overlap += 1
                    if row_strains.intersection(disallowed_strains):
                        strain_overlap += 1
                if species_overlap:
                    _append_error(
                        task_report,
                        check="augmented_exclusive_species_overlap",
                        message="augmented-exclusive eval rows still contain training species keys",
                        task=str(task_root.name),
                        experiment=exp_name,
                        set=set_name,
                        overlap_rows=species_overlap,
                    )
                if strain_overlap:
                    _append_error(
                        task_report,
                        check="augmented_exclusive_strain_overlap",
                        message="augmented-exclusive eval rows still contain training strain phrases",
                        task=str(task_root.name),
                        experiment=exp_name,
                        set=set_name,
                        overlap_rows=strain_overlap,
                    )

            set_report["augmented_exclusive_verified"] = bool(exclusion_path.exists() and eval_meta_path.exists())
            exp_report["by_set"][set_name] = set_report

        exp_report["ok"] = not any(
            err.get("experiment") == exp_name for err in list(task_report.get("errors") or [])
        )
        out["by_experiment"][exp_name] = exp_report
    out["ok"] = not any(
        err["check"].startswith("augmented_exclusive_")
        or err["check"] == "train_variant_checksums_outside_source_splits"
        and err.get("scope") != "base"
        for err in list(task_report.get("errors") or [])
    )
    return out


def _verify_calibration_relation(
    *,
    profile_id: str,
    task_name: str,
    split_data_by_set: dict[str, dict],
    calibration_index: dict[str, dict[str, dict[str, dict]]],
    calibration_eval_split: str,
    task_report: dict,
) -> dict:
    out = {
        "ok": True,
        "calibration_eval_split": str(calibration_eval_split),
        "by_set": {},
    }
    profile_data = calibration_index.get(profile_id)
    if profile_data is None:
        _append_error(
            task_report,
            check="calibration_profile_missing",
            message="matching profile was not found under the calibration results root",
            task=task_name,
            profile=profile_id,
        )
        out["ok"] = False
        return out

    task_data = profile_data.get(task_name)
    if task_data is None:
        _append_error(
            task_report,
            check="calibration_task_missing",
            message="matching task was not found under the calibration profile root",
            task=task_name,
            profile=profile_id,
        )
        out["ok"] = False
        return out

    for set_name, target_data in split_data_by_set.items():
        set_report = {"ok": True}
        calib_data = task_data.get(set_name)
        if calib_data is None:
            _append_error(
                task_report,
                check="calibration_set_missing",
                message="matching lineage set was not found under the calibration task root",
                task=task_name,
                profile=profile_id,
                set=set_name,
            )
            set_report["ok"] = False
            out["by_set"][set_name] = set_report
            continue

        cal_splits = dict(calib_data.get("split_checksums") or {})
        target_splits = dict(target_data.get("split_checksums") or {})
        cal_train = set(str(x) for x in list(cal_splits.get("train") or []))
        cal_eval = set(str(x) for x in list(cal_splits.get(str(calibration_eval_split)) or []))
        cal_test = set(str(x) for x in list(cal_splits.get("test") or []))
        target_train = set(str(x) for x in list(target_splits.get("train") or []))
        target_test = set(str(x) for x in list(target_splits.get("test") or []))

        set_report["calibration_train_size"] = len(cal_train)
        set_report["calibration_eval_size"] = len(cal_eval)
        set_report["calibration_test_size"] = len(cal_test)
        set_report["target_train_size"] = len(target_train)
        set_report["target_test_size"] = len(target_test)

        if str(calibration_eval_split) not in cal_splits:
            _append_error(
                task_report,
                check="calibration_eval_split_missing",
                message="requested calibration eval split is missing from calibration lineage splits",
                task=task_name,
                profile=profile_id,
                set=set_name,
                calibration_eval_split=str(calibration_eval_split),
            )
            set_report["ok"] = False
            out["by_set"][set_name] = set_report
            continue

        if target_test != cal_test:
            _append_error(
                task_report,
                check="calibration_campaign_test_mismatch",
                message="campaign test split does not match calibration test split",
                task=task_name,
                profile=profile_id,
                set=set_name,
            )
            set_report["ok"] = False

        if target_train != (cal_train | cal_eval):
            _append_error(
                task_report,
                check="calibration_campaign_train_union_mismatch",
                message="campaign train split is not equal to calibration train plus calibration eval split",
                task=task_name,
                profile=profile_id,
                set=set_name,
                calibration_eval_split=str(calibration_eval_split),
            )
            set_report["ok"] = False

        out["by_set"][set_name] = set_report

    out["ok"] = not any(
        err["check"].startswith("calibration_")
        and err.get("task") == task_name
        and err.get("profile") == profile_id
        for err in list(task_report.get("errors") or [])
    )
    return out


def verify_results_root_claims(
    results_root: Path,
    *,
    tasks: list[str] | None = None,
    experiments: list[str] | None = None,
    calibration_results_root: Path | None = None,
    calibration_eval_split: str = "dev",
) -> dict:
    root = Path(results_root)
    report = {
        "results_root": str(root),
        "tasks_requested": list(tasks or []),
        "experiments_requested": list(experiments or []),
        "calibration_results_root": str(calibration_results_root) if calibration_results_root else None,
        "calibration_eval_split": str(calibration_eval_split),
        "profiles": {},
        "errors": [],
        "ok": True,
        "n_errors": 0,
    }

    if not root.exists() or not root.is_dir():
        _append_error(
            report,
            check="results_root_missing",
            message="results root does not exist or is not a directory",
            results_root=str(root),
        )
        report["n_errors"] = len(report["errors"])
        report["ok"] = False
        return report

    discovered_tasks: set[str] = set()
    discovered_experiments: set[str] = set()
    calibration_index: dict[str, dict[str, dict[str, dict]]] | None = None
    if calibration_results_root is not None:
        calibration_root = Path(calibration_results_root)
        if not calibration_root.exists() or not calibration_root.is_dir():
            _append_error(
                report,
                check="calibration_results_root_missing",
                message="calibration results root does not exist or is not a directory",
                calibration_results_root=str(calibration_root),
            )
            report["n_errors"] = len(report["errors"])
            report["ok"] = False
            return report
        calibration_index = _build_split_index(calibration_root, tasks)

    for profile_id, profile_root in _discover_profile_roots(root).items():
        profile_report = {
            "root": str(profile_root),
            "tasks": {},
            "ok": True,
        }
        task_roots = _discover_task_roots(profile_root, tasks)
        for task_name, task_root in task_roots.items():
            discovered_tasks.add(str(task_name))
            task_report = {
                "ok": True,
                "errors": [],
            }
            set_roots = _discover_set_roots(task_root)
            split_data_by_set = {
                set_name: _load_set_split_data(set_root)
                for set_name, set_root in set_roots.items()
            }
            task_report["lineage_checks"] = _verify_lineage_sets(task_root, set_roots, task_report)
            task_report["base_train_checks"] = _verify_base_train_claims(
                task_root,
                set_roots,
                split_data_by_set,
                task_report,
            )
            task_report["experiment_checks"] = _verify_experiment_claims(
                task_root=task_root,
                set_roots=set_roots,
                split_data_by_set=split_data_by_set,
                experiment_roots=_discover_experiment_roots(task_root, experiments),
                task_report=task_report,
            )
            if calibration_index is not None:
                task_report["calibration_comparison"] = _verify_calibration_relation(
                    profile_id=str(profile_id),
                    task_name=str(task_name),
                    split_data_by_set=split_data_by_set,
                    calibration_index=calibration_index,
                    calibration_eval_split=str(calibration_eval_split),
                    task_report=task_report,
                )
            discovered_experiments |= set(task_report["experiment_checks"]["by_experiment"].keys())
            task_report["ok"] = not bool(task_report["errors"])
            profile_report["tasks"][task_name] = task_report
            for err in list(task_report["errors"]):
                err["profile"] = str(profile_id)
                report["errors"].append(err)
        profile_report["ok"] = all(
            bool(task_info.get("ok"))
            for task_info in profile_report["tasks"].values()
        )
        report["profiles"][profile_id] = profile_report

    if not discovered_tasks:
        _append_error(
            report,
            check="no_task_artifacts_found",
            message="no task artifacts were discovered under the results root",
            results_root=str(root),
        )

    for task_name in list(tasks or []):
        if str(task_name) not in discovered_tasks:
            _append_error(
                report,
                check="requested_task_missing",
                message="requested task was not found under the results root",
                task=str(task_name),
                results_root=str(root),
            )

    for exp_name in list(experiments or []):
        if str(exp_name) not in discovered_experiments:
            _append_error(
                report,
                check="requested_experiment_missing",
                message="requested experiment was not found under the filtered task roots",
                experiment=str(exp_name),
                results_root=str(root),
            )

    report["n_errors"] = len(report["errors"])
    report["ok"] = report["n_errors"] == 0
    return report
