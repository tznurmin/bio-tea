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

import random
from collections.abc import Iterable, Mapping

_RESERVED_SPLIT_CFG_KEYS = {
    "split_seed",
    "tasks",
    "test_fraction",
    "train_fraction",
    "dev_fraction",
}


def category_from_curation_key(key: str) -> str:
    """Return the curated category for a curation key.

    Convention: the category is the prefix before the first '/'.
    Example: 'pathogen/yellow' -> 'pathogen'.
    """
    if not key:
        return ""
    return key.split("/", 1)[0]


def categories_for_article(curation_data_for_article: dict) -> set[str]:
    cats: set[str] = set()
    for k in (curation_data_for_article or {}).keys():
        c = category_from_curation_key(str(k))
        if c:
            cats.add(c)
    return cats


def build_article_category_map(task_curation: dict[str, dict]) -> dict[str, set[str]]:
    return {csum: categories_for_article(task_curation.get(csum) or {}) for csum in task_curation.keys()}


def _coverage(articles: Iterable[str], csum2cats: dict[str, set[str]]) -> set[str]:
    out: set[str] = set()
    for csum in articles:
        out |= set(csum2cats.get(csum) or set())
    return out


def _category_counts(articles: Iterable[str], csum2cats: dict[str, set[str]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for csum in articles:
        for cat in (csum2cats.get(csum) or set()):
            out[cat] = int(out.get(cat, 0)) + 1
    return out


def _is_removable_from_split(csum: str, split_set: set[str], csum2cats: dict[str, set[str]]) -> bool:
    cats = set(csum2cats.get(csum) or set())
    if not cats:
        return True
    remaining = split_set - {csum}
    cov = _coverage(remaining, csum2cats)
    return cats.issubset(cov)


def _as_fraction(v: object, *, key: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError(f"Split {key!r} fraction must be numeric")
    return float(v)


def split_specs_from_config(splits_cfg: Mapping[str, object] | None) -> dict[str, dict]:
    """Normalize split configuration into explicit named split specs.

    Supported forms:
    1) Named split form:
       splits:
         train: 0.85
         test:
           fraction: 0.15
           coverage:
             min_articles_per_category: 1

    2) Fraction-only form:
       splits:
         test_fraction: 0.15
         # optional train_fraction/dev_fraction
    """

    cfg = dict(splits_cfg or {})
    out: dict[str, dict] = {}

    # Named split form.
    for k, v in cfg.items():
        if k in _RESERVED_SPLIT_CFG_KEYS:
            continue
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out[str(k)] = {"fraction": float(v)}
            continue
        if isinstance(v, Mapping):
            if "fraction" not in v:
                raise ValueError(f"Split {k!r} must include 'fraction'")
            spec = {"fraction": _as_fraction(v["fraction"], key=str(k))}
            cov = v.get("coverage")
            if isinstance(cov, Mapping):
                spec["coverage"] = dict(cov)
            out[str(k)] = spec
            continue
        raise ValueError(f"Invalid split config entry for {k!r}")

    if out:
        _validate_split_fractions(out)
        return out

    # Train/test configuration form.
    test_fraction = float(cfg.get("test_fraction", 0.15))
    dev_fraction = float(cfg.get("dev_fraction", 0.0))
    if "train_fraction" in cfg:
        train_fraction = float(cfg.get("train_fraction", 0.0))
    else:
        train_fraction = 1.0 - test_fraction - dev_fraction

    out = {"train": {"fraction": train_fraction}}
    if dev_fraction > 0.0:
        out["dev"] = {"fraction": dev_fraction}
    out["test"] = {"fraction": test_fraction, "coverage": {"min_articles_per_category": 1}}
    _validate_split_fractions(out)
    return out


def _validate_split_fractions(split_specs: Mapping[str, Mapping[str, object]]) -> None:
    if not split_specs:
        raise ValueError("At least one split is required")

    total = 0.0
    for name, spec in split_specs.items():
        frac = _as_fraction(spec.get("fraction"), key=name)
        if frac < 0.0 or frac > 1.0:
            raise ValueError(f"Split {name!r} fraction must be in [0, 1]")
        total += frac

    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"Split fractions must sum to 1.0 (got {total:.12f})")


def _initial_counts(n: int, split_specs: Mapping[str, Mapping[str, object]]) -> dict[str, int]:
    names = list(split_specs.keys())
    raw = [n * float(split_specs[name]["fraction"]) for name in names]
    floors = [int(x) for x in raw]
    rem = n - sum(floors)

    # Largest remainder, deterministic tie-break by split order.
    remainders = [raw[i] - floors[i] for i in range(len(names))]
    order = sorted(range(len(names)), key=lambda i: remainders[i], reverse=True)
    for i in range(rem):
        floors[order[i]] += 1

    return {name: floors[i] for i, name in enumerate(names)}


def _choose_donor_order(splits: Mapping[str, set[str]], target: str) -> list[str]:
    names = [n for n in splits.keys() if n != target]
    if "train" in names:
        rest = [n for n in names if n != "train"]
        rest = sorted(rest, key=lambda n: (-len(splits[n]), n))
        return ["train", *rest]
    return sorted(names, key=lambda n: (-len(splits[n]), n))


def _choose_best_donor_candidate(
    *,
    splits: Mapping[str, set[str]],
    target: str,
    csum2cats: dict[str, set[str]],
    missing: set[str],
) -> tuple[str | None, str | None]:
    """Pick swap-in article maximizing missing-category coverage.

    Tie-breaks:
    1) donor split priority (`train` first)
    2) checksum lexical order
    """

    donor_order = _choose_donor_order(splits, target)
    donor_rank = {name: i for i, name in enumerate(donor_order)}

    best: tuple[int, int, str, str] | None = None
    # Tuple layout: (-covered_missing_count, donor_rank, checksum, donor_name)
    for donor_name in donor_order:
        for csum in sorted(list(splits[donor_name])):
            covered = len((csum2cats.get(csum) or set()).intersection(missing))
            if covered <= 0:
                continue
            cur = (-covered, donor_rank[donor_name], csum, donor_name)
            if best is None or cur < best:
                best = cur

    if best is None:
        return None, None
    return best[3], best[2]


def _choose_best_swap_out(
    *,
    target_set: set[str],
    removable: list[str],
    csum2cats: dict[str, set[str]],
) -> str:
    """Choose minimally disruptive removable article.

    Priorities:
    1) irrelevant-only article (no categories) priority
    2) higher category redundancy priority
    3) checksum lexical order for deterministic tie-break
    """

    cat_counts = _category_counts(target_set, csum2cats)
    best_key: tuple[int, int, str] | None = None
    best_csum: str | None = None

    for csum in sorted(list(removable)):
        cats = set(csum2cats.get(csum) or set())
        is_irrelevant_only = 1 if not cats else 0
        redundancy = sum(max(int(cat_counts.get(cat, 0)) - 1, 0) for cat in cats)
        cur_key = (-is_irrelevant_only, -int(redundancy), csum)
        if best_key is None or cur_key < best_key:
            best_key = cur_key
            best_csum = csum

    if best_csum is None:
        raise ValueError("removable list must be non-empty")
    return best_csum


def _apply_coverage_min_one(
    *,
    splits: dict[str, set[str]],
    target: str,
    csum2cats: dict[str, set[str]],
    rng: random.Random,
) -> int:
    """Ensure target split has at least one article per category.

    Returns net growth in target split size.
    """

    stats = _apply_coverage_min_one_with_stats(
        splits=splits,
        target=target,
        csum2cats=csum2cats,
        rng=rng,
    )
    return int(stats['n_final'] - stats['n_initial'])


def _apply_coverage_min_one_with_stats(
    *,
    splits: dict[str, set[str]],
    target: str,
    csum2cats: dict[str, set[str]],
    rng: random.Random,
) -> dict:
    """Coverage enforcement with decision trace for manifest reporting."""

    target_set = splits[target]
    target_initial = len(target_set)

    # Coverage enforcement is fully deterministic by score + checksum
    # tie-breakers; rng is accepted for API consistency.
    _ = rng

    all_cats = _coverage(csum2cats.keys(), csum2cats)
    n_missing_initial = len(all_cats - _coverage(target_set, csum2cats))
    n_swaps = 0
    n_adds = 0
    decisions: list[dict] = []

    donor_order = _choose_donor_order(splits, target)
    donor_rank = {name: i for i, name in enumerate(donor_order)}

    while True:
        missing = set(all_cats - _coverage(target_set, csum2cats))
        if not missing:
            break

        donor_name, candidate = _choose_best_donor_candidate(
            splits=splits,
            target=target,
            csum2cats=csum2cats,
            missing=missing,
        )
        if donor_name is None or candidate is None:
            decisions.append(
                {
                    'action': 'unresolved_missing',
                    'missing_categories_before': sorted(list(missing)),
                }
            )
            break

        donor_cover = len((csum2cats.get(candidate) or set()).intersection(missing))
        donor_key = [-int(donor_cover), int(donor_rank.get(donor_name, 999999)), str(candidate)]

        removable = sorted([cs for cs in target_set if _is_removable_from_split(cs, target_set, csum2cats)])
        if removable:
            chosen_target = _choose_best_swap_out(
                target_set=target_set,
                removable=removable,
                csum2cats=csum2cats,
            )
            cat_counts = _category_counts(target_set, csum2cats)
            out_cats = set(csum2cats.get(chosen_target) or set())
            out_irrelevant_only = not out_cats
            out_redundancy = sum(max(int(cat_counts.get(cat, 0)) - 1, 0) for cat in out_cats)
            swap_out_key = [-int(1 if out_irrelevant_only else 0), -int(out_redundancy), str(chosen_target)]

            splits[donor_name].remove(candidate)
            target_set.add(candidate)
            target_set.remove(chosen_target)
            splits[donor_name].add(chosen_target)
            n_swaps += 1
            decisions.append(
                {
                    'action': 'swap',
                    'missing_categories_before': sorted(list(missing)),
                    'donor_split': donor_name,
                    'donor_checksum': candidate,
                    'donor_cover_count': int(donor_cover),
                    'donor_selection_key': donor_key,
                    'swap_out_checksum': chosen_target,
                    'swap_out_is_irrelevant_only': bool(out_irrelevant_only),
                    'swap_out_redundancy': int(out_redundancy),
                    'swap_out_selection_key': swap_out_key,
                }
            )
        else:
            # Add fallback: grow target split when no coverage-safe swap-out exists.
            splits[donor_name].remove(candidate)
            target_set.add(candidate)
            n_adds += 1
            decisions.append(
                {
                    'action': 'add',
                    'missing_categories_before': sorted(list(missing)),
                    'donor_split': donor_name,
                    'donor_checksum': candidate,
                    'donor_cover_count': int(donor_cover),
                    'donor_selection_key': donor_key,
                }
            )

    n_missing_final = len(all_cats - _coverage(target_set, csum2cats))
    return {
        'n_initial': int(target_initial),
        'n_final': int(len(target_set)),
        'n_missing_initial': int(n_missing_initial),
        'n_missing_final': int(n_missing_final),
        'n_steps': int(len([d for d in decisions if d.get('action') in {'swap', 'add'}])),
        'n_swaps': int(n_swaps),
        'n_adds': int(n_adds),
        'decision_trace': decisions,
    }


def split_articles_multi(
    *,
    checksums: list[str],
    csum2cats: dict[str, set[str]],
    split_specs: Mapping[str, Mapping[str, object]],
    seed: int,
) -> tuple[dict[str, list[str]], dict]:
    """Split articles into arbitrary named splits with optional split policies."""

    _validate_split_fractions(split_specs)

    rng = random.Random(int(seed))
    shuffled = list(checksums)
    rng.shuffle(shuffled)

    n = len(shuffled)
    counts = _initial_counts(n, split_specs)

    splits: dict[str, list[str]] = {}
    at = 0
    for name in split_specs.keys():
        c = counts[name]
        splits[name] = sorted(shuffled[at : at + c])
        at += c

    initial_sizes = {k: len(v) for k, v in splits.items()}
    split_sets = {k: set(v) for k, v in splits.items()}
    coverage_growth: dict[str, int] = {}
    coverage_details: dict[str, dict] = {}

    for name, spec in split_specs.items():
        cov = spec.get("coverage")
        if not isinstance(cov, Mapping):
            continue
        min_per_cat = int(cov.get("min_articles_per_category", 0))
        if min_per_cat <= 0:
            continue
        if min_per_cat != 1:
            raise ValueError("Only min_articles_per_category=1 is currently supported")
        detail = _apply_coverage_min_one_with_stats(
            splits=split_sets,
            target=name,
            csum2cats=csum2cats,
            rng=rng,
        )
        coverage_growth[name] = int(detail['n_final'] - detail['n_initial'])
        coverage_details[name] = detail

    final_splits = {k: sorted(list(v)) for k, v in split_sets.items()}
    all_cats = sorted(list(_coverage(csum2cats.keys(), csum2cats)))

    stats = {
        "seed": int(seed),
        "n_total": int(n),
        "all_categories": all_cats,
        "splits": {},
    }
    for name in split_specs.keys():
        stats["splits"][name] = {
            "fraction": float(split_specs[name]["fraction"]),
            "n_initial": int(initial_sizes[name]),
            "n_final": int(len(final_splits[name])),
            "n_added_for_category_coverage": int(coverage_growth.get(name, 0)),
            "categories": sorted(list(_coverage(final_splits[name], csum2cats))),
            "coverage": dict(split_specs[name].get("coverage") or {}),
            "coverage_actions": {
                "n_missing_initial": int((coverage_details.get(name) or {}).get("n_missing_initial", 0)),
                "n_missing_final": int((coverage_details.get(name) or {}).get("n_missing_final", 0)),
                "n_steps": int((coverage_details.get(name) or {}).get("n_steps", 0)),
                "n_swaps": int((coverage_details.get(name) or {}).get("n_swaps", 0)),
                "n_adds": int((coverage_details.get(name) or {}).get("n_adds", 0)),
            },
            "coverage_decision_trace": list((coverage_details.get(name) or {}).get("decision_trace") or []),
        }

    return final_splits, stats


def ensure_test_category_coverage(
    *,
    train: list[str],
    test: list[str],
    csum2cats: dict[str, set[str]],
    rng: random.Random,
) -> tuple[list[str], list[str]]:
    """Apply train/test category coverage constraints."""

    # Preserve the incoming split membership while applying coverage constraints.
    split_sets = {"train": set(train), "test": set(test)}
    _apply_coverage_min_one(splits=split_sets, target="test", csum2cats=csum2cats, rng=rng)
    return sorted(list(split_sets["train"])), sorted(list(split_sets["test"]))


def split_articles(
    *,
    checksums: list[str],
    csum2cats: dict[str, set[str]],
    test_fraction: float,
    seed: int,
) -> tuple[list[str], list[str], dict]:
    """Split articles into train/test partitions."""

    split_specs = {
        "train": {"fraction": 1.0 - float(test_fraction)},
        "test": {"fraction": float(test_fraction), "coverage": {"min_articles_per_category": 1}},
    }
    split_map, stats_multi = split_articles_multi(
        checksums=checksums,
        csum2cats=csum2cats,
        split_specs=split_specs,
        seed=seed,
    )

    train = split_map["train"]
    test = split_map["test"]
    stats = {
        "seed": int(seed),
        "test_fraction": float(test_fraction),
        "n_total": int(len(checksums)),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_test_initial": int(stats_multi["splits"]["test"]["n_initial"]),
        "n_moved_for_category_coverage": int(stats_multi["splits"]["test"]["n_added_for_category_coverage"]),
        "all_categories": list(stats_multi["all_categories"]),
        "test_categories": list(stats_multi["splits"]["test"]["categories"]),
    }
    return train, test, stats
