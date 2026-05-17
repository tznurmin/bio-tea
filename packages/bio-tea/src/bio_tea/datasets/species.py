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

import re
from typing import Iterable


_SPEC_RE = re.compile(r"[A-Z](?:[a-z]+|\.)\s[a-z]+")


def species_keys_in_text(text: str, *, all_species: set[str]) -> set[tuple[str, str]]:
    """Return abbreviation-keys (genus initial, epithet) for recognized species.

    Keys are derived from the exact mention string if it is present in all_species.
    - Full: 'Escherichia coli' -> ('E','coli')
    - Abbrev: 'E. coli' -> ('E','coli')

    This key-space is used for overlap exclusion between training and augmented
    test sets.
    """
    out: set[tuple[str, str]] = set()
    for m in _SPEC_RE.findall(text):
        if m not in all_species:
            continue
        parts = m.split(' ')
        if len(parts) != 2:
            continue
        initial = parts[0][0]
        epithet = parts[1]
        out.add((initial, epithet))
    return out


def build_restricted_pool(
    *,
    full_species_pool: Iterable[str],
    disallowed_keys: set[tuple[str, str]],
) -> list[str]:
    """Filter a full species pool to exclude any species with disallowed keys."""
    out: list[str] = []
    for sp in full_species_pool:
        parts = sp.split(' ')
        if len(parts) != 2:
            continue
        key = (parts[0][0], parts[1])
        if key in disallowed_keys:
            continue
        out.append(sp)
    return out


def species_key_from_phrase(phrase: str) -> tuple[str, str] | None:
    parts = str(phrase or "").strip().split(" ")
    if len(parts) != 2:
        return None
    genus = parts[0].strip()
    epithet = parts[1].strip()
    if not genus or not epithet:
        return None
    return (genus[0], epithet)


def full_species_pool_from_all_species(all_species: Iterable[str]) -> list[str]:
    out: list[str] = []
    for sp in sorted(set(all_species)):
        if len(sp) > 1 and sp[1] == '.':
            continue
        if species_key_from_phrase(sp) is None:
            continue
        out.append(sp)
    return out


def build_switch_pool_for_mode(
    *,
    all_species: Iterable[str],
    curated_species_index: list[dict] | None,
    mode: str,
) -> list[str]:
    mode_norm = str(mode or "full").strip().lower()
    curated_species_index = list(curated_species_index or [])

    if mode_norm == "full":
        return full_species_pool_from_all_species(all_species)

    if mode_norm == "curated_only":
        reps: set[str] = set()
        for row in curated_species_index:
            rep = str((row or {}).get("representative") or "").strip()
            if rep:
                reps.add(rep)
        return sorted(reps)

    if mode_norm == "exclude_curated":
        disallowed: set[tuple[str, str]] = set()
        for row in curated_species_index:
            key = (row or {}).get("key")
            if isinstance(key, (list, tuple)) and len(key) == 2:
                disallowed.add((str(key[0]), str(key[1])))
        return build_restricted_pool(
            full_species_pool=full_species_pool_from_all_species(all_species),
            disallowed_keys=disallowed,
        )

    raise ValueError("species switch pool_mode must be one of: full, curated_only, exclude_curated")


def build_curated_species_coverage_plan(
    *,
    records: list[dict],
    all_species: set[str],
    curated_species_index: list[dict],
    candidate_scores: dict[str, int] | None = None,
) -> dict:
    """Assign at most one target species key per candidate example.

    The mapping is deterministic (sorted by `example_id` and species key).
    """

    candidates: list[str] = []
    for rec in sorted(records, key=lambda r: str(r.get("example_id") or "")):
        exid = str(rec.get("example_id") or "")
        txt = str(rec.get("text") or "")
        if not exid or not txt:
            continue
        if species_keys_in_text(txt, all_species=all_species):
            candidates.append(exid)
    if candidate_scores:
        default_score = 10**12
        candidates = sorted(
            candidates,
            key=lambda exid: (int(candidate_scores.get(exid, default_score)), exid),
        )

    targets: list[dict] = []
    for row in list(curated_species_index or []):
        key = (row or {}).get("key")
        rep = str((row or {}).get("representative") or "").strip()
        if not (isinstance(key, (list, tuple)) and len(key) == 2 and rep):
            continue
        targets.append(
            {
                "key": (str(key[0]), str(key[1])),
                "representative": rep,
            }
        )
    targets.sort(key=lambda r: (r["key"][0], r["key"][1]))

    assignments: dict[str, str] = {}
    assignment_keys: dict[str, tuple[str, str]] = {}
    unassigned: list[dict] = []
    ci = 0
    for t in targets:
        if ci >= len(candidates):
            unassigned.append(
                {
                    "key": [t["key"][0], t["key"][1]],
                    "representative": t["representative"],
                    "reason": "no_candidate_record",
                }
            )
            continue
        exid = candidates[ci]
        ci += 1
        assignments[exid] = str(t["representative"])
        assignment_keys[exid] = (t["key"][0], t["key"][1])

    return {
        "assignments": assignments,
        "assignment_keys": {k: [v[0], v[1]] for k, v in assignment_keys.items()},
        "unassigned": unassigned,
        "n_candidates": len(candidates),
        "n_targets": len(targets),
        "n_assigned": len(assignments),
        "n_unassigned": len(unassigned),
    }
