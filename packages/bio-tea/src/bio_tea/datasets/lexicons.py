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
from collections.abc import Callable, Mapping

_SPEC_RE = re.compile(r"^[A-Z](?:[a-z]+|\.)\s[a-z]+$")


def _label_code_from_key(key: str) -> str:
    prefix = str(key).split("/", 1)[0]
    return prefix[0:4].upper()


def _species_key_and_form(phrase: str) -> tuple[tuple[str, str], bool] | None:
    parts = str(phrase or "").strip().split(" ")
    if len(parts) != 2:
        return None
    genus = parts[0].strip()
    epithet = parts[1].strip()
    if not genus or not epithet:
        return None
    key = (genus[0], epithet)
    is_abbrev = genus.endswith(".")
    return key, is_abbrev


def _normalize_code_set(raw: object, *, default: set[str], field: str) -> set[str]:
    if raw is None:
        return set(default)
    if isinstance(raw, str):
        vals = [raw]
    elif isinstance(raw, (list, tuple, set)):
        vals = list(raw)
    else:
        raise ValueError(f"lexicons.{field} must be list/tuple/set/string or mapping by task")

    out: set[str] = set()
    for v in vals:
        s = str(v).strip()
        if not s:
            continue
        out.add(s.upper()[:4])
    return out or set(default)


def resolve_lexicon_code_sets(*, lex_cfg: Mapping[str, object] | None, task: str) -> tuple[set[str], set[str]]:
    """Resolve species/strain code sets for a task.

    Supports both global list form and task-scoped mapping form:

    - lexicons.species_codes: [SPEC, PATH]
    - lexicons.species_codes:
        pathogens: [PATH, OPPO, PROB]
        strains: [SPEC]
        default: [SPEC]

    Mapping lookup precedence:
    1) explicit task name
    2) `default`
    3) `*`
    4) built-in default
    """

    cfg = dict(lex_cfg or {})
    species_default = {"SPEC"}
    strain_default = {"STRA"}

    def _for_field(field: str, default: set[str]) -> set[str]:
        raw = cfg.get(field)
        if isinstance(raw, Mapping):
            scoped = raw.get(task)
            if scoped is None:
                scoped = raw.get("default")
            if scoped is None:
                scoped = raw.get("*")
            return _normalize_code_set(scoped, default=default, field=field)
        return _normalize_code_set(raw, default=default, field=field)

    return _for_field("species_codes", species_default), _for_field("strain_codes", strain_default)


def _span_phrase(tokens: list[str], loc: str) -> str:
    s, e = str(loc).split("+", 1)
    s_i = int(s)
    e_i = int(e)
    return " ".join(tokens[s_i:s_i + e_i])


def extract_curated_lexicons(
    *,
    task_curation: Mapping[str, Mapping[str, list[str]]],
    load_tokens: Callable[[str], list[str]],
    species_codes: set[str] | None = None,
    strain_codes: set[str] | None = None,
) -> dict:
    """Extract additional species names and reserved strain phrases from curated spans.

    Species extraction keeps only spans matching TEA's species detection pattern.
    Strain extraction keeps all continuous span phrases for configured strain label codes.
    """

    species_codes = set(species_codes or {"SPEC"})
    strain_codes = set(strain_codes or {"STRA"})

    species_spans_all: set[str] = set()
    reserved_strains: set[str] = set()
    species_spans_by_code_raw: dict[str, int] = {}
    species_spans_by_code_matching_pattern: dict[str, int] = {}
    strain_spans_by_code_raw: dict[str, int] = {}
    extracted_span_counts: dict[tuple[str, str, str, bool], int] = {}

    for checksum in sorted(task_curation.keys()):
        curation_data = task_curation.get(checksum) or {}
        if not curation_data:
            continue
        tokens = load_tokens(checksum)

        for k, locs in curation_data.items():
            code = _label_code_from_key(str(k))
            if code in species_codes:
                for loc in list(locs):
                    phrase = _span_phrase(tokens, str(loc))
                    is_match = bool(_SPEC_RE.match(phrase))
                    species_spans_all.add(phrase)
                    species_spans_by_code_raw[code] = int(species_spans_by_code_raw.get(code, 0)) + 1
                    extracted_key = ("species", code, phrase, is_match)
                    extracted_span_counts[extracted_key] = int(extracted_span_counts.get(extracted_key, 0)) + 1
                    if is_match:
                        species_spans_by_code_matching_pattern[code] = int(
                            species_spans_by_code_matching_pattern.get(code, 0)
                        ) + 1
            if code in strain_codes:
                for loc in list(locs):
                    phrase = _span_phrase(tokens, str(loc))
                    is_match = bool(_SPEC_RE.match(phrase))
                    reserved_strains.add(phrase)
                    strain_spans_by_code_raw[code] = int(strain_spans_by_code_raw.get(code, 0)) + 1
                    extracted_key = ("strain", code, phrase, is_match)
                    extracted_span_counts[extracted_key] = int(extracted_span_counts.get(extracted_key, 0)) + 1

    extra_species = sorted([s for s in species_spans_all if _SPEC_RE.match(s)])
    reserved_strains_sorted = sorted(reserved_strains)

    by_key: dict[tuple[str, str], dict[str, set[str]]] = {}
    for sp in extra_species:
        parsed = _species_key_and_form(sp)
        if parsed is None:
            continue
        key, is_abbrev = parsed
        cur = by_key.setdefault(key, {"full_forms": set(), "abbreviations": set()})
        if is_abbrev:
            cur["abbreviations"].add(sp)
        else:
            cur["full_forms"].add(sp)

    species_key_index: list[dict] = []
    for key in sorted(by_key.keys(), key=lambda k: (k[0], k[1])):
        cur = by_key[key]
        full_forms = sorted(cur["full_forms"])
        abbreviations = sorted(cur["abbreviations"])
        orphan = bool((not full_forms) and abbreviations)
        representative = full_forms[0] if full_forms else (abbreviations[0] if abbreviations else "")
        species_key_index.append(
            {
                "key": [key[0], key[1]],
                "full_forms": full_forms,
                "abbreviations": abbreviations,
                "orphan_abbreviation": orphan,
                "representative": representative,
            }
        )

    extra_species_set = set(extra_species)
    reserved_strains_set = set(reserved_strains_sorted)
    extracted_spans: list[dict] = []
    for key in sorted(extracted_span_counts.keys(), key=lambda k: (k[0], k[1], k[2], int(k[3]))):
        source, code, phrase, is_match = key
        extracted_spans.append(
            {
                "source": source,
                "code": code,
                "phrase": phrase,
                "species_pattern_match": bool(is_match),
                "n_occurrences": int(extracted_span_counts[key]),
                "included_in_extra_species": bool(phrase in extra_species_set),
                "included_in_reserved_strains": bool(phrase in reserved_strains_set),
            }
        )

    return {
        "extra_species": extra_species,
        "reserved_strains": reserved_strains_sorted,
        "extracted_spans": extracted_spans,
        "species_key_index": species_key_index,
        "stats": {
            "n_species_spans_raw": len(species_spans_all),
            "n_extra_species": len(extra_species),
            "n_species_keys": len(species_key_index),
            "n_species_keys_with_full_form": len([1 for r in species_key_index if r["full_forms"]]),
            "n_species_keys_orphan_abbrev": len([1 for r in species_key_index if r["orphan_abbreviation"]]),
            "n_reserved_strains": len(reserved_strains_sorted),
            "species_codes": sorted(species_codes),
            "strain_codes": sorted(strain_codes),
            "species_spans_by_code_raw": dict(sorted(species_spans_by_code_raw.items(), key=lambda kv: kv[0])),
            "species_spans_by_code_matching_pattern": dict(
                sorted(species_spans_by_code_matching_pattern.items(), key=lambda kv: kv[0])
            ),
            "strain_spans_by_code_raw": dict(sorted(strain_spans_by_code_raw.items(), key=lambda kv: kv[0])),
            "n_extracted_span_entries": len(extracted_spans),
        },
    }
