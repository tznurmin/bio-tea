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
import random
from statistics import median
from pathlib import Path
from typing import Mapping, Sequence

from .io import read_conll, read_jsonl


ANSI_RESET = "\x1b[0m"
ANSI_BOLD = "\x1b[1m"

_NO_SPACE_BEFORE = {
    ".",
    ",",
    ";",
    ":",
    "!",
    "?",
    "%",
    ")",
    "]",
    "}",
    "'s",
    "'re",
    "'ve",
    "'ll",
    "'d",
    "'m",
    "n't",
}
_NO_SPACE_AFTER = {"(", "[", "{", "$"}
_TYPE_COLORS = [
    "\x1b[38;5;39m",   # blue
    "\x1b[38;5;34m",   # green
    "\x1b[38;5;202m",  # orange
    "\x1b[38;5;199m",  # magenta
    "\x1b[38;5;45m",   # cyan
    "\x1b[38;5;220m",  # yellow
]


def infer_meta_path(set_path: Path) -> Path | None:
    p = Path(set_path)
    if p.suffix != ".set":
        return None
    cand = p.with_suffix(".meta.jsonl")
    return cand if cand.exists() else None


def load_examples_and_meta(set_path: Path, meta_path: Path | None = None) -> tuple[list[list[tuple[str, str]]], list[dict] | None]:
    examples = read_conll(Path(set_path))
    mp = Path(meta_path) if meta_path is not None else infer_meta_path(Path(set_path))
    meta = read_jsonl(mp) if mp is not None and mp.exists() else None
    return examples, meta


def label_type(label: str) -> str | None:
    lab = str(label)
    if lab == "O" or "-" not in lab:
        return None
    _pref, typ = lab.split("-", 1)
    return typ or None


def types_from_labels(labels: Sequence[str]) -> list[str]:
    out = sorted({t for t in (label_type(lab) for lab in labels) if t})
    return out


def _needs_space(prev_tok: str | None, tok: str) -> bool:
    if prev_tok is None:
        return False
    if tok in _NO_SPACE_BEFORE:
        return False
    if tok.startswith("'"):
        return False
    if prev_tok in _NO_SPACE_AFTER:
        return False
    if tok in {"-", "/", "–", "—"}:
        return False
    if prev_tok in {"-", "/", "–", "—"}:
        return False
    return True


def build_label_palette(labels: Sequence[str]) -> dict[str, str]:
    types = sorted({t for t in (label_type(lab) for lab in labels) if t})
    palette: dict[str, str] = {}
    for i, typ in enumerate(types):
        palette[typ] = _TYPE_COLORS[i % len(_TYPE_COLORS)]
    return palette


def render_example_text(
    tokens: Sequence[str],
    labels: Sequence[str],
    *,
    palette: Mapping[str, str] | None = None,
    color: bool = True,
) -> str:
    if len(tokens) != len(labels):
        raise ValueError("tokens and labels must have equal length")
    pal = dict(palette or build_label_palette(labels))
    out = ""
    prev_tok: str | None = None
    for tok, lab in zip(tokens, labels):
        typ = label_type(lab)
        styled = str(tok)
        if color and typ and typ in pal:
            if str(lab).startswith("B-"):
                styled = f"{ANSI_BOLD}{pal[typ]}{tok}{ANSI_RESET}"
            else:
                styled = f"{pal[typ]}{tok}{ANSI_RESET}"
        if _needs_space(prev_tok, str(tok)):
            out += " "
        out += styled
        prev_tok = str(tok)
    return out


def render_example_inline_labels(
    tokens: Sequence[str],
    labels: Sequence[str],
) -> str:
    """Render token stream as plain text and append inline IOB labels for non-O tokens."""

    if len(tokens) != len(labels):
        raise ValueError("tokens and labels must have equal length")
    out = ""
    prev_tok: str | None = None
    for tok, lab in zip(tokens, labels):
        tok_s = str(tok)
        lab_s = str(lab)
        piece = tok_s if lab_s == "O" else f"{tok_s} ({lab_s})"
        if _needs_space(prev_tok, tok_s):
            out += " "
        out += piece
        prev_tok = tok_s
    return out


def _group_keys_for_example(
    *,
    labels: Sequence[str],
    meta_row: Mapping[str, object] | None,
    group_by: str,
) -> list[str]:
    gb = str(group_by or "types_raw")
    if gb == "none":
        return ["all"]
    if gb == "category":
        if meta_row is not None and meta_row.get("category"):
            return [str(meta_row.get("category"))]
        return ["unknown"]
    if gb == "types_pooled":
        if meta_row is not None:
            vals = list(meta_row.get("types_pooled") or [])
            return [str(v) for v in vals] if vals else ["O-only"]
        vals = types_from_labels(labels)
        return vals if vals else ["O-only"]
    # default: types_raw
    if meta_row is not None:
        vals = list(meta_row.get("types_raw") or [])
        return [str(v) for v in vals] if vals else ["O-only"]
    vals = types_from_labels(labels)
    return vals if vals else ["O-only"]


def sample_examples_by_group(
    examples: list[list[tuple[str, str]]],
    meta_rows: list[dict] | None,
    *,
    group_by: str = "types_raw",
    samples_per_group: int = 1,
    seed: int = 0,
) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for i, ex in enumerate(examples):
        labels = [lab for _tok, lab in ex]
        mr = meta_rows[i] if meta_rows is not None and i < len(meta_rows) else None
        keys = _group_keys_for_example(labels=labels, meta_row=mr, group_by=group_by)
        for k in keys:
            groups.setdefault(k, []).append(i)

    rng = random.Random(int(seed))
    picked: dict[str, list[int]] = {}
    for k in sorted(groups.keys()):
        ids = sorted(groups[k])
        rng.shuffle(ids)
        n = max(int(samples_per_group), 0)
        picked[k] = sorted(ids[:n])
    return picked


def pair_examples_by_id(
    base_examples: list[list[tuple[str, str]]],
    base_meta_rows: list[dict] | None,
    other_examples: list[list[tuple[str, str]]],
    other_meta_rows: list[dict] | None,
) -> tuple[list[dict], dict]:
    """Align two example sets by example_id for semantic diff inspection."""

    if base_meta_rows is None or other_meta_rows is None:
        raise ValueError("Diff mode requires metadata for both sets")
    if len(base_examples) != len(base_meta_rows):
        raise ValueError("base example/meta length mismatch")
    if len(other_examples) != len(other_meta_rows):
        raise ValueError("other example/meta length mismatch")

    def _build_index(meta_rows: list[dict], side: str) -> dict[str, int]:
        idx: dict[str, int] = {}
        for i, mr in enumerate(meta_rows):
            eid = str(mr.get("example_id") or "")
            if not eid:
                raise ValueError(f"{side} metadata missing example_id at row {i}")
            if eid in idx:
                raise ValueError(f"{side} metadata has duplicate example_id: {eid}")
            idx[eid] = i
        return idx

    base_idx = _build_index(base_meta_rows, "base")
    other_idx = _build_index(other_meta_rows, "other")

    base_ids = set(base_idx.keys())
    other_ids = set(other_idx.keys())
    common_ids = sorted(base_ids.intersection(other_ids))
    base_only = sorted(base_ids - other_ids)
    other_only = sorted(other_ids - base_ids)

    pairs: list[dict] = []
    n_changed = 0
    n_changed_tokens = 0
    n_changed_labels = 0
    for eid in common_ids:
        bi = int(base_idx[eid])
        oi = int(other_idx[eid])
        bex = list(base_examples[bi])
        oex = list(other_examples[oi])
        btoks = [t for t, _l in bex]
        blabs = [l for _t, l in bex]
        otoks = [t for t, _l in oex]
        olabs = [l for _t, l in oex]
        changed_tokens = btoks != otoks
        changed_labels = blabs != olabs
        changed = changed_tokens or changed_labels
        if changed_tokens:
            n_changed_tokens += 1
        if changed_labels:
            n_changed_labels += 1
        if changed:
            n_changed += 1
        pairs.append(
            {
                "example_id": eid,
                "base_index": bi,
                "other_index": oi,
                "base_example": bex,
                "other_example": oex,
                "base_meta": dict(base_meta_rows[bi]),
                "other_meta": dict(other_meta_rows[oi]),
                "changed_tokens": bool(changed_tokens),
                "changed_labels": bool(changed_labels),
                "changed": bool(changed),
            }
        )

    stats = {
        "n_base": int(len(base_examples)),
        "n_other": int(len(other_examples)),
        "n_common": int(len(common_ids)),
        "n_base_only": int(len(base_only)),
        "n_other_only": int(len(other_only)),
        "n_changed": int(n_changed),
        "n_changed_tokens": int(n_changed_tokens),
        "n_changed_labels": int(n_changed_labels),
        "base_only_example_ids": base_only,
        "other_only_example_ids": other_only,
    }
    return pairs, stats


def summarize_examples(
    examples: list[list[tuple[str, str]]],
    meta_rows: list[dict] | None = None,
) -> dict:
    """Summarize label/category distributions for a materialized example set."""

    n_examples = int(len(examples))
    token_lengths = [int(len(ex)) for ex in examples]
    n_tokens = int(sum(token_lengths))
    label_counts: dict[str, int] = {}
    label_prefix_counts: dict[str, int] = {}
    label_type_token_counts: dict[str, int] = {}
    label_type_example_counts: dict[str, int] = {}

    for ex in examples:
        seen_types: set[str] = set()
        for _tok, lab in ex:
            lab_s = str(lab)
            label_counts[lab_s] = int(label_counts.get(lab_s, 0)) + 1
            if lab_s == "O":
                label_prefix_counts["O"] = int(label_prefix_counts.get("O", 0)) + 1
                continue
            if "-" in lab_s:
                pref, typ = lab_s.split("-", 1)
                label_prefix_counts[pref] = int(label_prefix_counts.get(pref, 0)) + 1
                if typ:
                    label_type_token_counts[typ] = int(label_type_token_counts.get(typ, 0)) + 1
                    seen_types.add(typ)
            else:
                label_prefix_counts["UNKNOWN"] = int(label_prefix_counts.get("UNKNOWN", 0)) + 1
        for typ in seen_types:
            label_type_example_counts[typ] = int(label_type_example_counts.get(typ, 0)) + 1

    category_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    forced_relevant_counts: dict[str, int] = {"true": 0, "false": 0}
    types_raw_counts: dict[str, int] = {}
    types_pooled_counts: dict[str, int] = {}
    n_meta = int(len(meta_rows)) if meta_rows is not None else None
    meta_length_match = (meta_rows is None) or (len(meta_rows) == len(examples))

    if meta_rows is not None:
        for mr in meta_rows:
            cat = str(mr.get("category") or "unknown")
            category_counts[cat] = int(category_counts.get(cat, 0)) + 1
            reason = str(mr.get("reason") or "unknown")
            reason_counts[reason] = int(reason_counts.get(reason, 0)) + 1
            forced = bool(mr.get("forced_relevant", False))
            forced_relevant_counts["true" if forced else "false"] = int(
                forced_relevant_counts.get("true" if forced else "false", 0)
            ) + 1
            for t in list(mr.get("types_raw") or []):
                types_raw_counts[str(t)] = int(types_raw_counts.get(str(t), 0)) + 1
            for t in list(mr.get("types_pooled") or []):
                types_pooled_counts[str(t)] = int(types_pooled_counts.get(str(t), 0)) + 1

    return {
        "n_examples": n_examples,
        "n_meta": n_meta,
        "meta_length_match": bool(meta_length_match),
        "n_tokens_total": n_tokens,
        "tokens_per_example": {
            "min": int(min(token_lengths)) if token_lengths else 0,
            "max": int(max(token_lengths)) if token_lengths else 0,
            "median": int(median(token_lengths)) if token_lengths else 0,
        },
        "label_counts": dict(sorted(label_counts.items(), key=lambda kv: kv[0])),
        "label_prefix_counts": dict(sorted(label_prefix_counts.items(), key=lambda kv: kv[0])),
        "label_type_token_counts": dict(sorted(label_type_token_counts.items(), key=lambda kv: kv[0])),
        "label_type_example_counts": dict(sorted(label_type_example_counts.items(), key=lambda kv: kv[0])),
        "category_counts": dict(sorted(category_counts.items(), key=lambda kv: kv[0])),
        "reason_counts": dict(sorted(reason_counts.items(), key=lambda kv: kv[0])),
        "forced_relevant_counts": dict(sorted(forced_relevant_counts.items(), key=lambda kv: kv[0])),
        "types_raw_counts": dict(sorted(types_raw_counts.items(), key=lambda kv: kv[0])),
        "types_pooled_counts": dict(sorted(types_pooled_counts.items(), key=lambda kv: kv[0])),
    }


def validate_example_set(
    examples: list[list[tuple[str, str]]],
    meta_rows: list[dict] | None = None,
    *,
    max_tokens: int | None = None,
    required_categories: Sequence[str] | None = None,
    required_raw_types: Sequence[str] | None = None,
) -> dict:
    base = summarize_examples(examples, meta_rows)
    errors: list[str] = []
    n_examples = int(base.get("n_examples", len(examples)))
    if meta_rows is not None and len(meta_rows) != n_examples:
        errors.append(f"example_meta_length_mismatch:{n_examples}!={len(meta_rows)}")

    token_label_mismatch = 0
    for i, ex in enumerate(examples):
        toks = [t for t, _l in ex]
        labs = [l for _t, l in ex]
        if len(toks) != len(labs):
            token_label_mismatch += 1
            errors.append(f"token_label_length_mismatch:{i}")
    max_obs = int((base.get("tokens_per_example") or {}).get("max", 0))
    if max_tokens is not None and int(max_obs) > int(max_tokens):
        errors.append(f"max_tokens_exceeded:{int(max_obs)}>{int(max_tokens)}")

    category_counts = dict(base.get("category_counts") or {})
    raw_type_counts = dict(base.get("types_raw_counts") or {})
    if not category_counts and meta_rows is None:
        # Fallback from labels when metadata is missing.
        for ex in examples:
            types = types_from_labels([lab for _tok, lab in ex])
            if not types:
                category_counts["O-only"] = int(category_counts.get("O-only", 0)) + 1
            for t in types:
                raw_type_counts[t] = int(raw_type_counts.get(t, 0)) + 1

    for cat in sorted({str(c) for c in (required_categories or []) if str(c)}):
        if int(category_counts.get(cat, 0)) <= 0:
            errors.append(f"missing_required_category:{cat}")

    for typ in sorted({str(t) for t in (required_raw_types or []) if str(t)}):
        if int(raw_type_counts.get(typ, 0)) <= 0:
            errors.append(f"missing_required_raw_type:{typ}")

    return {
        **base,
        "max_tokens_observed": int(max_obs),
        "token_label_mismatches": int(token_label_mismatch),
        "errors": list(errors),
        "ok": len(errors) == 0,
    }


def dumps_summary(obj: Mapping[str, object]) -> str:
    return json.dumps(obj, indent=2, sort_keys=True)
