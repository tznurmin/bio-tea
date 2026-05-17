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

from typing import Mapping

from .classify import classify_example
from .pool import pool_bio
from .species import species_keys_in_text


def _sorted_unique_phrases(values: list[str] | set[str]) -> list[str]:
    out = {str(v).strip() for v in values if str(v).strip()}
    return sorted(out, key=lambda s: (-len(s), s))


def _phrases_present_in_text(text: str, phrases: list[str]) -> list[str]:
    hay = f" {str(text or '').strip()} "
    out: set[str] = set()
    for p in list(phrases or []):
        phrase = str(p).strip()
        if not phrase:
            continue
        if f" {phrase} " in hay:
            out.add(phrase)
    return _sorted_unique_phrases(out)


def _applied_scramble_outputs(*, base_text: str, mapping: dict[str, str] | None) -> list[str]:
    if not mapping:
        return []
    hay = f" {str(base_text or '').strip()} "
    out: set[str] = set()
    for src, dst in dict(mapping).items():
        src_s = str(src).strip()
        dst_s = str(dst).strip()
        if not src_s or not dst_s:
            continue
        if f" {src_s} " in hay:
            out.add(dst_s)
    return _sorted_unique_phrases(out)


def _apply_with_retries(
    *,
    base: str,
    fn,
    args: list,
    num_tokens,
    max_final_len: int,
    tries: int,
) -> str | None:
    """Apply a transform with budget retries. Returns None on failure."""
    remaining = tries
    while remaining > 0:
        try:
            cand = fn(base, *args)
        except Exception:
            return None
        if num_tokens(cand) <= max_final_len:
            return cand
        remaining -= 1
    return None


def _apply_with_retries_with_state(
    *,
    base: str,
    fn,
    args: list,
    num_tokens,
    max_final_len: int,
    tries: int,
    rng,
) -> tuple[str | None, object | None]:
    """Apply a transform with budget retries and return successful RNG pre-state."""

    remaining = tries
    while remaining > 0:
        state = rng.getstate() if rng is not None else None
        try:
            cand = fn(base, *args)
        except Exception:
            return None, None
        if num_tokens(cand) <= max_final_len:
            return cand, state
        remaining -= 1
    return None, None


def _apply_scramble_with_retries_with_state(
    *,
    base: str,
    tea,
    phrases: list[str],
    num_tokens,
    max_final_len: int,
    tries: int,
    rng,
    reserved_phrases: set[str] | None,
    ensure_unique_scramble_outputs: bool,
) -> tuple[str | None, object | None, dict[str, str] | None, str | None]:
    """Scramble with optional uniqueness tracking and detailed failure reason."""

    remaining = tries
    seen_budget_fail = False
    seen_reserved_exhausted = False
    while remaining > 0:
        state = rng.getstate() if rng is not None else None
        try:
            if ensure_unique_scramble_outputs:
                res = tea.scramble(
                    base,
                    phrases,
                    reserved_phrases=reserved_phrases,
                    return_mapping=True,
                )
                cand, mapping, exhausted = res
                if exhausted:
                    seen_reserved_exhausted = True
                    remaining -= 1
                    continue
            else:
                res = tea.scramble(
                    base,
                    phrases,
                    reserved_phrases=reserved_phrases,
                    return_mapping=True,
                )
                if isinstance(res, tuple) and len(res) == 3:
                    cand, mapping, _exhausted = res
                else:
                    cand = res
                    mapping = {}
        except Exception:
            return None, None, None, 'scramble_error'

        if num_tokens(cand) <= max_final_len:
            return cand, state, mapping, None

        seen_budget_fail = True
        remaining -= 1

    if seen_reserved_exhausted and not seen_budget_fail:
        return None, None, None, 'scramble_reserved_exhausted'
    if seen_budget_fail and not seen_reserved_exhausted:
        return None, None, None, 'scramble_budget_exceeded'
    if seen_budget_fail and seen_reserved_exhausted:
        return None, None, None, 'scramble_reserved_or_budget_exhausted'
    return None, None, None, 'scramble_failed'


def materialize_variant_bundle(
    records: list[dict],
    *,
    tea,
    variants: list[str],
    l2l: Mapping[str, str] | None,
    negative_labels: set[str] | None,
    max_final_len: int,
    switch_pool: list[str] | None = None,
    enforce_transform_alignment: bool = True,
    keep_none_without_all: bool = True,
    ensure_unique_scramble_outputs: bool = False,
    forced_species_by_example_id: Mapping[str, str] | None = None,
    return_stats: bool = False,
) -> dict[str, tuple[list[list[tuple[str, str]]], list[dict]]] | tuple[
    dict[str, tuple[list[list[tuple[str, str]]], list[dict]]],
    dict[str, dict],
]:
    """Materialize several variants in one pass with coupled transform decisions."""

    allowed = {'none', 'species', 'strains', 'all'}
    req = list(dict.fromkeys(variants))
    for v in req:
        if v not in allowed:
            raise ValueError(f"Unknown variant: {v}")

    examples_by_variant: dict[str, list[list[tuple[str, str]]]] = {v: [] for v in req}
    meta_by_variant: dict[str, list[dict]] = {v: [] for v in req}
    stats_by_variant: dict[str, dict] = {
        v: {
            'input_records': 0,
            'final_examples': 0,
            'dropped_total': 0,
            'drop_reasons': {},
        }
        for v in req
    }

    def _drop(v: str, reason: str) -> None:
        sv = stats_by_variant[v]
        sv['dropped_total'] += 1
        dr = sv['drop_reasons']
        dr[reason] = int(dr.get(reason, 0)) + 1
    requested = set(req)
    requested_transformed = [v for v in req if v in {'species', 'strains', 'all'}]
    reserved_dynamic: set[str] = set(getattr(tea, 'reserved_strains', set()))
    forced_species_by_example_id = dict(forced_species_by_example_id or {})

    # Stable ordering is required because augmentation uses stateful randomness.
    records_sorted = sorted(records, key=lambda r: (r['example_id']))
    if ensure_unique_scramble_outputs:
        for r0 in records_sorted:
            for p in (r0.get('scramble_phrases') or []):
                if p:
                    reserved_dynamic.add(str(p))

    for r in records_sorted:
        for v in req:
            stats_by_variant[v]['input_records'] += 1

        base_text: str = r['text']
        labels_raw: list[str] = r['labels_raw']
        clf = classify_example(labels_raw, l2l=l2l, negative_labels=negative_labels)
        labels_pooled = pool_bio(labels_raw, l2l)

        variant_texts: dict[str, str | None] = {'none': base_text}

        # When transform alignment is enabled, the `strains` branch is coupled
        # to the species-switched path so `all` and `strains` stay decision-
        # aligned for the same example instead of drifting independently.
        need_switch = ('species' in requested) or ('all' in requested) or ('strains' in requested and enforce_transform_alignment)
        switched_text: str | None = None
        switch_fail_reason: str | None = None
        forced_species = str(forced_species_by_example_id.get(str(r.get('example_id') or ''), '') or '').strip()
        if need_switch:
            if switch_pool is not None and len(switch_pool) == 0:
                switched_text = None
                switch_fail_reason = 'switch_pool_empty'
            else:
                local_pool = [forced_species] if forced_species else switch_pool
                switched_text, _ = _apply_with_retries_with_state(
                    base=base_text,
                    fn=lambda s: tea.switch(s, species_pool=local_pool),
                    args=[],
                    num_tokens=tea.num_tokens,
                    max_final_len=max_final_len,
                    tries=5,
                    rng=getattr(tea, 'rng', None),
                )
                if switched_text is None:
                    switch_fail_reason = 'switch_budget_or_error'

        all_text: str | None = None
        scramble_state: object | None = None
        all_mapping: dict[str, str] | None = None
        all_scramble_reason: str | None = None
        reserved_for_all: set[str] | None = None
        strains_mapping: dict[str, str] | None = None
        need_all_scramble = ('all' in requested) or ('strains' in requested and enforce_transform_alignment)
        if need_all_scramble and switched_text is not None:
            phrases = r.get('scramble_phrases') or []
            reserved_for_all = (
                set(reserved_dynamic)
                if ensure_unique_scramble_outputs
                else set(getattr(tea, 'reserved_strains', set()))
            )
            all_text, scramble_state, all_mapping, all_scramble_reason = _apply_scramble_with_retries_with_state(
                base=switched_text,
                tea=tea,
                phrases=phrases,
                num_tokens=tea.num_tokens,
                max_final_len=max_final_len,
                tries=5,
                rng=getattr(tea, 'rng', None),
                reserved_phrases=reserved_for_all,
                ensure_unique_scramble_outputs=ensure_unique_scramble_outputs,
            )
            if all_text is not None and ensure_unique_scramble_outputs and all_mapping:
                reserved_dynamic.update(all_mapping.values())

        species_text: str | None = switched_text if ('species' in requested) else None

        strains_text: str | None = None
        strains_scramble_reason: str | None = None
        if 'strains' in requested:
            phrases = r.get('scramble_phrases') or []
            if enforce_transform_alignment:
                if scramble_state is not None and hasattr(tea, 'rng'):
                    saved = tea.rng.getstate()
                    tea.rng.setstate(scramble_state)
                    if ensure_unique_scramble_outputs:
                        cand, mapping, exhausted = tea.scramble(
                            base_text,
                            phrases,
                            reserved_phrases=reserved_for_all or set(reserved_dynamic),
                            return_mapping=True,
                        )
                        if exhausted:
                            cand = None
                        elif cand is not None and mapping:
                            reserved_dynamic.update(mapping.values())
                    else:
                        res = tea.scramble(
                            base_text,
                            phrases,
                            reserved_phrases=set(getattr(tea, 'reserved_strains', set())),
                            return_mapping=True,
                        )
                        if isinstance(res, tuple) and len(res) == 3:
                            cand, mapping, _exhausted = res
                        else:
                            cand = res
                            mapping = {}
                    tea.rng.setstate(saved)
                    if cand is not None and tea.num_tokens(cand) <= max_final_len:
                        strains_text = cand
                        strains_mapping = dict(mapping or {})
                    elif cand is None:
                        strains_scramble_reason = 'scramble_reserved_exhausted'
                    else:
                        strains_scramble_reason = 'scramble_budget_exceeded'
            else:
                strains_text, _s_state, strains_mapping, strains_scramble_reason = _apply_scramble_with_retries_with_state(
                    base=base_text,
                    tea=tea,
                    phrases=phrases,
                    num_tokens=tea.num_tokens,
                    max_final_len=max_final_len,
                    tries=5,
                    rng=getattr(tea, 'rng', None),
                    reserved_phrases=reserved_dynamic if ensure_unique_scramble_outputs else set(getattr(tea, 'reserved_strains', set())),
                    ensure_unique_scramble_outputs=ensure_unique_scramble_outputs,
                )
                if strains_text is not None and ensure_unique_scramble_outputs and strains_mapping:
                    reserved_dynamic.update(strains_mapping.values())

        if 'species' in requested:
            variant_texts['species'] = species_text
        if 'strains' in requested:
            variant_texts['strains'] = strains_text
        if 'all' in requested:
            variant_texts['all'] = all_text

        drop_reason: dict[str, str | None] = {v: None for v in req}
        if 'species' in requested and species_text is None:
            if need_switch and switched_text is None:
                drop_reason['species'] = switch_fail_reason or 'switch_budget_exceeded'
            else:
                drop_reason['species'] = 'transform_unavailable'

        if 'all' in requested and all_text is None:
            if need_switch and switched_text is None:
                drop_reason['all'] = switch_fail_reason or 'switch_budget_exceeded'
            elif need_all_scramble:
                drop_reason['all'] = all_scramble_reason or 'scramble_budget_exceeded'
            else:
                drop_reason['all'] = 'transform_unavailable'

        if 'strains' in requested and strains_text is None:
            if not enforce_transform_alignment:
                drop_reason['strains'] = strains_scramble_reason or 'scramble_budget_exceeded'
            elif need_switch and switched_text is None:
                drop_reason['strains'] = switch_fail_reason or 'switch_budget_exceeded'
            elif need_all_scramble and all_text is None:
                drop_reason['strains'] = strains_scramble_reason or all_scramble_reason or 'scramble_budget_exceeded'
            else:
                drop_reason['strains'] = 'transform_alignment_dropped'

        variant_tokens: dict[str, list[str] | None] = {}
        for v in req:
            txt = variant_texts.get(v)
            if txt is None:
                variant_tokens[v] = None
                continue
            toks = txt.split()
            if len(toks) == len(labels_pooled):
                variant_tokens[v] = toks
            else:
                variant_tokens[v] = None
                drop_reason[v] = 'token_label_length_mismatch'

        if enforce_transform_alignment and requested_transformed:
            transformed_ok = all(variant_tokens.get(v) is not None for v in requested_transformed)
            if not transformed_ok:
                for v in requested_transformed:
                    if variant_tokens[v] is not None:
                        drop_reason[v] = 'transform_alignment_dropped'
                    variant_tokens[v] = None

        # In paired mode, keep `none` only when the corresponding `all`
        # variant is also available.
        if (not keep_none_without_all) and ('none' in requested) and ('all' in requested):
            if variant_tokens.get('none') is not None and variant_tokens.get('all') is None:
                variant_tokens['none'] = None
                drop_reason['none'] = 'dropped_without_all_pair'

        for v in req:
            toks = variant_tokens.get(v)
            if toks is None:
                _drop(v, drop_reason.get(v) or 'unknown')
                continue
            out_text = ' '.join(toks)
            strain_original = _phrases_present_in_text(out_text, list(r.get('scramble_phrases') or []))
            if v == 'all':
                strain_scrambled = _applied_scramble_outputs(base_text=switched_text or base_text, mapping=all_mapping)
            elif v == 'strains':
                strain_scrambled = _applied_scramble_outputs(base_text=base_text, mapping=strains_mapping)
            else:
                strain_scrambled = []
            strain_union = _sorted_unique_phrases(strain_original + strain_scrambled)
            examples_by_variant[v].append(list(zip(toks, labels_pooled)))
            stats_by_variant[v]['final_examples'] += 1
            meta_by_variant[v].append(
                {
                    'example_id': r['example_id'],
                    'checksum': r['checksum'],
                    'span': r['span'],
                    'variant': v,
                    'category': clf.category,
                    'forced_relevant': clf.forced_relevant,
                    'reason': clf.reason,
                    'types_raw': clf.types_raw,
                    'types_pooled': clf.types_pooled,
                    'entities': r.get('entities', []),
                    'triggers': r.get('triggers', []),
                    'token_count': tea.num_tokens(out_text),
                    'species_keys': sorted(
                        [list(k) for k in species_keys_in_text(out_text, all_species=tea.all_species)]
                    ),
                    'strain_phrases_original': strain_original,
                    'strain_phrases_scrambled': strain_scrambled,
                    'strain_phrases': strain_union,
                    'species_forced_target': forced_species or None,
                }
            )

    bundle = {v: (examples_by_variant[v], meta_by_variant[v]) for v in req}
    if return_stats:
        return bundle, stats_by_variant
    return bundle


def materialize_records(
    records: list[dict],
    *,
    tea,
    variant: str,
    l2l: Mapping[str, str] | None,
    negative_labels: set[str] | None,
    max_final_len: int,
    switch_pool: list[str] | None = None,
) -> tuple[list[list[tuple[str, str]]], list[dict]]:
    """Materialize a dataset variant.

    Parameters
    ----------
    records:
        JSON-compatible dict records (from ExampleIndexRecord.to_json()).
    variant:
        One of: 'none', 'species', 'strains', 'all'.

    Returns
    -------
    examples:
        List of CoNLL examples (token,label) pairs.
    meta_rows:
        List of JSONL rows with provenance and classification metadata.

    Notes
    -----
    - Categorization is performed on raw labels.
    - Pooling is applied after categorization.
    - Transform retry logic mirrors tea.augment pipeline (non-compounding).
    """

    bundle = materialize_variant_bundle(
        records,
        tea=tea,
        variants=[variant],
        l2l=l2l,
        negative_labels=negative_labels,
        max_final_len=max_final_len,
        switch_pool=switch_pool,
        enforce_transform_alignment=False,
    )
    return bundle[variant]
