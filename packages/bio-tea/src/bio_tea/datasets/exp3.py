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


def build_exp3_mix(
    *,
    ex_source: list[list[tuple[str, str]]],
    mr_source: list[dict],
    irr_red: float,
    rng_seed: int,
    source_variant: str,
) -> tuple[list[list[tuple[str, str]]], list[dict], dict]:
    """Build one Exp3 irrelevant-ablation train cell and return outputs + stats."""

    if len(ex_source) != len(mr_source):
        raise RuntimeError('Example/meta length mismatch in base set outputs')

    idx_rel = [i for i, r in enumerate(mr_source) if r.get('category') == 'relevant']
    idx_irr = [i for i, r in enumerate(mr_source) if r.get('category') != 'relevant']
    idx_irr_sorted = sorted(idx_irr, key=lambda i: str(mr_source[i].get('example_id') or ''))
    irr_keep = int(len(idx_irr_sorted) * (1.0 - (float(irr_red) / 100.0)))

    rng = random.Random(int(rng_seed))

    irr_idx = list(idx_irr_sorted)
    rng.shuffle(irr_idx)
    irr_kept = set(irr_idx[:irr_keep])

    items: list[tuple[str, list[tuple[str, str]], dict]] = []

    for i0 in idx_rel:
        r0 = dict(mr_source[i0])
        items.append((str(r0.get('example_id') or ''), ex_source[i0], r0))

    for i0 in sorted(list(irr_kept), key=lambda j: str(mr_source[j].get('example_id') or '')):
        r0 = dict(mr_source[i0])
        items.append((str(r0.get('example_id') or ''), ex_source[i0], r0))

    items.sort(key=lambda t: t[0])

    out_examples = [it[1] for it in items]
    out_meta = [it[2] for it in items]
    n_total = len(items)
    n_relevant = sum(1 for mr in out_meta if str(mr.get('category') or '') == 'relevant')
    n_irrelevant = n_total - n_relevant

    stats = {
        'source_variant': str(source_variant),
        'irrelevant_reduction_percent': float(irr_red),
        'kept_irrelevant': int(len(irr_kept)),
        'n_relevant': int(n_relevant),
        'n_irrelevant': int(n_irrelevant),
        'n_total_rows': int(n_total),
        'n_unique_example_ids': int(len({str(mr.get('example_id') or '') for mr in out_meta})),
        'n_removed_irrelevant': int(len(idx_irr_sorted) - len(irr_kept)),
    }
    return out_examples, out_meta, stats
