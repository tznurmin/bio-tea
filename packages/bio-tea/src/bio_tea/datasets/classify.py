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

from dataclasses import dataclass
from typing import Mapping

from .pool import bio_types


@dataclass(frozen=True)
class Classification:
    category: str  # 'relevant' | 'irrelevant'
    forced_relevant: bool
    reason: str
    types_raw: list[str]
    types_pooled: list[str]


def classify_example(
    labels_raw: list[str],
    *,
    l2l: Mapping[str, str] | None,
    negative_labels: set[str] | None = None,
) -> Classification:
    """Classify an example as relevant/irrelevant.

    Rules implemented:

    - Pooling is not applied to the returned labels; this function only uses raw
      BIO labels for classification.
    - negative_labels can be provided in pooled and/or raw type space.
    - Raw-label fallback:
        if pooled labels contain no non-negative type, but raw labels contain a
        non-negative type that was pooled away (for example to `O`), the example
        is still relevant.
    - Forced relevant rule:
        if a negative type and another non-negative type coexist (pooled or raw
        fallback view), category must be relevant.
    - Default behavior:
        relevant if there exists any non-O pooled label that is not negative.
        otherwise irrelevant.

    The coexistence of negative and non-negative entity types forces relevance.
    """

    negative_labels = negative_labels or set()

    raw_types = bio_types(labels_raw)

    # Apply mapping to raw types for pooled-view decisions.
    if l2l:
        mapped_raw_types = {l2l.get(t, 'O') for t in raw_types}
        pooled_types = set(mapped_raw_types)
        pooled_types.discard('O')
    else:
        mapped_raw_types = set(raw_types)
        pooled_types = set(raw_types)

    def _is_negative_raw(raw_t: str) -> bool:
        if raw_t in negative_labels:
            return True
        pooled_t = l2l.get(raw_t, 'O') if l2l else raw_t
        return pooled_t in negative_labels

    has_negative = any(t in negative_labels for t in pooled_types)
    has_other = any((t not in negative_labels) for t in pooled_types)

    forced = has_negative and has_other
    if forced:
        return Classification(
            category='relevant',
            forced_relevant=True,
            reason='negative_and_other',
            types_raw=sorted(raw_types),
            types_pooled=sorted(pooled_types),
        )

    # Default: relevant if any non-negative label exists.
    if has_other:
        return Classification(
            category='relevant',
            forced_relevant=False,
            reason='has_non_negative_label',
            types_raw=sorted(raw_types),
            types_pooled=sorted(pooled_types),
        )

    # Raw fallback view: preserve semantic relevance even if pooling maps types to O.
    raw_has_negative = any(_is_negative_raw(t) for t in raw_types)
    raw_has_other = any((not _is_negative_raw(t)) for t in raw_types)
    if raw_has_negative and raw_has_other:
        return Classification(
            category='relevant',
            forced_relevant=True,
            reason='negative_and_raw_other',
            types_raw=sorted(raw_types),
            types_pooled=sorted(pooled_types),
        )
    if raw_has_other:
        return Classification(
            category='relevant',
            forced_relevant=False,
            reason='has_raw_label_only',
            types_raw=sorted(raw_types),
            types_pooled=sorted(pooled_types),
        )

    return Classification(
        category='irrelevant',
        forced_relevant=False,
        reason='no_non_negative_labels',
        types_raw=sorted(raw_types),
        types_pooled=sorted(pooled_types),
    )
