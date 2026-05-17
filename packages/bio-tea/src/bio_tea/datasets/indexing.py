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

from bio_tea.augment.labels import build_labels
from bio_tea.augment.words import extract_span_phrases

from .curation import spans_overlapping_window


@dataclass(frozen=True)
class ExampleIndexRecord:
    """A base example record produced from curated data.

    Designed to be serialized to JSONL.
    """

    example_id: str  # '{checksum}:{s},{e}'
    checksum: str
    s: int
    e: int
    text: str
    labels_raw: list[str]
    triggers: list[dict]
    entities: list[dict]
    scramble_phrases: list[str]

    def to_json(self) -> dict:
        return {
            'example_id': self.example_id,
            'checksum': self.checksum,
            'span': {'s': self.s, 'e': self.e},
            'text': self.text,
            'labels_raw': self.labels_raw,
            'triggers': self.triggers,
            'entities': self.entities,
            'scramble_phrases': self.scramble_phrases,
        }


def build_index_for_article(
    *,
    checksum: str,
    tokens: list[str],
    curation_data: Mapping[str, list[str]],
    maximise,
    scramble_tags: list[str],
    fit_num_tokens=None,
    fit_max_final_len: int | None = None,
) -> tuple[list[ExampleIndexRecord], list[str]]:
    """Build index records for a single article.

    Returns (records, labels_raw_for_full_article).

    maximise must accept (loc, tokens) and return (s, e).
    """

    labels_raw = build_labels(tokens, curation_data, l2l=None)

    # Determine scramble phrases for this article.
    scramble_words: set[str] = set()
    for key, locs in curation_data.items():
        tag_prefix = key.split('/', 1)[0]
        if tag_prefix in scramble_tags:
            scramble_words.update(extract_span_phrases(tokens, {tag_prefix: locs}))

    # Deterministic scramble phrase ordering.
    scramble_phrases = sorted(scramble_words, key=lambda s: (-len(s), s))

    def _fit_window_to_budget(s: int, e: int, focus_idx: int) -> tuple[int, int]:
        if fit_num_tokens is None or fit_max_final_len is None:
            return s, e
        max_len = int(fit_max_final_len)
        if max_len <= 0:
            return s, e

        while s < e and fit_num_tokens(' '.join(tokens[s : e + 1])) > max_len:
            left_margin = focus_idx - s
            right_margin = e - focus_idx
            if right_margin > left_margin:
                e -= 1
            else:
                s += 1
        return s, e

    # Compute candidate windows keyed by 's,e' and collect trigger provenance.
    pos_to_triggers: dict[str, list[dict]] = {}
    for key, locs in curation_data.items():
        for loc in locs:
            s0 = int(loc.split('+', 1)[0])
            s, e = maximise(s0, tokens)
            s, e = _fit_window_to_budget(s, e, s0)
            pos = f"{s},{e}"
            pos_to_triggers.setdefault(pos, []).append({'source_key': key, 'loc': loc, 'seed_index': s0})

    records: list[ExampleIndexRecord] = []
    for pos, triggers in pos_to_triggers.items():
        s, e = map(int, pos.split(',', 1))
        text = ' '.join(tokens[s : e + 1])
        labels_slice = labels_raw[s : e + 1]
        entities = spans_overlapping_window(curation_data, s=s, e=e, tokens=tokens)

        # Deterministic triggers ordering.
        triggers_sorted = sorted(triggers, key=lambda d: (d['seed_index'], d['source_key'], d['loc']))

        rec = ExampleIndexRecord(
            example_id=f"{checksum}:{pos}",
            checksum=checksum,
            s=s,
            e=e,
            text=text,
            labels_raw=labels_slice,
            triggers=triggers_sorted,
            entities=entities,
            scramble_phrases=scramble_phrases,
        )
        records.append(rec)

    records.sort(key=lambda r: (r.checksum, r.s, r.e))
    return records, labels_raw
