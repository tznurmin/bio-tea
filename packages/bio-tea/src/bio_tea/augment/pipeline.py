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
"""Token-level entity augmentation pipeline."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from .labels import build_labels
from .words import extract_span_phrases


def augment_tokens(
    tokens: list[str],
    curation_data: Mapping[str, list[str]],
    *,
    maximise: Callable[[int, list[str]], tuple[int, int]],
    switch: Callable[[str], str],
    scramble: Callable[..., str],
    num_tokens: Callable[[str], int],
    max_final_len: int,
    scramble_tags: list[str] | None = None,
    l2l: dict[str, str] | None = None,
) -> dict:
    scramble_tags = scramble_tags or []

    # Scramble wordlist must be deterministic; using a set directly makes output
    # depend on hash iteration order.
    words: set[str] = set()
    results: dict = {
        'original': {},
        'scrambled': {},
        'switched': {},
        'all': {},
    }

    # Labels are constructed once over the full token sequence.
    labels = build_labels(tokens, curation_data, l2l=l2l)

    # Determine which phrases are eligible for scrambling.
    for t, locations in curation_data.items():
        t0 = t.split('/')[0]
        if t0 in scramble_tags:
            words.update(extract_span_phrases(tokens, {t0: locations}))

        for loc in locations:
            s0 = int(loc.split('+')[0])
            s, e = maximise(s0, tokens)
            results['original'][f"{s},{e}"] = ' '.join(tokens[s:e + 1])

    results['labels'] = labels

    # Deterministic ordering for scramble: longer phrases first avoids partial
    # replacements when phrases overlap.
    words_list = sorted(words, key=lambda s: (-len(s), s))

    for pos, window_text in results['original'].items():
        transformations = {
            'switched': [switch],
            'scrambled': [scramble],
            'all': [switch, scramble],
        }
        extra_params = {scramble: [words_list]}

        for out_key, tfs in transformations.items():
            current = window_text

            for idx, tf in enumerate(tfs):
                base = current
                tries = 5

                while tries > 0:
                    params = extra_params.get(tf, [])
                    candidate = tf(base, *params)
                    if num_tokens(candidate) <= max_final_len:
                        current = candidate
                        break
                    tries -= 1

                if tries == 0:
                    break

                if idx == len(tfs) - 1:
                    results[out_key][pos] = current

    return results
