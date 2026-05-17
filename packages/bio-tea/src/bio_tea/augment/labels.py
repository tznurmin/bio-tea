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
"""Label construction for TEA augmentation.

Input curation data uses:
- curation_data: dict[str, list[str]]
- location strings in "start+len" token-index format
"""

from __future__ import annotations

from collections.abc import Mapping


def build_labels(tokens: list[str], curation_data: Mapping[str, list[str]], l2l: dict[str, str] | None = None) -> list[str]:
    labels = ['O'] * len(tokens)

    for t, locations in curation_data.items():
        t0 = t.split('/')[0]
        label = t0[0:4].upper()
        if l2l is not None:
            label = l2l.get(label, 'O')

        if label == 'O':
            continue

        for loc in locations:
            t_sta, t_len = loc.split('+')
            t_sta_i = int(t_sta)
            t_len_i = int(t_len)
            for idx in range(t_sta_i, t_sta_i + t_len_i):
                if idx == t_sta_i:
                    labels[idx] = f"B-{label}"
                else:
                    labels[idx] = f"I-{label}"

    return labels
