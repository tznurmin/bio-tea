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
"""Utilities for extracting word/phrase lists from curated spans."""

from __future__ import annotations

from collections.abc import Mapping


def extract_span_phrases(tokens: list[str], curation_subset: Mapping[str, list[str]]) -> set[str]:
    """Return set of ' '.join(tokens[start:start+len]) for each curated location."""
    word_list: set[str] = set()
    for _, locs in curation_subset.items():
        for loc in locs:
            s, e = loc.split('+')
            s_i = int(s)
            e_i = int(e)
            word_list.add(' '.join(tokens[s_i:s_i + e_i]))
    return word_list
