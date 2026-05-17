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
from typing import Iterable, Mapping


@dataclass(frozen=True)
class CuratedSpan:
    """A single curated span from TEA_curated_data.

    start and length are indices into whitespace tokens from load_article().
    label4 is the 4-letter code TEA derives from the tag prefix.
    """

    tag_prefix: str
    source_key: str
    start: int
    length: int
    label4: str

    @property
    def end_exclusive(self) -> int:
        return self.start + self.length


def parse_loc(loc: str) -> tuple[int, int]:
    """Parse a TEA curation location string 'start+len'."""
    s, l = loc.split('+', 1)
    return int(s), int(l)


def iter_curated_spans(curation_data: Mapping[str, list[str]]) -> Iterable[CuratedSpan]:
    """Iterate all curated spans in a checksum-level curation dict."""
    for key, locs in curation_data.items():
        tag_prefix = key.split('/', 1)[0]
        label4 = tag_prefix[:4].upper()
        for loc in locs:
            start, length = parse_loc(loc)
            yield CuratedSpan(
                tag_prefix=tag_prefix,
                source_key=key,
                start=start,
                length=length,
                label4=label4,
            )


def spans_overlapping_window(
    curation_data: Mapping[str, list[str]],
    *,
    s: int,
    e: int,
    tokens: list[str],
) -> list[dict]:
    """Collect provenance metadata for curated spans overlapping a window.

    Window is inclusive [s, e] over `tokens`.
    """
    out: list[dict] = []
    window_start = s
    window_end_excl = e + 1

    for sp in iter_curated_spans(curation_data):
        sp_start = sp.start
        sp_end_excl = sp.end_exclusive
        if sp_end_excl <= window_start or sp_start >= window_end_excl:
            continue

        surface = ' '.join(tokens[sp_start:sp_end_excl])
        out.append(
            {
                'tag': sp.tag_prefix,
                'label4': sp.label4,
                'source_key': sp.source_key,
                'start': sp_start,
                'length': sp.length,
                'end': sp_end_excl - 1,
                'surface': surface,
            }
        )

    # Deterministic ordering
    out.sort(key=lambda d: (d['start'], d['end'], d['source_key']))
    return out
