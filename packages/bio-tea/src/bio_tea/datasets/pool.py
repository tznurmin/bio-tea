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


def pool_bio(labels: list[str], l2l: Mapping[str, str] | None) -> list[str]:
    """Pool labels with l2l.

    By default, l2l preserves all labels. If l2l is provided, labels absent
    from the mapping are assigned 'O'.
    """
    if not l2l:
        return list(labels)

    pooled: list[str] = []
    for lab in labels:
        if lab == 'O':
            pooled.append('O')
            continue
        # Expected BIO label with a prefix and label type.
        if '-' not in lab:
            pooled.append('O')
            continue
        pref, typ = lab.split('-', 1)
        new_typ = l2l.get(typ, 'O')
        if new_typ == 'O':
            pooled.append('O')
        else:
            pooled.append(f"{pref}-{new_typ}")
    return pooled


def bio_types(labels: list[str]) -> set[str]:
    """Return the set of type codes present in a BIO label list (excluding O)."""
    out: set[str] = set()
    for lab in labels:
        if lab == 'O':
            continue
        if '-' not in lab:
            continue
        _, typ = lab.split('-', 1)
        out.add(typ)
    return out
