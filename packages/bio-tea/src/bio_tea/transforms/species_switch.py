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
"""Species-name switching transform."""

from __future__ import annotations

import re
from collections.abc import Callable

_SPEC_RE = re.compile(r"[A-Z](?:[a-z]+|\.)\s[a-z]+")


def _replace_keys_one_pass(text: str, replacements: dict[str, str]) -> str:
    """Apply substitutions against the original text in one pass.

    One-pass replacement prevents sequential replacement cascades.
    """

    if not replacements:
        return text
    keys = sorted(list(replacements.keys()), key=len, reverse=True)
    pat = re.compile("|".join(re.escape(k) for k in keys))
    return pat.sub(lambda m: replacements[m.group(0)], text)


def switch_species(text: str, *, all_species: set[str], sample: Callable[[], str]) -> str:
    """Replace recognized species mentions with sampled alternatives.

    Parameters
    ----------
    text:
        Input string.
    all_species:
        Set containing full binomials and abbreviated-genus forms.
    sample:
        Callable returning a replacement full binomial (e.g. "Escherichia coli").
    """
    verified: dict[str, object] = {}
    temp = _SPEC_RE.findall(text)
    for candidate in temp:
        if candidate in all_species:
            if candidate[1] == '.':
                if candidate not in verified:
                    verified[candidate] = True
            else:
                verified[candidate] = True
                verified[f"{candidate[0]}. {candidate.split(' ')[1]}"] = candidate

    for s in list(verified.keys()):
        if s[1] == '.':
            continue
        # Sample replacements from the configured species pool. Identity replacements
        # are possible when the sampled species equals the original mention.
        new_species = sample()
        verified[s] = new_species
        verified[f"{s[0]}. {s.split(' ')[1]}"] = f"{new_species[0]}. {new_species.split(' ')[1]}"

    for s in list(verified.keys()):
        if s[1] != '.':
            continue
        if verified[s] is not True:
            continue
        new_species = sample()
        verified[f"{s[0]}. {s.split(' ')[1]}"] = f"{new_species[0]}. {new_species.split(' ')[1]}"

    replacements: dict[str, str] = {}
    for k, v in verified.items():
        if isinstance(v, str):
            replacements[str(k)] = v
    return _replace_keys_one_pass(text, replacements)
