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
"""Strain-name scrambling transform."""

from __future__ import annotations

import random
import re
from collections.abc import Iterable


def scramble_text(
    text: str,
    wordlist: Iterable[str],
    *,
    force_diff: bool = False,
    skipped_chars: set[str] | None = None,
    conserved: list[str] | None = None,
    reserved_phrases: set[str] | None = None,
    max_reserved_retries: int = 5,
    return_mapping: bool = False,
    rng=random,
) -> str | tuple[str, dict[str, str], bool]:
    base_text = text
    skipped_chars = skipped_chars or set(['δ', 'Δ'])
    conserved = conserved or [
        'strain', 'subsp', 'subspecies', 'isolate', 'pathovar', 'serovar', 'serotype',
        'genotype', 'ecotype', 'sequence', 'mutant', 'wild-type', 'complementation',
        'complemented', 'pv', 'wt', 'type', 'sp'
    ]

    reserved = set([p for p in (reserved_phrases or set()) if p])
    attempts = 0
    while True:
        jt: dict[str, str] = {}

        numbers = sorted(list(set(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']) - skipped_chars))
        downcase_v = set(['a', 'e', 'i', 'o', 'u', 'y'])
        downcase_c = set(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']) - downcase_v

        upcase_v = sorted(list(set([c.upper() for c in downcase_v]) - skipped_chars))
        upcase_c = sorted(list(set([c.upper() for c in downcase_c]) - skipped_chars))

        downcase_v = sorted(list(downcase_v - skipped_chars))
        downcase_c = sorted(list(downcase_c - skipped_chars))

        g_downcase = sorted(list(set(['α', 'β', 'γ', 'δ', 'ε', 'ζ', 'η', 'θ', 'ι', 'κ', 'λ', 'μ', 'ν', 'ξ', 'ο', 'π', 'ρ', 'σ', 'τ', 'υ', 'φ', 'χ', 'ψ', 'ω',]) - skipped_chars))
        g_upcase = sorted(list(set(['Α', 'Β', 'Γ', 'Δ', 'Ε', 'Ζ', 'Η', 'Θ', 'Ι', 'Κ', 'Λ', 'Μ', 'Ν', 'Ξ', 'Ο', 'Π', 'Ρ', 'Σ', 'Τ', 'Υ', 'Φ', 'Χ', 'Ψ', 'Ω']) - skipped_chars))

        for arr in [numbers, downcase_v, downcase_c, upcase_v, upcase_c, g_downcase, g_upcase]:
            arr2 = sorted(list(arr))
            rng.shuffle(arr2)
            temp = arr
            if force_diff:
                temp = sorted(list(arr2))
                t = temp.pop(0)
                temp.append(t)

            for idx, c in enumerate(temp):
                jt[c] = arr2[idx]

        new_words: dict[str, str] = {}
        for words in wordlist:
            temp_s = ''
            for word in words.split(' '):
                if re.sub(r'[^a-z0-9]', '', word.lower()) in conserved:
                    temp_s += word
                else:
                    for c in word:
                        if c in jt:
                            temp_s += jt[c]
                        else:
                            temp_s += c
                temp_s += ' '
            temp_s = temp_s.strip()
            new_words[words] = temp_s

        if reserved and any(v in reserved for v in new_words.values()):
            attempts += 1
            if attempts < int(max_reserved_retries):
                continue
            # Edge case: all attempts produce reserved outputs; keep original text.
            if return_mapping:
                return base_text, {}, True
            return base_text

        out = text
        for k, v in new_words.items():
            out = out.replace(k, v)
        if return_mapping:
            return out, new_words, False
        return out
