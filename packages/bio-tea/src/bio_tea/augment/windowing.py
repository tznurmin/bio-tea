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
"""Sentence windowing for TEA.

This module operates on TEA's token stream and provides sentence-boundary and
window-expansion helpers used by the augmentation pipeline.
"""

from __future__ import annotations

from typing import Callable, Iterable, List, Sequence, Tuple


def is_stop(word: str, non_stops: Iterable[str]) -> bool:
    """Return True when the token is treated as a sentence boundary."""

    if not word or word[-1] != ".":
        return False

    # Special-case: treat '].' as a stop even though it's short.
    if word != "]." and (len(word) < 3 or word.lower() in set(map(str.lower, non_stops))):
        return False

    return True


def extract_sentence(sp: int, tokens: Sequence[str], non_stops: Iterable[str]) -> Tuple[int, int]:
    """Extract the sentence span (s, e) containing token index sp.

    A "sentence" is a maximal contiguous run of tokens not interrupted by is_stop.
    """

    s = sp
    e = sp

    while e < len(tokens) - 1 and not is_stop(tokens[e], non_stops):
        e += 1

    while s > 0 and not is_stop(tokens[s - 1], non_stops):
        s -= 1

    return (s, e)


def maximise(
    loc: int,
    tokens: Sequence[str],
    maxlen: int,
    token_counter: Callable[[str], int],
    non_stops: Iterable[str],
    rng,
) -> Tuple[int, int]:
    """Expand a token window by adding complete neighbouring sentences.

    Expansion starts from the sentence containing the target span, attempts to
    add left context first, then samples remaining left/right context while
    preserving the token-budget constraint.
    """

    r_count = 0
    f_count = 0
    l_count = 0

    s, e = extract_sentence(loc, tokens, non_stops)

    s_done = s == 0
    e_done = e == len(tokens) - 1

    while not (s_done and e_done):
        l_count += 1
        if not s_done and (r_count < 2 or rng.randint(0, 1) == 0):
            new_s, _ = extract_sentence(s - 1, tokens, non_stops)
            if token_counter(" ".join(tokens[new_s : e + 1])) < maxlen:
                r_count += 1
                s = new_s
                if s == 0:
                    s_done = True
            else:
                s_done = True
        else:
            if e >= len(tokens) - 1:
                e_done = True
                continue
            _, new_e = extract_sentence(e + 1, tokens, non_stops)
            if token_counter(" ".join(tokens[s : new_e + 1])) < maxlen:
                f_count += 1
                e = new_e
                if e == len(tokens) - 1:
                    e_done = True
            else:
                e_done = True

    return s, e
