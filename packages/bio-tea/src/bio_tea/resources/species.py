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
"""Species list loading and normalisation.

The resource loader:
- reads bio_tea.data/species.txt
- builds full-binomial and abbreviated-genus lookup forms
- returns a shuffled full-binomial list for sampling replacements
"""

from __future__ import annotations

import importlib.resources
import random
from typing import Iterable


def load_species_text() -> str:
    """Read the packaged species list as a single string."""
    # Load packaged species data through importlib.resources.files().
    return importlib.resources.files("bio_tea.data").joinpath("species.txt").read_text(encoding="utf-8")


def build_all_species(species_text: str) -> set[str]:
    """Build the species lookup set (full + abbreviated-genus forms)."""
    all_species: set[str] = set()
    for line in species_text.split("\n"):
        temp = line.strip()
        if not temp:
            continue
        all_species.add(temp)
        parts = temp.split(" ")
        if len(parts) >= 2:
            all_species.add(f"{temp[0]}. {parts[1]}")
    return all_species


def build_species_list(all_species: Iterable[str], rng=random) -> list[str]:
    """Build and shuffle the list of full binomials used for sampling."""
    species_list = sorted([
        spec
        for spec in all_species
        if len(spec) > 1 and spec[1] != '.'
    ])
    rng.shuffle(species_list)
    return species_list
