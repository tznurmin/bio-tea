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

import hashlib
import json
from pathlib import Path
from typing import Iterable


def sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding='utf-8')


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")


def write_conll(path: Path, examples: Iterable[list[tuple[str, str]]]) -> None:
    """Write CoNLL-style file.

    Each example is a list of (token, label) pairs. Examples are separated by a
    blank line.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        first = True
        for ex in examples:
            if not first:
                f.write("\n")
            first = False
            for tok, lab in ex:
                f.write(f"{tok} {lab}\n")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def read_conll(path: Path) -> list[list[tuple[str, str]]]:
    """Read CoNLL-style examples written by write_conll."""

    examples: list[list[tuple[str, str]]] = []
    cur: list[tuple[str, str]] = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line.strip():
                if cur:
                    examples.append(cur)
                    cur = []
                continue
            parts = line.split(' ', 1)
            if len(parts) != 2:
                continue
            tok, lab = parts[0], parts[1]
            cur.append((tok, lab))
    if cur:
        examples.append(cur)
    return examples
