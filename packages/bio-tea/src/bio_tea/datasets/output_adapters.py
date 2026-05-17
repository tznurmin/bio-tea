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

from typing import Iterable


_SCHEMA_KEYS = ("example_id", "tokens", "labels", "checksum", "variant", "category")


def _sorted(rows: Iterable[dict]) -> list[dict]:
    return sorted(rows, key=lambda r: str(r["example_id"]))


def _project_row(row: dict) -> dict:
    out = {k: row[k] for k in _SCHEMA_KEYS}
    out["tokens"] = list(out["tokens"])
    out["labels"] = list(out["labels"])
    return out


def to_jsonl_rows(rows: Iterable[dict]) -> list[dict]:
    """Return stable-schema rows sorted by example_id."""

    return [_project_row(r) for r in _sorted(rows)]


def to_conll_examples(rows: Iterable[dict]) -> list[list[tuple[str, str]]]:
    """Convert canonical rows into CoNLL examples."""

    examples: list[list[tuple[str, str]]] = []
    for r in to_jsonl_rows(rows):
        toks = r["tokens"]
        labels = r["labels"]
        if len(toks) != len(labels):
            raise ValueError(f"token/label length mismatch for example_id={r['example_id']}")
        examples.append(list(zip(toks, labels)))
    return examples


def to_hf_dataset(rows: Iterable[dict]):
    """Convert canonical rows into Hugging Face Dataset."""

    try:
        from datasets import Dataset
    except Exception as e:  # pragma: no cover
        raise RuntimeError("datasets is required for HF dataset adapter") from e

    return Dataset.from_list(to_jsonl_rows(rows))

