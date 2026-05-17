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

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from bio_tea.datasets.io import read_conll, read_jsonl


def load_canonical_rows(set_path: Path, meta_path: Path | None = None) -> list[dict]:
    """Load canonical row records from a .set file and optional meta jsonl."""

    examples = read_conll(Path(set_path))
    meta_rows: list[dict] | None = None
    if meta_path is not None:
        meta_rows = read_jsonl(Path(meta_path))
        if len(meta_rows) != len(examples):
            raise ValueError(
                f"example/meta length mismatch: n_examples={len(examples)} n_meta={len(meta_rows)}"
            )

    rows: list[dict] = []
    for i, ex in enumerate(examples):
        toks = [t for t, _lab in ex]
        labs = [lab for _t, lab in ex]

        meta = dict(meta_rows[i]) if meta_rows is not None else {}
        eid = str(meta.get("example_id") or f"ex_{i:08d}")
        row = {
            "example_id": eid,
            "tokens": toks,
            "labels": labs,
        }
        # Keep useful provenance attributes for training/reporting surfaces.
        for k in [
            "category",
            "types_raw",
            "types_pooled",
            "checksum",
            "variant",
            "reason",
            "entities",
            "triggers",
        ]:
            if k in meta:
                row[k] = meta[k]
        rows.append(row)
    return rows


def build_label_maps(rows: Iterable[Mapping[str, Any]]) -> tuple[dict[str, int], dict[int, str]]:
    labels: set[str] = set()
    for r in rows:
        for lab in r.get("labels") or []:
            labels.add(str(lab))

    ordered = ["O"] + sorted([x for x in labels if x != "O"])
    label2id = {lab: i for i, lab in enumerate(ordered)}
    id2label = {i: lab for lab, i in label2id.items()}
    return label2id, id2label


def align_word_labels(
    *,
    word_ids: Sequence[int | None],
    word_labels: Sequence[str],
    label2id: Mapping[str, int],
    ignore_index: int = -100,
) -> list[int]:
    out: list[int] = []
    prev = None
    for wid in word_ids:
        if wid is None:
            out.append(int(ignore_index))
            prev = wid
            continue
        if wid < 0 or wid >= len(word_labels):
            raise ValueError(f"word index out of range during label alignment: wid={wid} n_words={len(word_labels)}")
        if wid != prev:
            lab = str(word_labels[wid])
            if lab not in label2id:
                raise KeyError(f"label not found in label2id: {lab}")
            out.append(int(label2id[lab]))
        else:
            out.append(int(ignore_index))
        prev = wid
    return out


def _encoding_word_ids(enc: Any) -> list[int | None]:
    wi = getattr(enc, "word_ids", None)
    if callable(wi):
        try:
            vals = wi()
        except TypeError:
            vals = wi(0)
        return list(vals)
    raise ValueError("tokenizer output does not expose word_ids() required for label alignment")


def _encode_input_ids(
    *,
    tokenizer: Any,
    tokens: Sequence[str],
    truncation: bool,
    max_length: int,
) -> list[int]:
    kwargs = {
        "is_split_into_words": True,
        "truncation": bool(truncation),
        "max_length": int(max_length),
    }
    with _suppress_hf_long_sequence_warning():
        try:
            enc = tokenizer(tokens, verbose=False, **kwargs)
        except TypeError:
            enc = tokenizer(tokens, **kwargs)
    return list(enc.get("input_ids") or [])


@contextmanager
def _suppress_hf_long_sequence_warning():
    """Suppress only HF advisory warning emitted before downstream hard clipping."""

    logger = logging.getLogger("transformers.tokenization_utils_base")

    class _LongSeqFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:  # pragma: no cover - logger plumbing
            msg = record.getMessage()
            return "Token indices sequence length is longer than the specified maximum sequence length" not in msg

    filt = _LongSeqFilter()
    logger.addFilter(filt)
    try:
        yield
    finally:
        logger.removeFilter(filt)


def prepare_tokenized_rows(
    *,
    rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    label2id: Mapping[str, int],
    max_length: int,
    ignore_index: int = -100,
    fail_on_truncation: bool = False,
    lowercase_tokens: bool = False,
) -> list[dict]:
    """Tokenize row-wise and align word-level IOB labels to subword tokens."""

    out: list[dict] = []
    for r in rows:
        tokens = list(r.get("tokens") or [])
        if lowercase_tokens:
            tokens = [str(t).lower() for t in tokens]
        word_labels = [str(x) for x in (r.get("labels") or [])]
        tok_kwargs = {
            "is_split_into_words": True,
            "truncation": True,
            "max_length": int(max_length),
        }
        with _suppress_hf_long_sequence_warning():
            try:
                # Ask HF to suppress advisory logs where supported.
                enc = tokenizer(tokens, verbose=False, **tok_kwargs)
            except TypeError:
                # Handle tokenizers that do not accept `verbose`.
                enc = tokenizer(tokens, **tok_kwargs)

        input_ids = list(enc.get("input_ids") or [])
        attention_mask = list(enc.get("attention_mask") or [1] * len(input_ids))
        word_ids = _encoding_word_ids(enc)

        # Some tokenizers can still emit >max_length sequences with pre-tokenized input.
        # Enforce hard clipping to guarantee model-safe boundaries.
        limit = int(max_length)
        overflow_len = len(input_ids) > limit or len(attention_mask) > limit or len(word_ids) > limit
        seen_word_ids = [int(wid) for wid in word_ids if wid is not None]
        overflow_words = False
        if tokens:
            if not seen_word_ids:
                overflow_words = True
            else:
                overflow_words = max(seen_word_ids) < (len(tokens) - 1)

        if bool(fail_on_truncation) and (overflow_len or overflow_words):
            # Overflow in word_ids can be a false positive with tokenizers that
            # drop/normalize some pre-tokenized words. Confirm against
            # untruncated encoding before failing.
            untruncated_ids = _encode_input_ids(
                tokenizer=tokenizer,
                tokens=tokens,
                truncation=False,
                max_length=limit,
            )
            if len(untruncated_ids) > limit:
                raise ValueError(
                    "tokenized row exceeds max_length budget and would require truncation: "
                    f"example_id={r.get('example_id')} tokens={len(tokens)} "
                    f"encoded={len(input_ids)} untruncated={len(untruncated_ids)} max_length={limit}"
                )

        if overflow_len:
            input_ids = input_ids[:limit]
            attention_mask = attention_mask[:limit]
            word_ids = word_ids[:limit]

        # Keep arrays strictly aligned before label construction.
        seq_len = min(len(input_ids), len(attention_mask), len(word_ids))
        input_ids = input_ids[:seq_len]
        attention_mask = attention_mask[:seq_len]
        word_ids = word_ids[:seq_len]

        labels = align_word_labels(
            word_ids=word_ids,
            word_labels=word_labels,
            label2id=label2id,
            ignore_index=ignore_index,
        )

        if not (len(input_ids) == len(attention_mask) == len(labels)):
            raise ValueError(
                "tokenized row length mismatch: "
                f"input_ids={len(input_ids)} attention_mask={len(attention_mask)} labels={len(labels)}"
            )

        rec = {
            "example_id": str(r.get("example_id")),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        for k in ["category", "types_raw", "types_pooled", "checksum", "variant"]:
            if k in r:
                rec[k] = r[k]
        out.append(rec)
    return out
