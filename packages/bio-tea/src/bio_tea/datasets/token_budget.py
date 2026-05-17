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
from typing import Any, Callable, Mapping

from .tokenizer_loader import load_tokenizer

_ALLOWED_BUDGET_BACKENDS = {"tokenize_count", "hf_encode"}


@dataclass(frozen=True)
class TokenBudgetBackendSpec:
    backend: str
    name_or_path: str
    do_lower_case: bool
    model_max_length: int
    special_tokens_overhead: int
    max_content_tokens: int

def _as_list(v: Any) -> list:
    if v is None:
        return []
    if isinstance(v, list):
        return v
    if isinstance(v, tuple):
        return list(v)
    return [v]


def _sanitized_tokenizer_cfg(v: Mapping[str, Any] | None) -> dict[str, Any]:
    out = dict(v or {})
    if out.get("name_or_path") is not None:
        out["name_or_path"] = str(out.get("name_or_path")).strip()
    return out


def collect_budget_tokenizer_cfgs(cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Collect tokenizer configs used for shared token-budget enforcement.

    Includes:
    - top-level `tokenizer`
    - per-model tokenizers from `training.models` (with top-level fallback)
    """

    base_tok = _sanitized_tokenizer_cfg(cfg.get("tokenizer") if isinstance(cfg, Mapping) else {})
    out: list[dict[str, Any]] = []

    if str(base_tok.get("name_or_path") or "").strip():
        out.append(dict(base_tok))

    train_cfg = dict((cfg.get("training") if isinstance(cfg, Mapping) else {}) or {})
    models = list(_as_list(train_cfg.get("models")))

    for m in models:
        if not isinstance(m, Mapping):
            continue
        model_tok_cfg = _sanitized_tokenizer_cfg(m.get("tokenizer") if isinstance(m, Mapping) else {})
        model_name = str(m.get("model_name_or_path") or m.get("name_or_path") or "").strip()

        tok_cfg = dict(base_tok)
        tok_cfg.update(model_tok_cfg)
        if not str(tok_cfg.get("name_or_path") or "").strip() and model_name:
            tok_cfg["name_or_path"] = model_name

        # Do not carry casing-policy defaults between tokenizer families.
        base_name = str(base_tok.get("name_or_path") or "").strip()
        tok_name = str(tok_cfg.get("name_or_path") or "").strip()
        if base_name and tok_name and tok_name != base_name:
            for key in ("do_lower_case", "require_case_sensitive", "require_cased"):
                if key not in model_tok_cfg:
                    tok_cfg.pop(key, None)

        if "max_length" in m and "model_max_length" not in tok_cfg:
            try:
                tok_cfg["model_max_length"] = int(m.get("max_length"))
            except Exception:
                pass

        if str(tok_cfg.get("name_or_path") or "").strip():
            out.append(tok_cfg)

    return out


def _tokenizer_count(tok: Any, text: str) -> int:
    tok_fn = getattr(tok, "tokenize", None)
    if callable(tok_fn):
        try:
            toks = tok_fn(text, verbose=False)
        except TypeError:
            toks = tok_fn(text)
        return len(list(toks))
    return len(str(text).split())


def _normalize_text_for_case(text: str, *, do_lower_case: bool) -> str:
    return str(text).lower() if bool(do_lower_case) else str(text)


def _encode_input_ids_len(tokenizer: Any, text: str, *, add_special_tokens: bool) -> int:
    """Return encoded input_ids length for a single text input."""

    kwargs = {
        "add_special_tokens": bool(add_special_tokens),
        "truncation": False,
    }
    try:
        enc = tokenizer(text, verbose=False, **kwargs)
    except TypeError:
        try:
            enc = tokenizer(text, **kwargs)
        except Exception:
            return _tokenizer_count(tokenizer, text)
    except Exception:
        return _tokenizer_count(tokenizer, text)

    ids = enc.get("input_ids") if isinstance(enc, Mapping) else None
    if isinstance(ids, list):
        if ids and isinstance(ids[0], list):
            return len(ids[0])
        return len(ids)
    return _tokenizer_count(tokenizer, text)


def resolve_token_budget_backend_id(cfg: Mapping[str, Any] | None) -> str:
    tb_cfg = dict((cfg or {}).get("token_budget") or {})
    backend = str(tb_cfg.get("backend") or "").strip().lower() or "hf_encode"
    if backend not in _ALLOWED_BUDGET_BACKENDS:
        raise ValueError("token_budget.backend must be one of: tokenize_count, hf_encode")
    return backend


def _entry_backend_spec(entry: Mapping[str, Any], *, backend: str) -> TokenBudgetBackendSpec:
    info = dict(entry.get("info") or {})
    tok = entry.get("tokenizer")
    do_lower = bool(info.get("do_lower_case", False))
    model_max_length = int(info.get("model_max_length", 0) or 0)

    special_overhead = 0
    if backend == "hf_encode":
        base = _encode_input_ids_len(tok, _normalize_text_for_case("", do_lower_case=do_lower), add_special_tokens=False)
        with_specials = _encode_input_ids_len(
            tok,
            _normalize_text_for_case("", do_lower_case=do_lower),
            add_special_tokens=True,
        )
        special_overhead = max(0, int(with_specials) - int(base))

    max_content = model_max_length
    if model_max_length > 0:
        max_content = max(0, int(model_max_length) - int(special_overhead))

    return TokenBudgetBackendSpec(
        backend=str(backend),
        name_or_path=str(info.get("name_or_path") or ""),
        do_lower_case=do_lower,
        model_max_length=model_max_length,
        special_tokens_overhead=int(special_overhead),
        max_content_tokens=int(max_content),
    )


def _count_with_backend(entry: Mapping[str, Any], *, text: str, backend: str) -> int:
    tok = entry.get("tokenizer")
    info = dict(entry.get("info") or {})
    do_lower = bool(info.get("do_lower_case", False))
    s = _normalize_text_for_case(str(text), do_lower_case=do_lower)
    if backend == "hf_encode":
        return int(_encode_input_ids_len(tok, s, add_special_tokens=False))
    return int(_tokenizer_count(tok, s))


def load_budget_tokenizers(cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Load and de-duplicate tokenizer entries used by shared token budgeting."""

    entries: list[dict[str, Any]] = []
    seen: set[tuple[str, bool, bool, str]] = set()

    for tok_cfg in collect_budget_tokenizer_cfgs(cfg):
        tok, info = load_tokenizer(tok_cfg)
        key = (
            str(info.get("name_or_path") or ""),
            bool(info.get("do_lower_case", False)),
            bool(info.get("local_files_only", False)),
            str(info.get("revision") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        entries.append(
            {
                "tokenizer": tok,
                "cfg": dict(tok_cfg),
                "info": dict(info),
            }
        )
    return entries


def build_composite_token_counter(
    entries: list[dict[str, Any]],
    *,
    backend: str = "tokenize_count",
) -> tuple[Callable[[str], int], list[dict[str, Any]]]:
    """Build max-over-tokenizers token counter and serializable metadata."""

    if backend not in _ALLOWED_BUDGET_BACKENDS:
        raise ValueError("budget backend must be one of: tokenize_count, hf_encode")

    counters: list[Callable[[str], int]] = []
    meta: list[dict[str, Any]] = []

    for e in entries:
        info = dict(e.get("info") or {})
        spec = _entry_backend_spec(e, backend=backend)

        def _count_one(text: str, *, _entry=e, _backend=backend) -> int:
            return int(_count_with_backend(_entry, text=str(text), backend=str(_backend)))

        counters.append(_count_one)
        meta.append(
            {
                "backend": str(backend),
                "name_or_path": str(info.get("name_or_path") or ""),
                "do_lower_case": bool(info.get("do_lower_case", False)),
                "model_max_length": int(info.get("model_max_length", 0) or 0),
                "special_tokens_overhead": int(spec.special_tokens_overhead),
                "max_content_tokens": int(spec.max_content_tokens),
                "tokenizer_class": str(info.get("tokenizer_class") or ""),
                "revision": info.get("revision"),
                "local_files_only": bool(info.get("local_files_only", False)),
            }
        )

    def _count(text: str) -> int:
        if not counters:
            return len(str(text).split())
        vals = [int(fn(str(text))) for fn in counters]
        return max(vals) if vals else len(str(text).split())

    return _count, meta


def install_token_counter(tea: Any, token_counter: Callable[[str], int] | None) -> None:
    """Override TEA token counter in-place when provided."""

    if callable(token_counter):
        tea._token_counter = token_counter
