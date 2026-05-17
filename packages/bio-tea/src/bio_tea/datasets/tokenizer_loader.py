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

from typing import Any, Mapping, Tuple


def _as_token_list(v: Any) -> list[str]:
    if isinstance(v, (list, tuple)):
        return [str(x) for x in v]
    return []


def _unk_token_set(tokenizer: Any) -> set[str]:
    out = {"[UNK]", "<unk>", "<UNK>", "unk", "UNK"}
    v = getattr(tokenizer, "unk_token", None)
    if isinstance(v, str) and v.strip():
        out.add(v)
    return out


def _count_unk(tokens: list[str], unk_tokens: set[str]) -> int:
    return sum(1 for t in tokens if str(t) in unk_tokens)


def _probe_case_sensitive(tokenizer: Any) -> bool:
    """Return True when tokenizer behavior appears case-sensitive.

    Uses multiple probes to reduce false positives from occasional [UNK] paths.
    """

    probes = [
        ("AbC DEF", "abc def"),
        ("Escherichia coli", "escherichia coli"),
        ("TNF alpha", "tnf alpha"),
    ]

    tok_fn = getattr(tokenizer, "tokenize", None)
    if not callable(tok_fn):
        return False

    for upper, lower in probes:
        try:
            up = tok_fn(upper)
            lo = tok_fn(lower)
        except Exception:
            return False
        if up != lo:
            return True
    return False


def _probe_lowercase_preference(tokenizer: Any) -> dict[str, Any]:
    """Probe whether tokenizer appears to require lowercase input."""

    probes = [
        ("Escherichia coli", "escherichia coli"),
        ("Staphylococcus aureus", "staphylococcus aureus"),
        ("Bacillus subtilis", "bacillus subtilis"),
    ]

    tok_fn = getattr(tokenizer, "tokenize", None)
    if not callable(tok_fn):
        return {"available": False, "prefers_lowercase": False, "n_probes": 0, "suspicious_probes": 0}

    unk_tokens = _unk_token_set(tokenizer)
    suspicious = 0
    for upper, lower in probes:
        try:
            up = _as_token_list(tok_fn(upper))
            lo = _as_token_list(tok_fn(lower))
        except Exception:
            return {"available": False, "prefers_lowercase": False, "n_probes": 0, "suspicious_probes": 0}
        if not up or not lo:
            continue
        up_unk = _count_unk(up, unk_tokens)
        lo_unk = _count_unk(lo, unk_tokens)
        if up_unk > lo_unk and up_unk > 0:
            suspicious += 1

    n_probes = len(probes)
    return {
        "available": True,
        "prefers_lowercase": bool(suspicious > 0),
        "n_probes": int(n_probes),
        "suspicious_probes": int(suspicious),
    }


def _resolve_do_lower_case(tokenizer: Any, cfg_do_lower_case: bool) -> bool:
        # Use the runtime tokenizer attribute when present.
    v = getattr(tokenizer, "do_lower_case", None)
    if isinstance(v, bool):
        return v

    # Hugging Face tokenizers commonly expose init kwargs.
    init_kwargs = getattr(tokenizer, "init_kwargs", None)
    if isinstance(init_kwargs, dict):
        v2 = init_kwargs.get("do_lower_case")
        if isinstance(v2, bool):
            return v2

    return bool(cfg_do_lower_case)


def _resolve_require_case_sensitive(
    *,
    name_or_path: str,
    cfg_do_lower_case: bool,
    tok_cfg: Mapping[str, Any],
) -> bool:
    # Use the explicit casing-policy key when present.
    if "require_case_sensitive" in tok_cfg:
        return bool(tok_cfg.get("require_case_sensitive"))

    # Support the shorter casing-policy key.
    if "require_cased" in tok_cfg:
        return bool(tok_cfg.get("require_cased"))

    # Casing enforcement is disabled unless explicitly requested.
    # Most Hugging Face model/tokenizer pairs expose the needed setting.
    _ = name_or_path
    _ = cfg_do_lower_case
    return False


def load_tokenizer(tok_cfg: Mapping[str, Any]) -> Tuple[Any, dict]:
    """Load tokenizer with cased/uncased safety checks.

    Parameters
    ----------
    tok_cfg:
        Config mapping containing at minimum `name_or_path`.
        Supported keys include:
        - do_lower_case (bool)
        - model_max_length (int)
        - require_cased (bool)
    """

    name_or_path = tok_cfg.get("name_or_path")
    if not name_or_path:
        raise RuntimeError("tokenizer.name_or_path is required")

    do_lower_case_raw = tok_cfg.get("do_lower_case", None)
    do_lower_case = None if do_lower_case_raw is None else bool(do_lower_case_raw)
    model_max_length = int(tok_cfg.get("model_max_length", 100000))
    local_files_only = bool(tok_cfg.get("local_files_only", False))
    revision_raw = tok_cfg.get("revision")
    revision = str(revision_raw).strip() if revision_raw is not None else ""
    revision = revision or None
    require_case_sensitive = _resolve_require_case_sensitive(
        name_or_path=str(name_or_path),
        cfg_do_lower_case=bool(do_lower_case_raw) if do_lower_case_raw is not None else False,
        tok_cfg=tok_cfg,
    )
    enforce_lowercase_probe = bool(tok_cfg.get("enforce_lowercase_probe", True))

    try:
        from transformers import AutoTokenizer
    except Exception as e:  # pragma: no cover
        raise RuntimeError("transformers is required for tokenizer loading") from e

    load_kwargs = {
        "model_max_length": model_max_length,
    }
    if do_lower_case is not None:
        load_kwargs["do_lower_case"] = bool(do_lower_case)
    if local_files_only:
        load_kwargs["local_files_only"] = True
    if revision is not None:
        load_kwargs["revision"] = revision

    tokenizer = AutoTokenizer.from_pretrained(
        str(name_or_path),
        **load_kwargs,
    )

    resolved_do_lower_case = _resolve_do_lower_case(
        tokenizer,
        bool(do_lower_case_raw) if do_lower_case_raw is not None else False,
    )
    casing_verified = False
    lowercase_probe = _probe_lowercase_preference(tokenizer)

    if require_case_sensitive:
        if bool(do_lower_case) or resolved_do_lower_case:
            raise RuntimeError(
                "Case-sensitive run requires do_lower_case=False and case-sensitive tokenizer behavior"
            )
        if not _probe_case_sensitive(tokenizer):
            raise RuntimeError("Tokenizer is not case-sensitive under runtime probe")
        casing_verified = True
    elif (not bool(resolved_do_lower_case)) and enforce_lowercase_probe:
        if bool(lowercase_probe.get("prefers_lowercase", False)):
            raise RuntimeError(
                "Tokenizer probe indicates lowercase input is required for stable tokenization; "
                "set do_lower_case=true or disable enforce_lowercase_probe."
            )

    info = {
        "name_or_path": str(name_or_path),
        "do_lower_case": bool(resolved_do_lower_case),
        "model_max_length": int(model_max_length),
        "tokenizer_class": tokenizer.__class__.__name__,
        "is_fast": bool(getattr(tokenizer, "is_fast", False)),
        "require_case_sensitive": require_case_sensitive,
        "require_cased": require_case_sensitive,
        "casing_verified": casing_verified,
        "enforce_lowercase_probe": enforce_lowercase_probe,
        "lowercase_probe": lowercase_probe,
        "local_files_only": local_files_only,
        "revision": revision,
    }
    return tokenizer, info
