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

import re
from pathlib import Path
from typing import Any, Mapping, Tuple


_ALLOWED_SCOPES = {"flat", "tokenizer", "tokenizer_maxlen", "model_tokenizer_maxlen"}


def _slug(v: object, *, fallback: str = "na", max_len: int = 80) -> str:
    s = str(v or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    if not s:
        s = fallback
    if len(s) > max_len:
        s = s[:max_len].rstrip("-")
    return s or fallback


def _profile_id(scope: str, tok_cfg: Mapping[str, Any], rp_cfg: Mapping[str, Any]) -> str | None:
    explicit = str(rp_cfg.get("id") or "").strip()
    if explicit:
        return _slug(explicit, fallback="profile")

    tokenizer_name = str(tok_cfg.get("name_or_path") or "").strip()
    tok_slug = _slug(tokenizer_name, fallback="tokenizer")
    max_len = int(tok_cfg.get("model_max_length", 0))

    model_name = str(
        rp_cfg.get("model_name")
        or tok_cfg.get("model_name_or_path")
        or tok_cfg.get("name_or_path")
        or ""
    ).strip()
    model_slug = _slug(model_name, fallback="model")

    if scope == "flat":
        return None
    if scope == "tokenizer":
        return f"tok-{tok_slug}"
    if scope == "tokenizer_maxlen":
        return f"tok-{tok_slug}--ml-{max_len}"
    if scope == "model_tokenizer_maxlen":
        return f"model-{model_slug}--tok-{tok_slug}--ml-{max_len}"
    raise ValueError(f"Unsupported results_profile.scope: {scope}")


def resolve_results_root(cfg: Mapping[str, Any], tok_cfg: Mapping[str, Any]) -> Tuple[Path, dict]:
    """Resolve output root for generated artifacts.

    Config keys:
    - results_root: base root (default: "results")
    - results_profile.scope: flat|tokenizer|tokenizer_maxlen|model_tokenizer_maxlen
    - results_profile.id: optional explicit profile id override
    - results_profile.model_name: optional model component for model_* scopes
    - results_profile.profiles_dir: optional parent dir under results_root (default: "profiles")
    """

    base_root = Path(str(cfg.get("results_root", "results")))
    rp_cfg = cfg.get("results_profile") or {}
    scope = str(rp_cfg.get("scope") or "flat").strip().lower()
    if scope not in _ALLOWED_SCOPES:
        raise ValueError(
            "results_profile.scope must be one of: flat, tokenizer, tokenizer_maxlen, model_tokenizer_maxlen"
        )
    pid = _profile_id(scope, tok_cfg, rp_cfg)

    if scope == "flat" or not pid:
        resolved = base_root
    else:
        profiles_dir = str(rp_cfg.get("profiles_dir") or "profiles").strip()
        resolved = (base_root / profiles_dir / pid) if profiles_dir else (base_root / pid)

    info = {
        "scope": scope,
        "profile_id": pid,
        "base_results_root": str(base_root),
        "resolved_results_root": str(resolved),
        "tokenizer_name_or_path": str(tok_cfg.get("name_or_path") or ""),
        "model_max_length": int(tok_cfg.get("model_max_length", 0)),
        "model_name": str(rp_cfg.get("model_name") or tok_cfg.get("model_name_or_path") or ""),
    }
    return resolved, info

