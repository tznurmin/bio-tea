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
from typing import Any, Mapping


def _as_list(v: Any) -> list:
    if v is None:
        return []
    if isinstance(v, list):
        return v
    if isinstance(v, tuple):
        return list(v)
    return [v]


def _slug(v: object, *, fallback: str = "na", max_len: int = 80) -> str:
    s = str(v or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    if not s:
        s = fallback
    if len(s) > max_len:
        s = s[:max_len].rstrip("-")
    return s or fallback


def resolve_dataset_model_profiles(cfg: Mapping[str, Any]) -> list[dict]:
    """Resolve dataset-generation model/tokenizer profiles from config.

    Dataset generation requires explicit `training.models` entries.
    """

    train_cfg = dict(cfg.get("training") or {})
    raw_models = list(_as_list(train_cfg.get("models")))
    if not raw_models:
        raise RuntimeError("training.models must contain at least one model entry for dataset generation")

    out: list[dict] = []
    for i, m in enumerate(raw_models):
        if not isinstance(m, Mapping):
            raise RuntimeError(f"training.models[{i}] must be an object")
        model_name = str(m.get("model_name_or_path") or m.get("name_or_path") or "").strip()
        if not model_name:
            raise RuntimeError(f"training.models[{i}].model_name_or_path is required")
        model_id = str(m.get("id") or "").strip() or _slug(model_name, fallback=f"model-{i+1}")

        tok_cfg = dict(m.get("tokenizer") or {})
        if not str(tok_cfg.get("name_or_path") or "").strip():
            tok_cfg["name_or_path"] = model_name

        max_len = int(m.get("max_length") or tok_cfg.get("model_max_length") or 510)
        tok_cfg["model_max_length"] = max_len

        out.append(
            {
                "id": model_id,
                "model_name_or_path": model_name,
                "max_length": max_len,
                "tokenizer": tok_cfg,
                "model_config": dict(m),
            }
        )
    return out
