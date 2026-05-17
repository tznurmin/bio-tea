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

import os
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Tuple


DEFAULT_DOWNLOAD_URL = "https://github.com/tznurmin/TEA_curated_data/archive/refs/tags/v1.1.tar.gz"


def _looks_like_curated_root(p: Path) -> bool:
    return (p / "curation_data").is_dir() and (p / "source_articles").is_dir()


def _safe_extract_all(tf: tarfile.TarFile, destination: Path) -> None:
    destination = destination.resolve()

    for member in tf.getmembers():
        member_name = str(member.name)

        if member_name.startswith("/"):
            raise RuntimeError(f"Unsafe archive member with absolute path: {member_name}")

        target = (destination / member_name).resolve()
        if target != destination and destination not in target.parents:
            raise RuntimeError(f"Unsafe archive member outside extraction root: {member_name}")

        if member.issym() or member.islnk():
            raise RuntimeError(f"Refusing archive link member: {member_name}")

        if member.isdev():
            raise RuntimeError(f"Refusing archive device member: {member_name}")

    try:
        tf.extractall(path=destination, filter="data")
    except TypeError:
        tf.extractall(path=destination)


def _download_and_extract(url: str, dest_root: Path) -> Path:
    """Download a tar.gz archive and extract it to ./TEA_curated_data."""

    dest_root = dest_root.resolve()
    target = dest_root / "TEA_curated_data"
    if target.exists():
        # Refuse to overwrite the target directory.
        raise RuntimeError(f"Refusing to overwrite existing {target}")

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        archive_path = td_path / "tea_curated.tar.gz"

        with urllib.request.urlopen(url) as resp:
            archive_path.write_bytes(resp.read())

        extract_dir = td_path / "extract"
        extract_dir.mkdir(parents=True, exist_ok=True)

        with tarfile.open(archive_path, "r:gz") as tf:
            _safe_extract_all(tf, extract_dir)

        # Find the single top-level directory.
        top_levels = [p for p in extract_dir.iterdir() if p.is_dir()]
        if len(top_levels) != 1:
            raise RuntimeError("Unexpected archive layout: expected a single top-level directory")

        extracted = top_levels[0]
        shutil.move(str(extracted), str(target))

    if not _looks_like_curated_root(target):
        raise RuntimeError(f"Downloaded curated data does not look valid at {target}")
    return target


def resolve_curated_root(cfg: Mapping[str, Any], *, allow_download: bool = True) -> Tuple[Path, dict]:
    """Resolve TEA_curated_data root.

    Resolution order:
      1) cfg['curated_root'] (or cfg['curated_data_root'])
      2) env TEA_CURATED_ROOT, then TEA_CURATED_DATA
      3) local ./TEA_curated_data if present
      4) download v1.1 into ./TEA_curated_data (if allow_download)

    Returns (path, info_dict).
    """

    # 1) config
    for key in ("curated_root", "curated_data_root"):
        v = cfg.get(key)
        if v:
            p = Path(str(v)).expanduser().resolve()
            if not _looks_like_curated_root(p):
                raise RuntimeError(f"Configured curated root does not look valid: {p}")
            return p, {"source": "config", "path": str(p)}

    # 2) environment
    for env_key in ("TEA_CURATED_ROOT", "TEA_CURATED_DATA"):
        v = os.environ.get(env_key)
        if v:
            p = Path(v).expanduser().resolve()
            if not _looks_like_curated_root(p):
                raise RuntimeError(f"Env {env_key} points to invalid curated root: {p}")
            return p, {"source": "env", "env": env_key, "path": str(p)}

    # 3) local
    local = Path("TEA_curated_data").resolve()
    if local.exists():
        if not _looks_like_curated_root(local):
            raise RuntimeError(f"Local TEA_curated_data exists but does not look valid: {local}")
        return local, {"source": "local", "path": str(local)}

    # 4) download
    if not allow_download:
        raise RuntimeError("No curated root found (config/env/local) and downloads are disabled")

    url = str(cfg.get("curated_download_url") or DEFAULT_DOWNLOAD_URL)
    target = _download_and_extract(url, Path("."))
    return target, {"source": "download", "path": str(target), "download_url": url}
