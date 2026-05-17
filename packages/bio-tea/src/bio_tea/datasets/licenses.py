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
from typing import Mapping


_HEX_RE = re.compile(r"^[a-f0-9]{8,128}$", re.IGNORECASE)
_DEFAULT_POLICY = {
    "mode": "exclude",
    "allow_noncommercial": True,
    "allow_share_alike": True,
    "disallow_no_derivatives": True,
    "disallow_all_rights_reserved": True,
    "unknown": "exclude",
    "allowed_license_ids": None,
}


def _sorted_counts(counts: Mapping[str, int]) -> dict[str, int]:
    return dict(sorted(((str(k), int(v)) for k, v in counts.items()), key=lambda kv: kv[0]))


def _is_cc_license_id(license_id: str) -> bool:
    lid = (license_id or "").strip().upper()
    return lid.startswith("CC-") or lid == "CC0-1.0"


def _normalize_allowed_license_ids(v: object) -> list[str]:
    if v is None:
        return []
    if not isinstance(v, (list, tuple, set)):
        raise ValueError("licenses.allowed_license_ids must be a list-like value")
    out = []
    for x in v:
        lid = normalize_license_id(str(x))
        if lid:
            out.append(lid)
    return sorted(list(dict.fromkeys(out)))


def normalize_license_id(raw: str) -> str:
    s = (raw or "").strip().lower()
    s = s.replace("_", "-")
    if not s:
        return "UNKNOWN"
    if "all rights reserved" in s:
        return "ALL-RIGHTS-RESERVED"
    if "creativecommons.org/publicdomain/zero" in s or "cc0" in s:
        return "CC0-1.0"
    if "creative commons" in s:
        has_nc = ("noncommercial" in s) or ("non-commercial" in s)
        has_nd = ("noderivatives" in s) or ("no derivatives" in s) or ("no-derivatives" in s)
        has_sa = ("sharealike" in s) or ("share alike" in s) or ("share-alike" in s)
        if has_nc and has_nd:
            return "CC-BY-NC-ND-4.0"
        if has_nc and has_sa:
            return "CC-BY-NC-SA-4.0"
        if has_nd:
            return "CC-BY-ND-4.0"
        if has_sa:
            return "CC-BY-SA-4.0"
        if has_nc:
            return "CC-BY-NC-4.0"
        if "attribution" in s:
            return "CC-BY-4.0"
    if "creativecommons.org/licenses/by-nc-nd/4.0" in s or "cc by-nc-nd" in s or "cc-by-nc-nd" in s:
        return "CC-BY-NC-ND-4.0"
    if "creativecommons.org/licenses/by-nc-sa/4.0" in s or "cc by-nc-sa" in s or "cc-by-nc-sa" in s:
        return "CC-BY-NC-SA-4.0"
    if "creativecommons.org/licenses/by-nd/4.0" in s or "cc by-nd" in s or "cc-by-nd" in s:
        return "CC-BY-ND-4.0"
    if "creativecommons.org/licenses/by-sa/4.0" in s or "cc by-sa" in s or "cc-by-sa" in s:
        return "CC-BY-SA-4.0"
    if "creativecommons.org/licenses/by-nc/4.0" in s or "cc by-nc" in s or "cc-by-nc" in s:
        return "CC-BY-NC-4.0"
    if "creativecommons.org/licenses/by/4.0" in s or "cc by 4.0" in s or "cc-by-4.0" in s:
        return "CC-BY-4.0"
    if "mit" == s or s.startswith("mit "):
        return "MIT"
    return raw.strip()


def license_policy_flags(license_id: str) -> dict:
    lid = (license_id or "UNKNOWN").strip().upper()
    if lid == "UNKNOWN":
        return {
            "attribution_required": False,
            "redistribution_allowed": False,
            "derivative_allowed": False,
            "policy_status": "unknown",
        }
    if lid == "ALL-RIGHTS-RESERVED":
        return {
            "attribution_required": False,
            "redistribution_allowed": False,
            "derivative_allowed": False,
            "policy_status": "disallowed",
        }
    if "-ND" in lid:
        return {
            "attribution_required": True,
            "redistribution_allowed": False,
            "derivative_allowed": False,
            "policy_status": "disallowed",
        }
    if lid in {"CC-BY-4.0", "CC-BY-NC-4.0", "CC-BY-SA-4.0", "CC-BY-NC-SA-4.0", "MIT"}:
        return {
            "attribution_required": lid != "MIT",
            "redistribution_allowed": True,
            "derivative_allowed": True,
            "policy_status": "allowed",
        }
    if lid == "CC0-1.0":
        return {
            "attribution_required": False,
            "redistribution_allowed": True,
            "derivative_allowed": True,
            "policy_status": "allowed",
        }
    return {
        "attribution_required": False,
        "redistribution_allowed": False,
        "derivative_allowed": False,
        "policy_status": "unknown",
    }


def _unknown_action(v: object) -> str:
    action = str(v or "allow").strip().lower()
    if action not in {"allow", "exclude", "fail"}:
        raise ValueError("licenses.unknown must be one of: allow, exclude, fail")
    return action


def license_policy_from_config(cfg: Mapping[str, object] | None) -> dict:
    c = dict(_DEFAULT_POLICY)
    if isinstance(cfg, Mapping):
        if "mode" in cfg:
            c["mode"] = str(cfg.get("mode") or "exclude").strip().lower()
        if "allow_noncommercial" in cfg:
            c["allow_noncommercial"] = bool(cfg.get("allow_noncommercial"))
        if "allow_share_alike" in cfg:
            c["allow_share_alike"] = bool(cfg.get("allow_share_alike"))
        if "disallow_no_derivatives" in cfg:
            c["disallow_no_derivatives"] = bool(cfg.get("disallow_no_derivatives"))
        if "disallow_all_rights_reserved" in cfg:
            c["disallow_all_rights_reserved"] = bool(cfg.get("disallow_all_rights_reserved"))
        if "unknown" in cfg:
            c["unknown"] = _unknown_action(cfg.get("unknown"))
        if "allowed_license_ids" in cfg:
            c["allowed_license_ids"] = _normalize_allowed_license_ids(cfg.get("allowed_license_ids"))

    mode = str(c["mode"]).strip().lower()
    if mode not in {"exclude", "fail", "warn"}:
        raise ValueError("licenses.mode must be one of: exclude, fail, warn")
    c["mode"] = mode
    c["unknown"] = _unknown_action(c.get("unknown"))
    return c


def policy_reasons_for_license(license_id: str, policy: Mapping[str, object], allowed_ids: set[str] | None = None) -> list[str]:
    lid = (license_id or "UNKNOWN").strip().upper()
    reasons: list[str] = []
    if lid == "UNKNOWN":
        unknown_action = str(policy.get("unknown") or "allow").strip().lower()
        if unknown_action in {"exclude", "fail"}:
            reasons.append("unknown_license")
        return reasons

    if allowed_ids is not None and lid not in allowed_ids:
        reasons.append("license_not_in_allowed_set")

    if bool(policy.get("disallow_all_rights_reserved", True)) and lid == "ALL-RIGHTS-RESERVED":
        reasons.append("all_rights_reserved_disallowed")
    if bool(policy.get("disallow_no_derivatives", True)) and "-ND" in lid:
        reasons.append("no_derivatives_disallowed")
    if (not bool(policy.get("allow_noncommercial", True))) and "-NC" in lid:
        reasons.append("noncommercial_disallowed")
    if (not bool(policy.get("allow_share_alike", True))) and "-SA" in lid:
        reasons.append("share_alike_disallowed")
    return reasons


def _license_info_for_checksum(checksum: str, attrib_map: Mapping[str, Mapping[str, object]]) -> dict:
    info = attrib_map.get(checksum)
    if info is None:
        return {"license_id": "UNKNOWN", **license_policy_flags("UNKNOWN")}
    lid = str(info.get("license_id", "UNKNOWN"))
    return {"license_id": lid, **license_policy_flags(lid)}


def parse_attribution_text(text: str) -> dict[str, dict]:
    """Parse checksum->license entries from attribution text.

    Supported patterns (case-insensitive):
    - `checksum: <hex>` / `sha256: <hex>` then `license: <value>` in same block.
    - `<hex>,<license>` or whitespace-separated `<hex> <license>` tabular lines.
    """

    out: dict[str, dict] = {}
    cur_checksum: str | None = None
    cur_license: str | None = None

    def flush():
        nonlocal cur_checksum, cur_license
        if cur_checksum:
            lid = normalize_license_id(cur_license or "UNKNOWN")
            out[cur_checksum] = {"license_id": lid, **license_policy_flags(lid)}
        cur_checksum = None
        cur_license = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            flush()
            continue

        m_c = re.match(r"^(?:checksum|sha256|hash)\s*[:=]\s*([a-f0-9]{8,128})\s*$", line, flags=re.IGNORECASE)
        if m_c:
            if cur_checksum and cur_checksum != m_c.group(1):
                flush()
            cur_checksum = m_c.group(1)
            continue

        m_l = re.match(r"^(?:license|licence)\s*[:=]\s*(.+)\s*$", line, flags=re.IGNORECASE)
        if not m_l:
            # Curated TEA attribution format also uses:
            # "is obtained by the following licence: <value>"
            m_l = re.search(r"\b(?:license|licence)\b\s*[:=]\s*(.+)\s*$", line, flags=re.IGNORECASE)
        if m_l:
            cur_license = m_l.group(1).strip()
            continue

        # Tabular-style line: checksum + separator + license
        parts = re.split(r"[\t,;|]+", line, maxsplit=1)
        if len(parts) == 2 and _HEX_RE.match(parts[0].strip()):
            checksum = parts[0].strip()
            lid = normalize_license_id(parts[1].strip())
            out[checksum] = {"license_id": lid, **license_policy_flags(lid)}
            continue

        # Space-separated fallback: "<hex> <license...>"
        tok = line.split()
        if len(tok) >= 2 and _HEX_RE.match(tok[0]):
            checksum = tok[0]
            lid = normalize_license_id(" ".join(tok[1:]))
            out[checksum] = {"license_id": lid, **license_policy_flags(lid)}
            continue

    flush()
    return out


def load_attribution_map(curated_root: Path) -> tuple[dict[str, dict], dict]:
    """Load curated attribution map. Missing file returns empty map + info."""

    cands = [
        curated_root / "attribution.txt",
        curated_root / "ATTRIBUTION.txt",
        curated_root / "licenses.txt",
    ]
    for p in cands:
        if p.exists():
            text = p.read_text(encoding="utf-8")
            parsed = parse_attribution_text(text)
            return parsed, {"source_path": str(p), "found": True}
    return {}, {"source_path": None, "found": False}


def summarize_licenses(checksums: list[str], attrib_map: dict[str, dict]) -> dict:
    details: dict[str, dict] = {}
    counts: dict[str, int] = {}
    n_unknown = 0
    n_disallowed = 0
    for csum in checksums:
        info = _license_info_for_checksum(csum, attrib_map)
        lid = str(info.get("license_id", "UNKNOWN"))
        counts[lid] = int(counts.get(lid, 0)) + 1
        if lid == "UNKNOWN" or str(info.get("policy_status")) == "unknown":
            n_unknown += 1
        if str(info.get("policy_status")) == "disallowed":
            n_disallowed += 1
        details[csum] = {
            "license_id": lid,
            "attribution_required": bool(info.get("attribution_required", False)),
            "redistribution_allowed": bool(info.get("redistribution_allowed", False)),
            "derivative_allowed": bool(info.get("derivative_allowed", False)),
            "policy_status": str(info.get("policy_status", "unknown")),
        }

    return {
        "n_checksums": int(len(checksums)),
        "n_unknown": int(n_unknown),
        "n_disallowed": int(n_disallowed),
        "license_counts": _sorted_counts(counts),
        "by_checksum": details,
    }


def filter_checksums_by_license(
    checksums: list[str],
    attrib_map: Mapping[str, Mapping[str, object]],
    *,
    cfg: Mapping[str, object] | None = None,
) -> tuple[list[str], dict]:
    policy = license_policy_from_config(cfg)
    mode = str(policy.get("mode") or "exclude")
    allowed_from_cfg = policy.get("allowed_license_ids")

    kept: list[str] = []
    excluded: dict[str, dict] = {}
    would_exclude: dict[str, dict] = {}
    reason_counts: dict[str, int] = {}
    lids_by_checksum: dict[str, str] = {}

    for csum in sorted(list(checksums)):
        info = _license_info_for_checksum(csum, attrib_map)
        lids_by_checksum[csum] = str(info.get("license_id", "UNKNOWN"))

    if isinstance(allowed_from_cfg, list) and allowed_from_cfg:
        allowed_ids = {str(x).strip().upper() for x in allowed_from_cfg}
    else:
        allowed_ids = {lid.upper() for lid in lids_by_checksum.values() if _is_cc_license_id(lid)}

    for csum in sorted(list(checksums)):
        lid = lids_by_checksum[csum]
        reasons = policy_reasons_for_license(lid, policy, allowed_ids=allowed_ids)
        if not reasons:
            kept.append(csum)
            continue

        row = {"license_id": lid, "reasons": list(sorted(reasons))}
        for r in reasons:
            reason_counts[r] = int(reason_counts.get(r, 0)) + 1

        if mode == "warn":
            kept.append(csum)
            would_exclude[csum] = row
            continue

        if mode == "exclude":
            excluded[csum] = row
            continue

        # fail mode
        raise ValueError(f"License policy rejected checksum {csum}: {lid} ({', '.join(sorted(reasons))})")

    report = {
        "policy": {
            **policy,
            "allowed_license_ids": sorted(list(allowed_ids)),
            "allowed_license_ids_source": "config"
            if isinstance(allowed_from_cfg, list) and len(allowed_from_cfg) > 0
            else "observed_cc",
        },
        "n_input": int(len(checksums)),
        "n_kept": int(len(kept)),
        "n_excluded": int(len(excluded)),
        "reason_counts": _sorted_counts(reason_counts),
        "excluded": excluded,
    }
    if would_exclude:
        report["would_exclude"] = would_exclude
    return kept, report
