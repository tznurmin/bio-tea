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

import argparse
import hashlib
import json
import random
import sys
from fnmatch import fnmatch
from pathlib import Path

from .claims import verify_results_root_claims
from .example_set_tools import (
    build_label_palette,
    dumps_summary,
    load_examples_and_meta,
    pair_examples_by_id,
    render_example_inline_labels,
    render_example_text,
    sample_examples_by_group,
    summarize_examples,
    validate_example_set,
)


def _csv_list(v: str | None) -> list[str]:
    if not v:
        return []
    return [x.strip() for x in str(v).split(",") if x.strip()]


def _as_str_list(v) -> list[str]:
    if v is None:
        return []
    if isinstance(v, str):
        return _csv_list(v)
    if isinstance(v, (list, tuple, set)):
        return [str(x).strip() for x in v if str(x).strip()]
    raise ValueError("Expected string or list-like value")


def _load_qa_rules(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except Exception as e:  # pragma: no cover
            raise RuntimeError("PyYAML is required to load YAML QA rules") from e
        obj = yaml.safe_load(text)
    else:
        obj = json.loads(text)

    if isinstance(obj, dict):
        if isinstance(obj.get("qa"), dict) and isinstance(obj["qa"].get("rules"), list):
            raw_rules = obj["qa"]["rules"]
        elif isinstance(obj.get("rules"), list):
            raw_rules = obj["rules"]
        else:
            raise ValueError("QA rules file must contain 'rules' list or 'qa.rules' list")
    elif isinstance(obj, list):
        raw_rules = obj
    else:
        raise ValueError("QA rules file must be a list or object")

    rules: list[dict] = []
    for i, rr in enumerate(raw_rules):
        if not isinstance(rr, dict):
            raise ValueError(f"Rule at index {i} is not an object")
        pattern = str(rr.get("pattern") or "").strip()
        if not pattern:
            raise ValueError(f"Rule at index {i} missing 'pattern'")
        rules.append(
            {
                "name": str(rr.get("name") or f"rule_{i+1}"),
                "pattern": pattern,
                "require_categories": _as_str_list(rr.get("require_categories")),
                "require_raw_types": _as_str_list(rr.get("require_raw_types")),
                "max_tokens": int(rr["max_tokens"]) if rr.get("max_tokens") is not None else None,
            }
        )
    return rules


def _aggregate_counts(objs: list[dict], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for obj in objs:
        d = obj.get(key) or {}
        if not isinstance(d, dict):
            continue
        for k, v in d.items():
            out[str(k)] = int(out.get(str(k), 0)) + int(v)
    return dict(sorted(out.items(), key=lambda kv: kv[0]))


def _delta_counts(a: dict | None, b: dict | None) -> dict[str, int]:
    da = dict(a or {})
    db = dict(b or {})
    keys = sorted(set([str(k) for k in da.keys()]) | set([str(k) for k in db.keys()]))
    out: dict[str, int] = {}
    for k in keys:
        out[k] = int(da.get(k, 0)) - int(db.get(k, 0))
    return out


def _resolve_variant_stats_from_manifest(
    manifest: dict,
    *,
    variant: str,
    train_set: str | None,
) -> dict:
    ds = dict(manifest.get("dataset_stats") or {})
    train = ds.get("train")
    if not isinstance(train, dict):
        raise ValueError("manifest.dataset_stats.train is missing or not an object")

    # Base set manifests: dataset_stats.train.variants.<variant>
    variants_obj = train.get("variants")
    if isinstance(variants_obj, dict) and isinstance(variants_obj.get(variant), dict):
        return dict(variants_obj.get(variant) or {})

    # Alternate shape: dataset_stats.train.<variant>
    if isinstance(train.get(variant), dict):
        return dict(train.get(variant) or {})

    # Experiment manifests: dataset_stats.train.<set>.<variant>
    set_name = str(train_set or "").strip()
    if set_name:
        set_obj = train.get(set_name)
        if not isinstance(set_obj, dict):
            raise ValueError(f"manifest.dataset_stats.train.{set_name} is missing")
        if isinstance(set_obj.get("variants"), dict) and isinstance(set_obj.get("variants", {}).get(variant), dict):
            return dict(set_obj.get("variants", {}).get(variant) or {})
        if isinstance(set_obj.get(variant), dict):
            return dict(set_obj.get(variant) or {})
        raise ValueError(
            f"variant '{variant}' missing under manifest.dataset_stats.train.{set_name}"
        )

    # Heuristic for experiment manifests when only one train set exists.
    candidate_sets = [k for k, v in train.items() if isinstance(v, dict)]
    matches: list[dict] = []
    for k in candidate_sets:
        v = dict(train.get(k) or {})
        if isinstance(v.get("variants"), dict) and isinstance(v.get("variants", {}).get(variant), dict):
            matches.append(dict(v.get("variants", {}).get(variant) or {}))
        elif isinstance(v.get(variant), dict):
            matches.append(dict(v.get(variant) or {}))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            "Multiple train-set candidates found; pass --train-set to disambiguate"
        )
    raise ValueError(
        f"variant '{variant}' not found in manifest.dataset_stats.train"
    )


def _resolve_variant_materialization(manifest: dict, *, variant: str) -> dict | None:
    mat = manifest.get("materialization")
    if not isinstance(mat, dict):
        return None
    v = mat.get(variant)
    if not isinstance(v, dict):
        return None
    return dict(v)


def inspect_main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Inspect a generated example set with readable colored text.")
    ap.add_argument("--set", required=True, dest="set_path", type=str, help="Path to .set file")
    ap.add_argument("--meta", default=None, type=str, help="Optional path to .meta.jsonl file")
    ap.add_argument("--diff-with", default=None, type=str, help="Optional second .set file to compare by example_id")
    ap.add_argument("--diff-meta", default=None, type=str, help="Optional second .meta.jsonl path")
    ap.add_argument("--include-unchanged", action="store_true", help="In diff mode, include unchanged examples")
    ap.add_argument(
        "--group-by",
        default="types_raw",
        choices=["types_raw", "types_pooled", "category", "none"],
        help="How to group example sampling",
    )
    ap.add_argument("--samples-per-group", default=1, type=int)
    ap.add_argument("--seed", default=0, type=int)
    ap.add_argument("--no-color", action="store_true")
    args = ap.parse_args(argv)

    examples, meta = load_examples_and_meta(
        Path(args.set_path),
        Path(args.meta) if args.meta else None,
    )

    if args.diff_with:
        other_examples, other_meta = load_examples_and_meta(
            Path(args.diff_with),
            Path(args.diff_meta) if args.diff_meta else None,
        )
        pairs, stats = pair_examples_by_id(examples, meta, other_examples, other_meta)
        if not bool(args.include_unchanged):
            pairs = [p for p in pairs if bool(p.get("changed"))]
        if not pairs:
            print("No paired examples found for diff output.", file=sys.stderr)
            return 1

        base_examples = [p["base_example"] for p in pairs]
        base_meta = [p["base_meta"] for p in pairs]
        sampled = sample_examples_by_group(
            base_examples,
            base_meta,
            group_by=args.group_by,
            samples_per_group=int(args.samples_per_group),
            seed=int(args.seed),
        )
        all_labels = []
        for p in pairs:
            all_labels.extend([lab for _tok, lab in p["base_example"]])
            all_labels.extend([lab for _tok, lab in p["other_example"]])
        palette = build_label_palette(all_labels)

        print("Diff stats:")
        print(dumps_summary(stats))
    else:
        sampled = sample_examples_by_group(
            examples,
            meta,
            group_by=args.group_by,
            samples_per_group=int(args.samples_per_group),
            seed=int(args.seed),
        )
        all_labels = []
        for ex in examples:
            all_labels.extend([lab for _tok, lab in ex])
        palette = build_label_palette(all_labels)

    if not sampled:
        print("No examples found.", file=sys.stderr)
        return 1

    print("Legend:")
    if palette:
        for typ in sorted(palette.keys()):
            if args.no_color:
                print(f"  {typ}")
            else:
                print(f"  {palette[typ]}{typ}\x1b[0m")
    else:
        print("  (only O labels)")

    for grp in sorted(sampled.keys()):
        ids = sampled[grp]
        print(f"\n[{grp}]")
        for i in ids:
            if args.diff_with:
                # In diff mode sampled indices reference pairs.
                pair = pairs[i]
                bex = pair["base_example"]
                oex = pair["other_example"]
                btoks = [t for t, _l in bex]
                blabs = [l for _t, l in bex]
                otoks = [t for t, _l in oex]
                olabs = [l for _t, l in oex]
                line_base = render_example_text(btoks, blabs, palette=palette, color=(not args.no_color))
                line_diff = render_example_text(otoks, olabs, palette=palette, color=(not args.no_color))
                print(
                    f"- example_id={pair['example_id']} "
                    f"changed_tokens={pair['changed_tokens']} changed_labels={pair['changed_labels']}"
                )
                print(f"  base: {line_base}")
                print(f"  diff: {line_diff}")
            else:
                ex = examples[i]
                toks = [t for t, _l in ex]
                labs = [l for _t, l in ex]
                line = render_example_text(toks, labs, palette=palette, color=(not args.no_color))
                eid = None
                category = None
                types_raw = None
                if meta is not None and i < len(meta):
                    mr = meta[i]
                    eid = mr.get("example_id")
                    category = mr.get("category")
                    types_raw = mr.get("types_raw")
                if eid is not None:
                    print(f"- example_id={eid} category={category} types_raw={types_raw}")
                else:
                    print(f"- idx={i}")
                print(f"  {line}")

    return 0


def _stable_seed(seed: int, key: str) -> int:
    h = hashlib.sha1(str(key).encode("utf-8")).hexdigest()[:8]
    return int(seed) ^ int(h, 16)


def _discover_sample_sets(
    *,
    results_root: Path,
    tasks: list[str],
    set_names: list[str],
    splits: list[str],
    variants: list[str],
) -> list[dict]:
    task_filter = {str(x) for x in tasks if str(x)}
    set_filter = {str(x) for x in set_names if str(x)}
    split_list = [str(x) for x in splits if str(x)]
    variant_list = [str(x) for x in variants if str(x)]

    rows: list[dict] = []
    if not results_root.exists():
        return rows
    for task_dir in sorted([p for p in results_root.iterdir() if p.is_dir()]):
        task = str(task_dir.name)
        if task == "profiles":
            continue
        if task_filter and task not in task_filter:
            continue

        # Per-set split files: results/<task>/set*/<split>/<variant>.set
        for set_dir in sorted([p for p in task_dir.iterdir() if p.is_dir() and p.name.startswith("set")]):
            set_name = str(set_dir.name)
            if set_filter and set_name not in set_filter:
                continue
            for split in split_list:
                split_dir = set_dir / split
                if not split_dir.exists() or not split_dir.is_dir():
                    continue
                for variant in variant_list:
                    sp = split_dir / f"{variant}.set"
                    if sp.exists():
                        rows.append(
                            {
                                "task": task,
                                "set": set_name,
                                "split": split,
                                "variant": variant,
                                "set_path": sp,
                                "meta_path": sp.with_suffix(".meta.jsonl"),
                            }
                        )

        # Task-level split files: results/<task>/<split>/*.set (e.g. test/unaugmented.set)
        for split in split_list:
            split_dir = task_dir / split
            if not split_dir.exists() or not split_dir.is_dir():
                continue
            for sp in sorted([p for p in split_dir.glob("*.set") if p.is_file()]):
                rows.append(
                    {
                        "task": task,
                        "set": None,
                        "split": split,
                        "variant": str(sp.stem),
                        "set_path": sp,
                        "meta_path": sp.with_suffix(".meta.jsonl"),
                    }
                )

    rows.sort(key=lambda r: str(Path(r["set_path"])))
    return rows


def _sample_indices_by_category(
    *,
    meta_rows: list[dict] | None,
    n_per_category: int,
    seed: int,
) -> dict[str, list[int]]:
    if meta_rows is None:
        return {"unknown": []}
    cats: dict[str, list[int]] = {}
    for i, mr in enumerate(meta_rows):
        cat = str((mr or {}).get("category") or "unknown")
        cats.setdefault(cat, []).append(i)
    rng = random.Random(int(seed))
    picked: dict[str, list[int]] = {}
    n = max(0, int(n_per_category))
    for cat in sorted(cats.keys()):
        ids = list(cats[cat])
        rng.shuffle(ids)
        picked[cat] = sorted(ids[:n])
    return picked


def sample_main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Write representative inline-labeled samples across generated datasets.")
    ap.add_argument("--results-root", default="results", type=str, help="Results root directory")
    ap.add_argument(
        "--output",
        default=None,
        type=str,
        help="Output text artifact path (default: <results-root>/sample_examples.txt)",
    )
    ap.add_argument("--tasks", default=None, type=str, help="Optional comma-separated task filter")
    ap.add_argument("--sets", default=None, type=str, help="Optional comma-separated set filter (set1,set2,...)")
    ap.add_argument(
        "--splits",
        default="train,dev,test",
        type=str,
        help="Comma-separated split names to scan (default: train,dev,test)",
    )
    ap.add_argument(
        "--variants",
        default="none,all,combined,species,strains",
        type=str,
        help="Comma-separated variants for per-set split files",
    )
    ap.add_argument("--samples-per-category", default=1, type=int, help="How many examples per category per dataset")
    ap.add_argument("--seed", default=0, type=int, help="Random seed for sampling")
    ap.add_argument(
        "--render-lowercase",
        action="store_true",
        help="Render sampled text lowercased (useful for lowercase tokenizer views)",
    )
    ap.add_argument(
        "--require-categories",
        default=None,
        type=str,
        help="Optional comma-separated categories to require in each sampled dataset",
    )
    args = ap.parse_args(argv)

    results_root = Path(args.results_root)
    output_path = Path(args.output) if args.output else (results_root / "sample_examples.txt")
    tasks = _csv_list(args.tasks)
    set_names = _csv_list(args.sets)
    splits = _csv_list(args.splits)
    variants = _csv_list(args.variants)
    required_categories = [str(x) for x in _csv_list(args.require_categories)]

    datasets = _discover_sample_sets(
        results_root=results_root,
        tasks=tasks,
        set_names=set_names,
        splits=splits,
        variants=variants,
    )
    if not datasets:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": "no_matching_datasets",
                    "results_root": str(results_root),
                    "n_datasets": 0,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 1

    lines: list[str] = []
    lines.append("# Bio-TEA Sample Examples")
    lines.append(f"results_root: {results_root}")
    lines.append(f"samples_per_category: {int(args.samples_per_category)}")
    lines.append(f"seed: {int(args.seed)}")
    lines.append(f"splits: {','.join(splits)}")
    lines.append(f"variants: {','.join(variants)}")
    lines.append("")

    n_sets_written = 0
    n_examples_written = 0
    missing_required: dict[str, list[str]] = {}
    for row in datasets:
        set_path = Path(row["set_path"])
        meta_path = Path(row["meta_path"])
        rel = str(set_path.relative_to(results_root))
        examples, meta = load_examples_and_meta(set_path, meta_path if meta_path.exists() else None)
        if not examples:
            continue

        local_seed = _stable_seed(int(args.seed), rel)
        sampled = _sample_indices_by_category(
            meta_rows=meta,
            n_per_category=int(args.samples_per_category),
            seed=local_seed,
        )
        if meta is None:
            sampled = {"unknown": list(range(min(int(args.samples_per_category), len(examples))))}

        observed_categories = sorted(sampled.keys())
        required = list(required_categories) if required_categories else observed_categories
        missing = [c for c in required if c not in sampled]
        if missing:
            missing_required[rel] = list(missing)

        n_sets_written += 1
        lines.append(
            "## "
            + f"task={row['task']} set={row['set'] or 'global'} split={row['split']} "
            + f"variant={row['variant']} path={rel}"
        )
        lines.append(f"categories_observed: {','.join(observed_categories) if observed_categories else 'none'}")
        if missing:
            lines.append(f"missing_required_categories: {','.join(missing)}")
        for cat in required:
            ids = list(sampled.get(cat) or [])
            if not ids:
                continue
            lines.append(f"### category={cat}")
            for i in ids:
                if i < 0 or i >= len(examples):
                    continue
                ex = examples[i]
                tokens = [str(t).lower() for t, _lab in ex] if bool(args.render_lowercase) else [t for t, _lab in ex]
                labels = [lab for _t, lab in ex]
                rendered = render_example_inline_labels(tokens, labels)
                eid = None
                if meta is not None and i < len(meta):
                    eid = str((meta[i] or {}).get("example_id") or "")
                lines.append(f"- idx={i} example_id={eid or 'na'} category={cat}")
                lines.append(f"  {rendered}")
                n_examples_written += 1
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    out = {
        "ok": True,
        "results_root": str(results_root),
        "output": str(output_path),
        "n_datasets": int(n_sets_written),
        "n_examples": int(n_examples_written),
        "render_lowercase": bool(args.render_lowercase),
        "missing_required_categories": missing_required,
    }
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


def validate_main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Validate generated example set before training.")
    ap.add_argument("--set", required=True, dest="set_path", type=str, help="Path to .set file")
    ap.add_argument("--meta", default=None, type=str, help="Optional path to .meta.jsonl file")
    ap.add_argument("--max-tokens", default=None, type=int, help="Optional max token threshold")
    ap.add_argument(
        "--require-categories",
        default=None,
        type=str,
        help="Comma-separated required categories (from metadata category field)",
    )
    ap.add_argument(
        "--require-raw-types",
        default=None,
        type=str,
        help="Comma-separated required raw type codes (from metadata types_raw field)",
    )
    args = ap.parse_args(argv)

    examples, meta = load_examples_and_meta(
        Path(args.set_path),
        Path(args.meta) if args.meta else None,
    )
    summary = validate_example_set(
        examples,
        meta,
        max_tokens=args.max_tokens,
        required_categories=_csv_list(args.require_categories),
        required_raw_types=_csv_list(args.require_raw_types),
    )
    print(dumps_summary(summary))
    return 0 if bool(summary.get("ok")) else 2


def stats_main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Summarize label/category distributions for an example set.")
    ap.add_argument("--set", required=True, dest="set_path", type=str, help="Path to .set file")
    ap.add_argument("--meta", default=None, type=str, help="Optional path to .meta.jsonl file")
    args = ap.parse_args(argv)

    examples, meta = load_examples_and_meta(
        Path(args.set_path),
        Path(args.meta) if args.meta else None,
    )
    print(dumps_summary(summarize_examples(examples, meta)))
    return 0


def qa_main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Validate all generated .set files under a results root.")
    ap.add_argument("--results-root", default="results", type=str, help="Results root directory")
    ap.add_argument("--pattern", default="**/*.set", type=str, help="Glob pattern relative to results root")
    ap.add_argument(
        "--rules",
        default=None,
        type=str,
        help="Optional JSON/YAML rules file with per-pattern validation requirements",
    )
    ap.add_argument("--output", default=None, type=str, help="Optional output JSON path")
    ap.add_argument("--max-tokens", default=None, type=int, help="Optional max token threshold")
    ap.add_argument("--require-categories", default=None, type=str, help="Comma-separated required categories")
    ap.add_argument("--require-raw-types", default=None, type=str, help="Comma-separated required raw type codes")
    ap.add_argument("--fail-on-error", action="store_true", help="Return non-zero if any set validation fails")
    args = ap.parse_args(argv)

    root = Path(args.results_root)
    set_paths = sorted([p for p in root.glob(args.pattern) if p.is_file()])
    rules: list[dict] = []
    if args.rules:
        rules = _load_qa_rules(Path(args.rules))

    def pick_rule(rel_path: str) -> dict | None:
        for rule in rules:
            if fnmatch(rel_path, str(rule["pattern"])):
                return rule
        return None

    if not set_paths:
        summary = {
            "results_root": str(root),
            "pattern": str(args.pattern),
            "rules_file": str(args.rules) if args.rules else None,
            "rules": rules,
            "n_sets": 0,
            "n_ok": 0,
            "n_failed": 0,
            "failed_sets": [],
            "sets": {},
        }
        out = json.dumps(summary, indent=2, sort_keys=True)
        if args.output:
            Path(args.output).write_text(out + "\n", encoding="utf-8")
        print(out)
        return 1

    sets: dict[str, dict] = {}
    summaries: list[dict] = []
    failed_sets: list[str] = []
    for p in set_paths:
        rel = str(p.relative_to(root))
        rule = pick_rule(rel)
        req_cats = list(rule.get("require_categories") or []) if rule else _csv_list(args.require_categories)
        req_types = list(rule.get("require_raw_types") or []) if rule else _csv_list(args.require_raw_types)
        max_tokens = rule.get("max_tokens") if rule else args.max_tokens

        examples, meta = load_examples_and_meta(p, None)
        s = validate_example_set(
            examples,
            meta,
            max_tokens=max_tokens,
            required_categories=req_cats,
            required_raw_types=req_types,
        )
        s["qa_rule"] = (
            {"name": rule["name"], "pattern": rule["pattern"]} if rule else {"name": "default", "pattern": None}
        )
        sets[rel] = s
        summaries.append(s)
        if not bool(s.get("ok")):
            failed_sets.append(rel)

    summary = {
        "results_root": str(root),
        "pattern": str(args.pattern),
        "rules_file": str(args.rules) if args.rules else None,
        "rules": rules,
        "n_sets": int(len(set_paths)),
        "n_ok": int(len(set_paths) - len(failed_sets)),
        "n_failed": int(len(failed_sets)),
        "failed_sets": sorted(failed_sets),
        "totals": {
            "n_examples": int(sum(int(s.get("n_examples") or 0) for s in summaries)),
            "n_tokens_total": int(sum(int(s.get("n_tokens_total") or 0) for s in summaries)),
            "category_counts": _aggregate_counts(summaries, "category_counts"),
            "label_counts": _aggregate_counts(summaries, "label_counts"),
            "types_raw_counts": _aggregate_counts(summaries, "types_raw_counts"),
        },
        "sets": sets,
    }
    out = json.dumps(summary, indent=2, sort_keys=True)
    if args.output:
        op = Path(args.output)
        op.parent.mkdir(parents=True, exist_ok=True)
        op.write_text(out + "\n", encoding="utf-8")
    print(out)

    if args.fail_on_error and failed_sets:
        return 2
    return 0


def manifest_compare_main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Compare two train variants from a dataset manifest (for example none vs all)."
    )
    ap.add_argument("--manifest", default=None, type=str, help="Path to manifest.json")
    ap.add_argument("--results-root", default="results", type=str, help="Results root (used when --manifest omitted)")
    ap.add_argument("--task", default=None, type=str, help="Task name (used when --manifest omitted)")
    ap.add_argument("--set", dest="set_name", default=None, type=str, help="Set name (used when --manifest omitted)")
    ap.add_argument("--train-set", default=None, type=str, help="Train set key for experiment manifests")
    ap.add_argument("--variant-a", default="none", type=str)
    ap.add_argument("--variant-b", default="all", type=str)
    ap.add_argument("--output", default=None, type=str, help="Optional output JSON path")
    args = ap.parse_args(argv)

    if args.manifest:
        manifest_path = Path(args.manifest)
    else:
        task = str(args.task or "").strip()
        set_name = str(args.set_name or "").strip()
        if not task or not set_name:
            raise ValueError("Provide --manifest or both --task and --set")
        manifest_path = Path(args.results_root) / task / set_name / "manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest file not found: {manifest_path}")

    manifest = dict(json.loads(manifest_path.read_text(encoding="utf-8")) or {})
    va = str(args.variant_a).strip()
    vb = str(args.variant_b).strip()
    train_set = str(args.train_set).strip() if args.train_set else None

    stats_a = _resolve_variant_stats_from_manifest(manifest, variant=va, train_set=train_set)
    stats_b = _resolve_variant_stats_from_manifest(manifest, variant=vb, train_set=train_set)
    mat_a = _resolve_variant_materialization(manifest, variant=va)
    mat_b = _resolve_variant_materialization(manifest, variant=vb)

    out = {
        "manifest_path": str(manifest_path),
        "task": str(manifest.get("task") or ""),
        "set": str(manifest.get("set") or ""),
        "train_set": train_set,
        "variant_a": va,
        "variant_b": vb,
        "dataset_stats": {
            va: stats_a,
            vb: stats_b,
        },
        "materialization": {
            va: mat_a,
            vb: mat_b,
        },
        "quick_delta": {
            "n_examples": int(stats_a.get("n_examples", 0)) - int(stats_b.get("n_examples", 0)),
            "n_tokens_total": int(stats_a.get("n_tokens_total", 0)) - int(stats_b.get("n_tokens_total", 0)),
            "category_counts": _delta_counts(stats_a.get("category_counts"), stats_b.get("category_counts")),
            "label_counts": _delta_counts(stats_a.get("label_counts"), stats_b.get("label_counts")),
            "label_type_example_counts": _delta_counts(
                stats_a.get("label_type_example_counts"), stats_b.get("label_type_example_counts")
            ),
            "label_type_token_counts": _delta_counts(
                stats_a.get("label_type_token_counts"), stats_b.get("label_type_token_counts")
            ),
            "types_raw_counts": _delta_counts(stats_a.get("types_raw_counts"), stats_b.get("types_raw_counts")),
        },
    }

    payload = json.dumps(out, indent=2, sort_keys=True)
    if args.output:
        op = Path(args.output)
        op.parent.mkdir(parents=True, exist_ok=True)
        op.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


def claims_main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Verify that generated dataset artifacts satisfy their documented split and exclusion claims."
    )
    ap.add_argument("--results-root", default="results", type=str, help="Results root directory")
    ap.add_argument("--tasks", default=None, type=str, help="Optional comma-separated task filter")
    ap.add_argument("--experiments", default=None, type=str, help="Optional comma-separated experiment filter")
    ap.add_argument(
        "--calibration-results-root",
        default=None,
        type=str,
        help="Optional calibration results root to verify campaign train=train+dev and test preservation",
    )
    ap.add_argument(
        "--calibration-eval-split",
        default="dev",
        type=str,
        help="Calibration eval split name that should be merged into final campaign train",
    )
    ap.add_argument("--output", default=None, type=str, help="Optional output JSON path")
    ap.add_argument("--fail-on-error", action="store_true", help="Return non-zero if any claim check fails")
    args = ap.parse_args(argv)

    report = verify_results_root_claims(
        Path(args.results_root),
        tasks=_csv_list(args.tasks),
        experiments=_csv_list(args.experiments),
        calibration_results_root=Path(args.calibration_results_root) if args.calibration_results_root else None,
        calibration_eval_split=str(args.calibration_eval_split),
    )
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    if args.fail_on_error and not bool(report.get("ok")):
        return 2
    return 0
