#!/usr/bin/env python3
"""Build the structured factor file from the hand-authored table and verify spans.

Every source_span must occur verbatim in the episode description; any mismatch
is reported and the build fails so the file can never silently drift.
"""
import gzip
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/datasets/instancenav/val10/val10.json.gz"
OUT = ROOT / "data/instancenav_factors/val10_factors_v1.json"
CSV = ROOT / "data/instancenav_factors/val10_factors_v1_review.csv"


def norm(text):
    return re.sub(r"\s+", " ", text or "").strip().lower()


def load_table():
    spec = importlib.util.spec_from_file_location("authored", ROOT / "tools/authored_factors_val10.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.FACTORS


def main():
    table = load_table()
    payload = json.load(gzip.open(DATA, "rt"))
    episodes = payload["episodes"]
    records, problems = {}, []
    for index, ep in enumerate(episodes):
        scene = Path(ep["scene_id"]).name.split(".")[0]
        # Two val10 episodes share one (scene, goal_object_id) pair (idx 33 and 65
        # both target yr17PDCnDDW_426); episode_id keeps their records distinct.
        key = f"{scene}_{ep['goal_object_id']}__ep{ep.get('episode_id')}"
        attrs = ep.get("goal_attributes") or {}
        source = norm(f"{attrs.get('intrinsic_attributes','')} {attrs.get('extrinsic_attributes','')}")
        entry = table[index]
        if entry["category"] != ep["object_category"]:
            problems.append(f"[{index}] category mismatch: {entry['category']} vs {ep['object_category']}")
        factors = []
        for i, (name, value, span) in enumerate(entry["attributes"], 1):
            if norm(span) not in source:
                problems.append(f"[{index}] attribute span not verbatim: {span!r}")
            factors.append({"id": f"a{i}", "kind": "attribute", "attribute": name,
                            "value": value, "source_span": span, "review_status": "authored"})
        for i, (rel, ref_cat, ref_attrs, span) in enumerate(entry["relations"], 1):
            if norm(span) not in source:
                problems.append(f"[{index}] relation span not verbatim: {span!r}")
            factors.append({"id": f"r{i}", "kind": "relation", "text": rel, "source_span": span,
                            "reference": {"category": ref_cat, "attributes": list(ref_attrs)},
                            "review_status": "authored"})
        records[key] = {
            "episode_index": index,
            "episode_id": str(ep.get("episode_id")),
            "goal_key": f"{scene}_{ep['goal_object_id']}",
            "scene_id": ep["scene_id"],
            "category": ep["object_category"],
            "relations_available": any(f["kind"] == "relation" for f in factors),
            "intrinsic_attributes": attrs.get("intrinsic_attributes", ""),
            "extrinsic_attributes": attrs.get("extrinsic_attributes", ""),
            "factors": factors,
            "notes": entry.get("notes", ""),
            "review": {"status": "needs_human_check" if entry.get("notes") else "pending",
                       "reviewer": None, "reviewed_at": None, "edits": 0},
        }

    if problems:
        print("[Build] FAILED span/category checks:")
        for p in problems[:40]:
            print("  ", p)
        print(f"[Build] {len(problems)} problems; file not written")
        return 1

    out = {
        "schema_version": "1",
        "authored_by": "claude (direct reading of descriptions, no model API)",
        "source": {"dataset": "instancenav", "split": "val", "data_path": str(DATA.relative_to(ROOT)),
                   "episode_count": len(records), "sha256": hashlib.sha256(DATA.read_bytes()).hexdigest()},
        "episodes": records,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2))

    rows = ["episode_key,episode_index,category,factor_id,kind,attribute,value,reference_category,"
            "reference_attributes,source_span,review_status"]
    for key, rec in records.items():
        for f in rec["factors"]:
            ref = f.get("reference") or {}
            def q(v):
                return '"' + str(v).replace('"', "'") + '"'
            rows.append(",".join([key, str(rec["episode_index"]), rec["category"], f["id"], f["kind"],
                                  q(f.get("attribute", "")), q(f.get("value", f.get("text", ""))),
                                  q(ref.get("category", "")), q("|".join(ref.get("attributes", []))),
                                  q(f["source_span"]), f["review_status"]]))
    CSV.write_text("\n".join(rows) + "\n")

    n_attr = sum(1 for r in records.values() for f in r["factors"] if f["kind"] == "attribute")
    n_rel = sum(1 for r in records.values() for f in r["factors"] if f["kind"] == "relation")
    flagged = sum(1 for r in records.values() if r["review"]["status"] == "needs_human_check")
    print(f"[Build] episodes={len(records)} attributes={n_attr} relations={n_rel}")
    print(f"[Build] relations_available={sum(1 for r in records.values() if r['relations_available'])}/100")
    print(f"[Build] flagged for human check={flagged}")
    print(f"[Build] wrote {OUT.relative_to(ROOT)}")
    print(f"[Build] wrote {CSV.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
