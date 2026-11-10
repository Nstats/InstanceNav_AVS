#!/usr/bin/env python3
"""Parse fixed InstanceNav episodes into a structured factor list (offline, one-shot).

Each episode's goal description is decomposed into
  - category   : the goal object name (episode object_category)
  - attributes : independently judgeable properties of the target itself
  - relations  : constraints linking the target to a reference object/context

Every attribute/relation must carry `source_span`, a verbatim substring of the
original description, so the file can be checked against the source paragraph.
No navigation code reads this file yet.
"""
import argparse
import gzip
import hashlib
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PROMPT_VERSION = "factors-v1"

SYSTEM = """You decompose a target-instance description into atomic constraints.

Rules:
1. `attributes` are properties of the TARGET OBJECT ITSELF that could be judged from
   one clear view: material, colour, shape, size, part-and-colour pairs, markings.
   Split whenever a sub-part is named separately, even inside one sentence.
   Example: "Its backrest is brown, the seat is black" ->
     ["brown backrest", "black seat"]
   Emit each attribute as {"attribute": <what is described, e.g. "backrest color">,
   "value": <the judged value, e.g. "brown">}. Keep both short and self-contained;
   do not copy whole sentences.
2. `relations` are constraints on the target's surroundings: position relative to a
   reference object, co-location with a reference, or a stated reference attribute
   that the target must be accompanied by. Each relation names ONE reference object.
   Example: "placed next to a white round table" ->
     relation "next to a white round table", reference {"category": "table",
     "attributes": ["white", "round"]}
3. Ignore filler, camera instructions, and statements that carry no judgeable
   content ("the picture does not provide enough information ...", "the floor is
   made of wooden boards" when the floor is not a reference the target sits on).
   If the text explicitly says surrounding information is unavailable, return an
   empty relations list.
4. Every item MUST include `source_span`: a verbatim, case-sensitive substring of the
   input text that supports it. Never paraphrase inside `source_span`.
5. Do not invent constraints that are not in the text. Prefer 1-6 attributes and
   0-3 relations. Return JSON only."""

SCHEMA = {
    "type": "object",
    "properties": {
        "attributes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "attribute": {"type": "string"},
                    "value": {"type": "string"},
                    "source_span": {"type": "string"},
                },
                "required": ["attribute", "value", "source_span"],
                "additionalProperties": False,
            },
        },
        "relations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "source_span": {"type": "string"},
                    "reference": {
                        "type": "object",
                        "properties": {
                            "category": {"type": "string"},
                            "attributes": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["category", "attributes"],
                        "additionalProperties": False,
                    },
                },
                "required": ["text", "source_span", "reference"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["attributes", "relations"],
    "additionalProperties": False,
}

RESPONSE_FORMAT = {"type": "json_schema", "json_schema": {
    "name": "target_factors", "strict": True, "schema": SCHEMA}}

UNINFORMATIVE = re.compile(
    r"not provide enough information|impossible to describe|no information about the surrounding",
    re.IGNORECASE)


def build_prompt(category, intrinsic, extrinsic):
    return (
        f"TARGET CATEGORY: {category}\n\n"
        f"INTRINSIC DESCRIPTION (the target object itself):\n{intrinsic}\n\n"
        f"EXTRINSIC DESCRIPTION (the target's surroundings):\n{extrinsic}\n\n"
        "Return one JSON object with keys \"attributes\" and \"relations\" as specified. "
        "Each item needs a verbatim source_span copied from the text above. JSON only.")


def normalise(text):
    return re.sub(r"\s+", " ", text or "").strip()


def check_span(span, haystack):
    """Return True when span occurs verbatim (whitespace-normalised)."""
    span = normalise(span)
    if not span:
        return False
    return span.lower() in normalise(haystack).lower()


def validate_payload(data, intrinsic, extrinsic):
    if not isinstance(data, dict) or set(data) != {"attributes", "relations"}:
        raise ValueError("expected exactly {attributes, relations}")
    source = f"{intrinsic}\n{extrinsic}"
    for key in ("attributes", "relations"):
        if not isinstance(data[key], list):
            raise ValueError(f"{key} must be a list")
    for item in data["attributes"]:
        if set(item) != {"attribute", "value", "source_span"}:
            raise ValueError("attribute item fields")
        if not item["attribute"].strip() or not item["value"].strip():
            raise ValueError("attribute text empty")
        if not check_span(item["source_span"], source):
            raise ValueError(f"attribute source_span not verbatim: {item['source_span']!r}")
    for item in data["relations"]:
        if set(item) != {"text", "source_span", "reference"}:
            raise ValueError("relation item fields")
        if not check_span(item["source_span"], source):
            raise ValueError(f"relation source_span not verbatim: {item['source_span']!r}")
        ref = item["reference"]
        if set(ref) != {"category", "attributes"} or not ref["category"].strip():
            raise ValueError("relation reference incomplete")
        if not isinstance(ref["attributes"], list):
            raise ValueError("reference attributes must be a list")


def extract_json(answer):
    answer = re.sub(r"^```(?:json)?\s*|\s*```$", "", answer.strip(), flags=re.IGNORECASE)
    try:
        return json.loads(answer)
    except json.JSONDecodeError:
        start, end = answer.find("{"), answer.rfind("}")
        if start >= 0 and end > start:
            return json.loads(answer[start:end + 1])
        raise


def parse_one(client, category, intrinsic, extrinsic):
    prompt = build_prompt(category, intrinsic, extrinsic)
    last_error = None
    for attempt in range(3):
        request = prompt if attempt == 0 else (
            prompt + f"\nFormatting reminder {attempt}: JSON object only, both keys present, "
                     "every source_span copied verbatim from the text.")
        try:
            answer = client.complete(request, response_format=RESPONSE_FORMAT,
                                     validator=lambda a: validate_payload(extract_json(a), intrinsic, extrinsic))
            data = extract_json(answer)
            validate_payload(data, intrinsic, extrinsic)
            return data, None
        except Exception as error:  # noqa: BLE001 - recorded per episode
            last_error = f"{type(error).__name__}: {error}"
    return None, last_error


def load_episodes(path):
    with gzip.open(path, "rt") as handle:
        payload = json.load(handle)
    episodes = payload["episodes"]
    keyed = []
    for index, ep in enumerate(episodes):
        scene = Path(ep["scene_id"]).name.split(".")[0]
        key = f"{scene}_{ep['goal_object_id']}"
        attrs = ep.get("goal_attributes") or {}
        keyed.append({
            "index": index,
            "key": key,
            "category": ep["object_category"],
            "intrinsic": str(attrs.get("intrinsic_attributes", "")).strip(),
            "extrinsic": str(attrs.get("extrinsic_attributes", "")).strip(),
            "episode_id": str(ep.get("episode_id")),
            "scene_id": ep["scene_id"],
        })
    return keyed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(ROOT / "data/datasets/instancenav/val10/val10.json.gz"))
    parser.add_argument("--out-dir", default=str(ROOT / "data/instancenav_factors"))
    parser.add_argument("--name", default="val10_factors_v1")
    parser.add_argument("--limit", type=int, default=0, help="0 = all episodes")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    from utils.model_client import get_model_client
    client = get_model_client()
    data_path = Path(args.data)
    episodes = load_episodes(data_path)
    if args.limit:
        episodes = episodes[:args.limit]

    print(f"[Parse] episodes={len(episodes)} model={client.model} prompt={PROMPT_VERSION}")

    def work(ep):
        if UNINFORMATIVE.search(ep["extrinsic"]) and not ep["intrinsic"]:
            return ep, {"attributes": [], "relations": []}, None
        return (ep, *parse_one(client, ep["category"], ep["intrinsic"], ep["extrinsic"]))

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        results = list(pool.map(work, episodes))

    records, failures = {}, []
    for ep, data, error in results:
        factors = []
        if data is None:
            status = "fallback"
            factors.append({"id": "a0", "kind": "attribute", "attribute": "category",
                            "value": ep["category"], "text": ep["category"],
                            "source_span": "", "review_status": "fallback"})
            failures.append({"key": ep["key"], "error": error})
        else:
            status = "pending"
            for i, item in enumerate(data["attributes"], 1):
                factors.append({"id": f"a{i}", "kind": "attribute",
                                "attribute": item["attribute"], "value": item["value"],
                                "text": f'{item["value"]} {item["attribute"]}'.strip(),
                                "source_span": item["source_span"], "review_status": "pending"})
            for i, item in enumerate(data["relations"], 1):
                factors.append({"id": f"r{i}", "kind": "relation", "text": item["text"],
                                "source_span": item["source_span"],
                                "reference": {"category": item["reference"]["category"],
                                              "attributes": item["reference"]["attributes"]},
                                "review_status": "pending"})
        records[ep["key"]] = {
            "episode_index": ep["index"],
            "episode_id": ep["episode_id"],
            "scene_id": ep["scene_id"],
            "category": ep["category"],
            "relations_available": any(f["kind"] == "relation" for f in factors),
            "intrinsic_attributes": ep["intrinsic"],
            "extrinsic_attributes": ep["extrinsic"],
            "factors": factors,
            "review": {"status": status, "reviewer": None, "reviewed_at": None, "edits": 0},
        }

    payload = {
        "schema_version": "1",
        "source": {"dataset": "instancenav", "split": "val", "data_path": str(data_path),
                   "episode_count": len(records),
                   "sha256": hashlib.sha256(data_path.read_bytes()).hexdigest()},
        "model": {"backend": client.backend, "name": client.model, "prompt_version": PROMPT_VERSION},
        "episodes": records,
        "failures": failures,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{args.name}.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    csv_path = out_dir / f"{args.name}_review.csv"
    rows = ["episode_key,episode_index,category,factor_id,kind,text,source_span,reference_category,reference_attributes,review_status"]
    for key, rec in records.items():
        for f in rec["factors"]:
            ref = f.get("reference") or {}
            text = f["text"].replace('"', "'")
            span = f["source_span"].replace('"', "'")
            rows.append(",".join([
                key, str(rec["episode_index"]), rec["category"], f["id"], f["kind"],
                f'"{text}"', f'"{span}"',
                ref.get("category", ""), "|".join(ref.get("attributes", [])), f["review_status"]]))

    csv_path.write_text("\n".join(rows) + "\n")
    print(f"[Parse] wrote {json_path}")
    print(f"[Parse] wrote {csv_path}")
    print(f"[Parse] failures={len(failures)}")
    for item in failures[:5]:
        print(f"  {item['key']}: {item['error']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
