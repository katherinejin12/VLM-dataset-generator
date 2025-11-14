# list_isaac_assets_s3.py
# Build all_envs.txt and per-category object lists from NVIDIA's public S3 (Isaac 4.5 asset roots)
import argparse
import os
import sys
from typing import List, Dict, Set

import boto3
from botocore import UNSIGNED
from botocore.config import Config
import re

BUCKET = "omniverse-content-production"
PREFIX_ROOT = "Assets/Isaac/5.0/Isaac/"
HTTP_BASE = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"

USD_EXT = ".usd"  # match Bash behavior (exclude .usda/.usdc/.usdz)

# Filenames to exclude in People/Characters
PEOPLE_CHAR_EXCLUDE_NAMES = {
    "Biped_Setup.usd",
}
PEOPLE_CHAR_EXCLUDE_PREFIXES = ("biped_demo",)
PEOPLE_CHAR_EXCLUDE_SUFFIXES = ("_meters.usd",)
PEOPLE_ANIM_SUFFIX = ".skelanim.usd"

# Category top-levels
GENERIC_DIRS = ["Robots", "Props", "Sensors", "Materials", "IsaacLab", "Samples"]

def s3_client():
    return boto3.client("s3", config=Config(signature_version=UNSIGNED), region_name="us-west-2")

def list_usd_keys_from_s3(prefix_root: str) -> List[str]:
    s3 = s3_client()
    paginator = s3.get_paginator("list_objects_v2")
    page_iter = paginator.paginate(Bucket=BUCKET, Prefix=prefix_root)
    keys: List[str] = []
    for page in page_iter:
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.lower().endswith(USD_EXT):
                keys.append(key)
    return keys

def url_for_key(key: str) -> str:
    return HTTP_BASE + key

def after_root(key: str, root: str) -> str:
    return key[len(root):] if key.startswith(root) else key

def has_common_noise(sub: str) -> bool:
    s = sub
    # Path-based noise
    if "/.thumbs/" in s:
        return True
    if "/Stage/" in s:
        return True
    if "/.usd.files/" in s:
        return True
    # Name-based noise
    lower = s.lower()
    if lower.endswith(".thumb.usd"):
        return True
    if lower.endswith(".usd.png"):
        return True
    if lower.endswith(".usd.last_generated"):
        return True
    return False

def is_one_level_deep_env(sub: str) -> bool:
    # Environments/[^/]+/[^/]+.usd (exactly two segments after "Environments")
    if not sub.startswith("Environments/"):
        return False
    parts = sub.split("/")
    # ["Environments", "<envdir>", "<file>"]
    return len(parts) == 3 and parts[-1].lower().endswith(USD_EXT)

def in_dir(sub: str, dirname: str) -> bool:
    return sub.startswith(dirname.rstrip("/") + "/")

def people_bucket(sub: str) -> str:
    """
    Returns one of:
      'people_characters', 'people_animations',
      'people_dh_packs', 'people_dh_packs_extended', 'people_misc', or '' if not under People/
    Applies Bash-like excludes for characters bucket.
    """
    if not in_dir(sub, "People"):
        return ""

    # DH packs (broad include)
    if in_dir(sub, "People/DH_Characters_Extended"):
        return "people_dh_packs_extended"
    if in_dir(sub, "People/DH_Characters"):
        return "people_dh_packs"

    # Animations
    if in_dir(sub, "People/Animations"):
        if sub.endswith(PEOPLE_ANIM_SUFFIX):
            return "people_animations"
        else:
            # Non-skelanim content under Animations goes to misc (matches bash intent: only take *.skelanim.usd here)
            return "people_misc"

    # Characters: exactly People/Characters/*/*.usd with excludes
    if in_dir(sub, "People/Characters"):
        parts = sub.split("/")
        # Expect ["People","Characters","<pack>","<file>"]
        if len(parts) == 4 and parts[-1].lower().endswith(USD_EXT):
            fname = parts[-1]
            # Exclude *.skelanim.usd
            if fname.endswith(PEOPLE_ANIM_SUFFIX):
                return "people_misc"
            # Exclude prefixes/suffixes/names
            stem = os.path.splitext(fname)[0]
            if fname in PEOPLE_CHAR_EXCLUDE_NAMES:
                return "people_misc"
            if any(fname.endswith(suf) for suf in PEOPLE_CHAR_EXCLUDE_SUFFIXES):
                return "people_misc"
            if any(stem.lower().startswith(pfx.lower()) for pfx in PEOPLE_CHAR_EXCLUDE_PREFIXES):
                return "people_misc"
            return "people_characters"
        else:
            # Deeper or shallower paths under Characters go to misc
            return "people_misc"

    # Everything else under People → misc
    return "people_misc"

def generic_bucket(sub: str) -> str:
    for d in GENERIC_DIRS:
        if in_dir(sub, d):
            return d.lower()  # robots, props, sensors, materials, isaaclab, samples
    return ""

def categorize(keys: List[str], prefix_root: str):
    # Prepare output buckets
    buckets: Dict[str, Set[str]] = {
        "environments": set(),
        "robots": set(),
        "props": set(),
        "sensors": set(),
        "materials": set(),
        "isaaclab": set(),
        "samples": set(),
        "people_characters": set(),
        "people_animations": set(),
        "people_dh_packs": set(),
        "people_dh_packs_extended": set(),
        "people_misc": set(),
        "other": set(),
    }

    for k in keys:
        sub = after_root(k, prefix_root)
        if has_common_noise(sub):
            continue

        # Environments: only keep one-level-deep USDs
        if is_one_level_deep_env(sub):
            buckets["environments"].add(url_for_key(k))
            continue
        elif sub.startswith("Environments/"):
            # Ignore deeper env files to match bash
            continue

        # People buckets
        if sub.startswith("People/"):
            b = people_bucket(sub)
            if b:
                buckets[b].add(url_for_key(k))
                continue

        # Generic dirs (Robots/Props/Sensors/Materials/IsaacLab/Samples)
        gb = generic_bucket(sub)
        if gb:
            buckets[gb].add(url_for_key(k))
            continue

        # Anything else
        buckets["other"].add(url_for_key(k))

    return buckets

def write_list(path: str, urls: Set[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for u in sorted(urls):
            f.write(u + "\n")

def main():
    ap = argparse.ArgumentParser(description="List Isaac assets from public S3 into grouped txt files (Bash-equivalent filters).")
    ap.add_argument("--prefix-root", default=PREFIX_ROOT, help="S3 prefix root (default: %(default)s)")
    ap.add_argument("--out-dir", default="./isaacsim-dataset-generator/data/web_assetlist", help="Output directory (default: %(default)s)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    objects_dir = os.path.join(args.out_dir, "objects")
    os.makedirs(objects_dir, exist_ok=True)

    print(f"[S3] Scanning s3://{BUCKET}/{args.prefix_root} ... (public, unsigned)")
    keys = list_usd_keys_from_s3(args.prefix_root)
    print(f"[S3] Found {len(keys)} USD-like assets before filtering.")

    buckets = categorize(keys, args.prefix_root)

    # all_envs.txt
    env_out = os.path.join(args.out_dir, "all_envs.txt")
    write_list(env_out, buckets["environments"])
    print(f"[OK] Environments: {len(buckets['environments'])} -> {env_out}")

    # objects/*.txt to mirror your Bash names
    mapping = {
        "robots": "robots.txt",
        "props": "props.txt",
        "sensors": "sensors.txt",
        "materials": "materials.txt",
        "isaaclab": "isaaclab.txt",
        "samples": "samples.txt",
        "people_characters": "people_characters.txt",
        "people_animations": "people_animations.txt",
        "people_dh_packs": "people_dh_packs.txt",
        "people_dh_packs_extended": "people_dh_packs_extended.txt",
        "people_misc": "people_misc.txt",
        # "other": optional; write if you care
    }

    for bucket, fname in mapping.items():
        out_path = os.path.join(objects_dir, fname)
        write_list(out_path, buckets[bucket])
        print(f"[OK] {bucket:20s}: {len(buckets[bucket]):6d} -> {out_path}")

    # Optional: also write "other"
    if buckets["other"]:
        other_out = os.path.join(objects_dir, "other.txt")
        write_list(other_out, buckets["other"])
        print(f"[OK] {'other':20s}: {len(buckets['other']):6d} -> {other_out}")

    # Bash-style summary
    print("\n[DONE] Asset lists written under", args.out_dir)
    print("========== Summary ==========")
    def line(label, count):
        print(f"{label:<12s} {count:>7d}")
    line("envs:", len(buckets["environments"]))
    line("robots:", len(buckets["robots"]))
    line("props:", len(buckets["props"]))
    line("sensors:", len(buckets["sensors"]))
    line("materials:", len(buckets["materials"]))
    line("isaaclab:", len(buckets["isaaclab"]))
    line("samples:", len(buckets["samples"]))
    line("people(base):", len(buckets["people_characters"]))
    line("people(anim):", len(buckets["people_animations"]))
    line("people(DH):", len(buckets["people_dh_packs"]))
    line("people(DH+):", len(buckets["people_dh_packs_extended"]))
    line("people(misc):", len(buckets["people_misc"]))
    if buckets["other"]:
        line("other:", len(buckets["other"]))
    print("=============================")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
