#!/usr/bin/env bash
# Recursively collect .usd assets by group into txt files and report counts.
# Filters out thumbnails and common noise; splits People into clear buckets.

ROOT="/home/deepak/Downloads/isaac_sim_assets_4_5/isaac-sim-assets-1@4.5.0-rc.36+release.19112.f59b3005/Assets/Isaac/4.5/Isaac"
OUT_DIR="./data/local_objects"
mkdir -p "$OUT_DIR/objects"

# Helper: generic USD finder with common noise filters
find_usd() {
  local folder="$1"
  local outfile="$2"
  find "$folder" -type f -name "*.usd" \
    -not -path "*/.thumbs/*" \
    -not -name "*.thumb.usd" \
    -not -name "*.usd.png" \
    -not -name "*.usd.last_generated" \
    -not -path "*/Stage/*" \
    > "$outfile"
}

echo "[INFO] Writing lists into: $OUT_DIR"

# --- Environments: keep just the real scene USDs one level inside each env dir
find "$ROOT/Environments" -mindepth 2 -maxdepth 2 -type f -name "*.usd" \
  -not -path "*/.thumbs/*" \
  -not -name "*.thumb.usd" \
  -not -name "*.usd.png" \
  -not -name "*.usd.last_generated" \
  -not -path "*/Stage/*" \
  > "$OUT_DIR/all_envs.txt"

# --- Robots, Props, Sensors, Materials, IsaacLab, Samples (generic filters)
find_usd "$ROOT/Robots"     "$OUT_DIR/objects/robots.txt"
find_usd "$ROOT/Props"      "$OUT_DIR/objects/props.txt"
find_usd "$ROOT/Sensors"    "$OUT_DIR/objects/sensors.txt"
find_usd "$ROOT/Materials"  "$OUT_DIR/objects/materials.txt"
find_usd "$ROOT/IsaacLab"   "$OUT_DIR/objects/isaaclab.txt"
find_usd "$ROOT/Samples"    "$OUT_DIR/objects/samples.txt"

# --- People: split into buckets ---------------------------------------------

# 1) Base character assets (the ones you likely want for spawning):
#    People/Characters/*/*.usd
#    Exclude animations, .usd.files internals, and meters/demo helpers.
find "$ROOT/People/Characters" -mindepth 2 -maxdepth 2 -type f -name "*.usd" \
  -not -path "*/.thumbs/*" \
  -not -path "*/.usd.files/*" \
  -not -name "*.skelanim.usd" \
  -not -name "*_meters.usd" \
  -not -name "biped_demo*.usd" \
  -not -name "Biped_Setup.usd" \
  > "$OUT_DIR/objects/people_characters.txt"

# 2) Animation clips (useful if you need to bind animations later)
find "$ROOT/People/Animations" -type f -name "*.usd" \
  -not -path "*/.thumbs/*" \
  -name "*.skelanim.usd" \
  > "$OUT_DIR/objects/people_animations.txt"

# 3) DH character packs (original + extended) — kept separate so they don’t
#    overwhelm the base list; these are many per-variant exports.
find "$ROOT/People/DH_Characters" -type f -name "*.usd" \
  -not -path "*/.thumbs/*" \
  -not -path "*/.usd.files/*" \
  > "$OUT_DIR/objects/people_dh_packs.txt"

find "$ROOT/People/DH_Characters_Extended" -type f -name "*.usd" \
  -not -path "*/.thumbs/*" \
  -not -path "*/.usd.files/*" \
  > "$OUT_DIR/objects/people_dh_packs_extended.txt"

# 4) Everything else under People (for completeness, but filtered)
find "$ROOT/People" -type f -name "*.usd" \
  -not -path "$ROOT/People/Characters/*" \
  -not -path "$ROOT/People/Animations/*" \
  -not -path "$ROOT/People/DH_Characters/*" \
  -not -path "$ROOT/People/DH_Characters_Extended/*" \
  -not -path "*/.thumbs/*" \
  -not -path "*/.usd.files/*" \
  > "$OUT_DIR/objects/people_misc.txt"

echo
echo "[DONE] Asset lists written under $OUT_DIR"
echo "========== Summary =========="
wc -l "$OUT_DIR/all_envs.txt"                             | sed 's/^/envs:       /'
wc -l "$OUT_DIR/objects/robots.txt"                       | sed 's/^/robots:     /'
wc -l "$OUT_DIR/objects/props.txt"                        | sed 's/^/props:      /'
wc -l "$OUT_DIR/objects/sensors.txt"                      | sed 's/^/sensors:    /'
wc -l "$OUT_DIR/objects/materials.txt"                    | sed 's/^/materials:  /'
wc -l "$OUT_DIR/objects/isaaclab.txt"                     | sed 's/^/isaaclab:   /'
wc -l "$OUT_DIR/objects/samples.txt"                      | sed 's/^/samples:    /'
wc -l "$OUT_DIR/objects/people_characters.txt"            | sed 's/^/people(base): /'
wc -l "$OUT_DIR/objects/people_animations.txt"            | sed 's/^/people(anim): /'
wc -l "$OUT_DIR/objects/people_dh_packs.txt"              | sed 's/^/people(DH):   /'
wc -l "$OUT_DIR/objects/people_dh_packs_extended.txt"     | sed 's/^/people(DH+):  /'
wc -l "$OUT_DIR/objects/people_misc.txt"                  | sed 's/^/people(misc): /'
echo "============================="