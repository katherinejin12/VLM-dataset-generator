#!/usr/bin/env bash
# Hardcoded: enumerate subfolders under Isaac assets and create marker .txt files

set -euo pipefail

# Hardcoded base path + name
BASE_PATH="/home/deepak/Downloads/isaac_sim_assets_4_5/isaac-sim-assets-1@4.5.0-rc.36+release.19112.f59b3005/Assets/Isaac/4.5"
NAME="test"

# Directory to scan (this is where Environments, Robots, People, etc live)
SCAN_DIR="$BASE_PATH/$NAME"

# Output root
OUT_ROOT="./data/$NAME"

# Create the category folders like before
mkdir -p "$OUT_ROOT"/{env,robots,people,props,sensors,materials,samples,isaaclab}

echo "[INFO] Base path : $BASE_PATH"
echo "[INFO] Name      : $NAME"
echo "[INFO] Scan dir  : $SCAN_DIR"
echo "[INFO] Out root  : $OUT_ROOT"

# Ensure scan dir exists
if [[ ! -d "$SCAN_DIR" ]]; then
  echo "[ERROR] Directory not found: $SCAN_DIR" >&2
  exit 1
fi

# One-level enumeration: for each immediate subfolder, make a marker file
shopt -s nullglob dotglob
count=0
for dir in "$SCAN_DIR"/*/ ; do
  sub="$(basename "$dir")"
  ls -1 "$dir" > "$OUT_ROOT/${sub}.txt"
  echo "[OK] created: $OUT_ROOT/${sub}.txt (from $dir)"
  ((count+=1))
done
shopt -u nullglob dotglob

echo "[DONE] Created $count marker file(s) under $OUT_ROOT"