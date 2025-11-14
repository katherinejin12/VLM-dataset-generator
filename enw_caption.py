# pip install -U google-genai pandas
# export GOOGLE_API_KEY="your-key"

import os, json, time, glob, random
from typing import List, Dict, Tuple
import pandas as pd
from google import genai
from google.genai import types

# --------------- CONFIG ---------------
API_KEY = os.environ.get("GOOGLE_API_KEY")
if not API_KEY:
    raise EnvironmentError("Set GOOGLE_API_KEY in your environment first.")

MODEL_NAME   = "gemini-2.5-pro"
FALLBACK_MODEL = "gemini-1.5-flash"
TEMPERATURE  = 0.1
POLL_SECS    = 2
TIMEOUT_SECS = 300

CSV_PATH   = "/home/deepak/WarehouseResultrobotnewmult2/seed_13765_numseed_935590915_np/trajectories/robots/World_Robots_Nova_Carter_trajectory.csv"
FRAMES_DIR = "/home/deepak/WarehouseResultrobotnewmult2/seed_13765_numseed_935590915_np/_World_Robots_Nova_Carter_chassis_link_sensors_front_hawk_left_camera_left/rgb"
OUT_DIR    = "/home/deepak/WarehouseResultrobotnewmult2/seed_13765_numseed_935590915_np/out_annotations"
os.makedirs(OUT_DIR, exist_ok=True)

FRAME_PREFIX = "rgb_"
FRAME_PAD    = 5
FRAME_OFFSET = 0
ALLOW_EXTS   = (".png", ".jpg", ".jpeg", ".webp")

FPS                 = 30
WINDOW_SECONDS      = 2
WINDOW_RADIUS       = (FPS*WINDOW_SECONDS)//2
STRIDE_FRAMES       = 30
SAMPLE_PER_WIN      = 2          # <-- start small to avoid overload; raise to 2 later if stable
WINDOW_COOLDOWN_SEC = 1.5        # <-- brief sleep between windows to throttle

# --------------- PROMPT ---------------
CAPTION_PROMPT = """
You are a robotics-aware VLM analyzing an egocentric warehouse sequence.

INPUTS:
- JSON with per-frame robot pose (meters, radians) for a short time window (~2s).
- A small set of RGB frames sampled from that window (start/mid/end and a few in between).

TASK:
Return strict JSON describing the robot's motion timeline and a concise likely instruction
consistent with both the images and numeric poses. Prefer "unclear" over guessing.
""".strip()

# --------------- CLIENT ---------------
client = genai.Client(api_key=API_KEY)
models = client.models

# --------------- HELPERS ---------------
def frame_path(f: int) -> str | None:
    n = f - FRAME_OFFSET
    if n < 0:
        return None
    for ext in ALLOW_EXTS:
        p = os.path.join(FRAMES_DIR, f"{FRAME_PREFIX}{n:0{FRAME_PAD}d}{ext}")
        if os.path.exists(p):
            return p
    hits = glob.glob(os.path.join(FRAMES_DIR, f"{FRAME_PREFIX}{n:0{FRAME_PAD}d}.*"))
    for h in hits:
        if os.path.splitext(h)[1].lower() in ALLOW_EXTS and os.path.exists(h):
            return h
    return None

def sample_frames_for_window(frames: List[int], k: int) -> List[int]:
    if len(frames) <= k:
        return frames
    idxs = [round(i*(len(frames)-1)/(k-1)) for i in range(k)]
    return [frames[i] for i in idxs]

def pose_json_part(rows: List[Dict]) -> types.Part:
    """Send pose context as plain text JSON string."""
    def r3(x): return float(f"{float(x):.3f}")
    ctx = {
        "fps": FPS,
        "window": [{
            "frame": int(r["frame"]),
            "time_s": r3(r["time_s"]),
            "pose": {
                "x": r3(r["tx"]), "y": r3(r["ty"]), "z": r3(r["tz"]),
                "yaw_rad": r3(r["yaw"])
            }
        } for r in rows]
    }
    return types.Part(text=json.dumps(ctx, separators=(",", ":")))

def wait_file_active(name: str, timeout_sec: int = TIMEOUT_SECS, poll_sec: int = POLL_SECS):
    start = time.time()
    while True:
        fo = client.files.get(name=name)
        state = fo.state if isinstance(fo.state, str) else getattr(fo.state, "name", "UNKNOWN")
        if state == "ACTIVE":
            return fo
        if state == "FAILED":
            raise RuntimeError(f"File {name} FAILED to process.")
        if time.time() - start > timeout_sec:
            raise TimeoutError(f"Timeout waiting for file {name} ACTIVE.")
        time.sleep(poll_sec)

def upload_images_get_parts(img_paths: List[str]) -> Tuple[List[types.Part], List[object]]:
    files, parts = [], []
    for p in img_paths:
        fo = client.files.upload(file=p)
        fo = wait_file_active(fo.name)
        files.append(fo)
        mime = "image/png"
        pl = p.lower()
        if pl.endswith(".jpg") or pl.endswith(".jpeg"):
            mime = "image/jpeg"
        elif pl.endswith(".webp"):
            mime = "image/webp"
        parts.append(types.Part.from_uri(file_uri=fo.uri, mime_type=mime))
    return parts, files

def call_gemini_with_retry(parts: List[types.Part], cfg: types.GenerateContentConfig, primary_model: str):
    """
    Retries on common overload/transient errors with exponential backoff + jitter.
    Falls back to a lighter model once if primary keeps failing.
    """
    TRANSIENT_MARKERS = (
        "INTERNAL",
        "RESOURCE_EXHAUSTED",
        "UNAVAILABLE",
        "ABORTED",
        "Deadline Exceeded",
        "temporarily overloaded",
        "please try again"
    )
    MAX_RETRIES = 8
    last_err = None

    for attempt in range(1, MAX_RETRIES+1):
        try:
            return models.generate_content(
                model=primary_model,
                contents=types.Content(parts=parts),
                config=cfg
            )
        except Exception as e:
            last_err = e
            msg = str(e)
            if any(tok in msg for tok in TRANSIENT_MARKERS) and attempt < MAX_RETRIES:
                backoff = min(1.5 * (2 ** attempt), 20.0) + random.uniform(0, 0.8)
                print(f"[WARN] {primary_model} overload/timeout — retry {attempt}/{MAX_RETRIES} in {backoff:.1f}s")
                time.sleep(backoff)
                continue
            break

    # One-shot fallback
    if primary_model != FALLBACK_MODEL:
        print(f"[WARN] Falling back to {FALLBACK_MODEL} for this window…")
        try:
            return models.generate_content(
                model=FALLBACK_MODEL,
                contents=types.Content(parts=parts),
                config=cfg
            )
        except Exception as e2:
            last_err = e2

    raise last_err

# --------------- CORE ---------------
def run_one_window(center_idx: int, df: pd.DataFrame, out_base: str):
    print(f"\n[DEBUG] --- Processing window centered at index {center_idx} ---")
    try:
        c_frame = int(df.iloc[center_idx]["frame"])
    except Exception as e:
        print(f"[ERROR] Could not read frame at index {center_idx}: {e}")
        return None, "frame_index_error"

    lo = max(0, center_idx - WINDOW_RADIUS)
    hi = min(len(df) - 1, center_idx + WINDOW_RADIUS)
    rows = df.iloc[lo:hi + 1].to_dict(orient="records")

    candidate_frames = [int(r["frame"]) for r in rows]
    chosen_frames = sample_frames_for_window(candidate_frames, SAMPLE_PER_WIN)
    img_paths = [frame_path(f) for f in chosen_frames if frame_path(f)]
    if not img_paths:
        print(f"[WARN] No images found in this window.")
        return None, "no_images_in_window"

    json_part = pose_json_part(rows)
    uploaded_files = []
    try:
        image_parts, uploaded_files = upload_images_get_parts(img_paths)
        parts = [types.Part(text=CAPTION_PROMPT), json_part] + image_parts
    except Exception as e:
        print(f"[ERROR] Failed building request parts: {e}")
        return None, "build_parts_failed"

    cfg = types.GenerateContentConfig(response_mime_type="application/json", temperature=TEMPERATURE)
    out_path = os.path.join(OUT_DIR, f"{out_base}_frame{c_frame:06d}.json")

    try:
        resp = call_gemini_with_retry(parts, cfg, primary_model=MODEL_NAME)
        raw = (resp.text or "").strip()
        data = json.loads(raw)  # expect valid JSON per prompt
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"[DEBUG] ✅ Saved JSON result to: {out_path}")
        return {
            "center_frame": c_frame,
            "window_start_frame": int(df.iloc[lo]["frame"]),
            "window_end_frame": int(df.iloc[hi]["frame"]),
            "images": img_paths,
            "result_path": out_path
        }, None
    except Exception as e:
        print(f"[ERROR] Model inference failed: {e}")
        return None, "inference_failed"
    finally:
        for fo in uploaded_files:
            try:
                client.files.delete(name=fo.name)
                print(f"[DEBUG] Deleted remote file {fo.name}")
            except Exception as de:
                print(f"[WARN] Could not delete remote file {getattr(fo, 'name', '?')}: {de}")

# --------------- MAIN ---------------
def main():
    df = pd.read_csv(CSV_PATH)
    required = {"time_s","frame","tx","ty","tz","roll","pitch","yaw"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"CSV missing columns. Need: {required}")
    df = df.sort_values("frame").reset_index(drop=True)

    records, base = [], os.path.splitext(os.path.basename(CSV_PATH))[0]
    for center_idx in range(0, len(df), STRIDE_FRAMES):
        rec, err = run_one_window(center_idx, df, out_base=base)
        if rec:
            records.append(rec)
            print(f"✓ {rec['center_frame']:06d} → {rec['result_path']}")
        else:
            print(f"✗ frame@idx {center_idx}: {err}")

        # throttle between windows to avoid overload
        time.sleep(WINDOW_COOLDOWN_SEC)

    jsonl_path = os.path.join(OUT_DIR, f"{base}_windows.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as jf:
        for r in records:
            jf.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Done. Index: {jsonl_path} ({len(records)} windows)")

if __name__ == "__main__":
    main()