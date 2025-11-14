# pip install -U google-genai
# export GOOGLE_API_KEY="your-key"

import os
import time
import json
from google import genai
from google.genai import types

# --------------- CONFIG ---------------
API_KEY = os.environ.get("GOOGLE_API_KEY")
if not API_KEY:
    raise EnvironmentError("Set GOOGLE_API_KEY in your environment first.")

VIDEO_PATH = "/home/deepak/WarehouseResultrobotnewmult2/seed_13765_numseed_935590915_np/_World_Robots_Nova_Carter_chassis_link_sensors_front_hawk_left_camera_left/videos/simulation.mp4"
MODEL_NAME = "gemini-2.5-pro"
POLL_SECS = 2
TIMEOUT_SECS = 300  # 5 minutes
TEMPERATURE = 0.1

CAPTION_PROMPT = """
You are a robotics-aware VLM analyzing an egocentric warehouse video.
Report ONLY what pixels support. If uncertain, say "unclear".

OUTPUT: strict JSON with EXACTLY two top-level fields: "robot_motion" and "environment_dynamics".

Before writing the summary, construct a timestamped event timeline. The summary MUST be consistent with the timeline.

Direction Disambiguation (MANDATORY):
- Determine left/right turns by global background motion: if distant static features (walls/shelves) move LEFT across the frame, the robot is turning RIGHT (positive yaw). If they move RIGHT, the robot is turning LEFT. If evidence is weak, use "slight" with low confidence or "unclear".

Angle Bins:
- Use one of: "slight (0–20°)", "moderate (20–60°)", "sharp (60–120°)", or "unclear".
- Provide a numeric estimate in degrees (right positive) when possible; else null.

Schema:
{
  "robot_motion": {
    "timeline": [
      {
        "t_s": 0.0,
        "action": "stationary | forward | reverse | turn-right | turn-left | stop | collision-contact | near-contact | unclear",
        "turn_angle_bin": "slight (0–20°) | moderate (20–60°) | sharp (60–120°) | unclear",
        "turn_angle_deg_est": 0.0,
        "details": "fact-only cue, e.g., 'global background flow left ⇒ right turn', 'front bumper touches wall'",
        "confidence": 0.0
      }
    ],
    "summary": "1–3 short sentences that match the timeline (no contradictions).",
    "likely_instruction": "High-level human-issued command inferred from FULL trajectory (e.g., 'proceed to the shelf ahead'). If insufficient cues, 'uncertain'. Do NOT invent shelf counts or area names.",
    "evidence": "Very brief visual cues that support the likely_instruction.",
    "confidence": 0.0
  },
  "environment_dynamics": {
    "static_layout": "Only visible static structures (walls, shelves, aisle). No invented labels.",
    "actors": [
      {
        "category": "human | forklift | robot | other",
        "state": "moving | stationary | unknown",
        "direction": "frame-left | frame-right | toward-camera | away-from-camera | unclear",
        "notes": "optional"
      }
    ]
  }
}

Verification (MANDATORY before returning):
- If any timeline event implies RIGHT turn, the summary must not say LEFT (and vice versa).
- If collision/contact is visible, include a 'collision-contact' event with approximate timestamp.
- Prefer 'unclear' over guessing.
"""
# --------------- CLIENT ---------------
client = genai.Client(api_key=API_KEY)

# --------------- UPLOAD + WAIT ---------------
print(f"Uploading {VIDEO_PATH}...")
file_obj = client.files.upload(file=VIDEO_PATH)

start = time.time()
while True:
    file_obj = client.files.get(name=file_obj.name)
    state = file_obj.state if isinstance(file_obj.state, str) else getattr(file_obj.state, "name", "UNKNOWN")
    if state == "ACTIVE":
        break
    if state == "FAILED":
        raise RuntimeError("Video processing failed.")
    if time.time() - start > TIMEOUT_SECS:
        raise TimeoutError("Timed out waiting for file to become ACTIVE.")
    print("  waiting for file to become ACTIVE...")
    time.sleep(POLL_SECS)

print(f"Upload done. URI: {file_obj.uri} (mime={file_obj.mime_type})")

# --------------- BUILD CONTENTS (URI PART) ---------------
# Use the URI-based Part to avoid strict validation issues.
text_part = types.Part(text=CAPTION_PROMPT)
video_part = types.Part.from_uri(file_uri=file_obj.uri, mime_type=file_obj.mime_type)
contents = types.Content(parts=[text_part, video_part])

# --------------- INFERENCE ---------------
print(f"Generating response with {MODEL_NAME}...")
cfg = types.GenerateContentConfig(
    response_mime_type="application/json",
    temperature=TEMPERATURE
)

resp = client.models.generate_content(
    model=MODEL_NAME,
    contents=contents,
    config=cfg
)

# --------------- SAVE JSON ---------------
out_path = VIDEO_PATH.replace(".mp4", "_caption.json")
try:
    data = json.loads(resp.text)  # ensure valid JSON
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved JSON to: {out_path}")
except json.JSONDecodeError:
    # One-shot repair attempt
    print("Model did not return valid JSON; attempting repair...")
    repair_prompt = CAPTION_PROMPT + "\nIf your previous output was not valid JSON, REPAIR it to match the schema using the same facts."
    contents_repair = types.Content(parts=[types.Part(text=repair_prompt), video_part])

    resp2 = client.models.generate_content(
        model=MODEL_NAME,
        contents=contents_repair,
        config=cfg
    )
    try:
        data2 = json.loads(resp2.text)
        with open(out_path, "w") as f:
            json.dump(data2, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved repaired JSON to: {out_path}")
    except json.JSONDecodeError:
        print("❌ Still not valid JSON. Raw output below:\n")
        print(resp2.text)

# --------------- CLEAN UP REMOTE FILE ---------------
try:
    client.files.delete(name=file_obj.name)
    print("Remote file deleted.")
except Exception as e:
    print(f"Warning: could not delete remote file: {e}")