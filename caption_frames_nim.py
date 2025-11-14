#!/usr/bin/env python3
"""
Caption Isaac Sim frames with a LLaMA VLM over NVIDIA NIM (OpenAI-compatible) API.

Usage examples:
  # caption every frame, also write SRT
  python caption_frames_nim.py /path/to/rgb --model meta/llama-3.2-11b-vision-instruct --sample-every 1 --save-srt

  # caption one frame per second at 30 FPS
  python caption_frames_nim.py /path/to/rgb --model meta/llama-3.2-11b-vision-instruct --sample-every 30 --fps 30 --save-srt

Environment (set in shell or .env):
  export NIM_API_KEY="nvapi_..."
  export NIM_BASE_URL="https://integrate.api.nvidia.com/v1"
"""

import os, io, base64, json, argparse, time
from pathlib import Path

from PIL import Image
from tqdm import tqdm
from natsort import natsorted

# Load .env if present (optional, safe)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from openai import OpenAI


# ---------- Utilities ----------
def ensure_out_dir(image_dir: str, out_name: str = "captions"):
    image_dir = str(Path(image_dir).resolve())
    parent = str(Path(image_dir).parent)
    out_dir = os.path.join(parent, out_name)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def to_data_url(img_path: str) -> str:
    ext = Path(img_path).suffix.lower().lstrip(".") or "png"
    if ext == "jpg":
        ext = "jpeg"
    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/{ext};base64,{b64}"


def write_json(data, out_path: str):
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_srt(captions_seq, out_path: str, fps: int, span_frames: int):
    def fmt_time(total_seconds: float):
        hh = int(total_seconds // 3600)
        mm = int((total_seconds % 3600) // 60)
        ss = int(total_seconds % 60)
        ms = int(round((total_seconds - int(total_seconds)) * 1000))
        return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"

    with open(out_path, "w", encoding="utf-8") as f:
        for i, (frame_idx, text) in enumerate(captions_seq, start=1):
            start = frame_idx / fps
            end = (frame_idx + max(1, span_frames)) / fps
            f.write(f"{i}\n")
            f.write(f"{fmt_time(start)} --> {fmt_time(end)}\n")
            f.write(text.strip() + "\n\n")


def extract_assistant_text(resp) -> str:
    """
    Robust extraction for NIM/OpenAI responses.
    - message.content may be a string, or a list of dict parts.
    """
    msg = resp.choices[0].message
    content = getattr(msg, "content", None)

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        # NIM often returns [{"type":"output_text","text":"..."}]
        for part in content:
            if isinstance(part, dict):
                if part.get("text"):
                    return part["text"].strip()
                if isinstance(part.get("content"), str):
                    return part["content"].strip()

    return ("" if content is None else str(content)).strip()


def caption_one_frame(
    client: OpenAI,
    model: str,
    img_path: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    retries: int = 3,
    backoff: float = 2.0,
) -> str:
    data_url = to_data_url(img_path)
    last_err = None
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}}
                    ],
                }],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return extract_assistant_text(resp) or "[Empty caption]"
        except Exception as e:
            last_err = e
            time.sleep(backoff * (attempt + 1))
    raise RuntimeError(f"Caption request failed after {retries} retries: {last_err}")


# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser(description="Caption frames with a LLaMA VLM over NVIDIA NIM API.")
    parser.add_argument("image_dir", help="Directory with frames (.png/.jpg).")
    parser.add_argument("--model", default="meta/llama-3.2-11b-vision-instruct",
                        help="Vision model name (e.g., meta/llama-3.2-11b-vision-instruct).")
    parser.add_argument("--prompt", default="Describe this frame succinctly and concretely.",
                        help="Prompt for the VLM.")
    parser.add_argument("--sample-every", type=int, default=1,
                        help="Caption every Nth frame (30 at 30FPS ≈ 1 caption per second).")
    parser.add_argument("--fps", type=int, default=30, help="FPS used for SRT timing.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature.")
    parser.add_argument("--max-tokens", type=int, default=80, help="Max new tokens.")
    parser.add_argument("--out-name", default="captions", help="Output folder name (sibling to image_dir).")
    parser.add_argument("--save-srt", action="store_true", help="Also write captions.srt aligned to frames.")
    parser.add_argument("--limit", type=int, default=None, help="Cap number of captions (debug / quick run).")
    args = parser.parse_args()

    # Env setup
    api_key = os.getenv("NIM_API_KEY")
    base_url = os.getenv("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
    if not api_key:
        raise RuntimeError("Missing NIM_API_KEY. Set it or put it in a .env file.")

    client = OpenAI(api_key=api_key, base_url=base_url)

    # Gather frames
    image_dir = str(Path(args.image_dir).resolve())
    imgs = [f for f in os.listdir(image_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    images = natsorted(imgs)
    if not images:
        raise FileNotFoundError(f"No images in {image_dir}")

    # Outputs
    out_dir = ensure_out_dir(image_dir, args.out_name)
    out_json = os.path.join(out_dir, "captions.json")
    out_srt = os.path.join(out_dir, "captions.srt")

    captions_dict = {}
    captions_seq = []

    count = 0
    for idx, name in enumerate(tqdm(images, desc="Captioning")):
        if idx % args.sample_every != 0:
            continue
        if args.limit is not None and count >= args.limit:
            break

        img_path = os.path.join(image_dir, name)
        text = caption_one_frame(
            client=client,
            model=args.model,
            img_path=img_path,
            prompt=args.prompt,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )

        captions_dict[name] = {"frame_index": idx, "caption": text}
        captions_seq.append((idx, text))
        count += 1

    write_json(captions_dict, out_json)
    print(f"✅ Captions JSON: {out_json}")

    if args.save_srt:
        write_srt(captions_seq, out_srt, fps=args.fps, span_frames=args.sample_every)
        print(f"✅ Captions SRT : {out_srt}")


if __name__ == "__main__":
    main()