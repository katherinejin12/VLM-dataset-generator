import os
import cv2
import argparse
from natsort import natsorted


def images_to_video(image_dir: str, video_name="output.mp4", fps=30, split=False, chunk_size=60):
    """Convert images to video. Optionally split every `chunk_size` frames (e.g., 2 s @ 30 fps)."""
    image_dir = os.path.abspath(image_dir.rstrip("/"))
    temp_dir = os.path.dirname(image_dir)
    parent_dir = os.path.join(temp_dir, "videos")
    os.makedirs(parent_dir, exist_ok=True)
    output_path = os.path.join(parent_dir, video_name)

    images = [f for f in os.listdir(image_dir)
              if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    images = natsorted(images)
    if not images:
        raise ValueError(f"No image files in {image_dir}")

    first_frame = cv2.imread(os.path.join(image_dir, images[0]))
    h, w, _ = first_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    def write_chunk(start_idx, end_idx, part_num=None):
        """Write a chunk of frames into one video file."""
        part_name = (f"{os.path.splitext(video_name)[0]}_part{part_num}.mp4"
                     if split else video_name)
        part_path = os.path.join(parent_dir, part_name)
        out = cv2.VideoWriter(part_path, fourcc, fps, (w, h))

        for img_name in images[start_idx:end_idx]:
            frame = cv2.imread(os.path.join(image_dir, img_name))
            if frame is not None:
                out.write(frame)
        out.release()
        print(f"✅ Saved: {part_path}")

    if split:
        for i in range(0, len(images), chunk_size):
            end = min(i + chunk_size, len(images))
            part_num = i // chunk_size + 1
            write_chunk(i, end, part_num)
    else:
        write_chunk(0, len(images))

    print("🎬 Done!")


def main():
    parser = argparse.ArgumentParser(description="Convert images to video.")
    parser.add_argument("image_dir", help="Path to image directory")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--name", type=str, default="output.mp4", help="Video file name")
    parser.add_argument("--split", action="store_true",
                        help="Split into segments (e.g., 60 frames = 2 seconds @ 30 fps)")
    parser.add_argument("--chunk", type=int, default=60, help="Frames per segment when --split is used")
    args = parser.parse_args()

    images_to_video(args.image_dir, args.name, args.fps, args.split, args.chunk)


if __name__ == "__main__":
    main()