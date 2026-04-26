import cv2
import json
import os
from vision_ai_extractor import analyze_frames_batch


VIDEO_PATH = "gameplay1.mp4"
FRAME_DIR = "frames"
OUTPUT_PATH = "video_events.json"

SECONDS_BETWEEN_FRAMES = 2


def main():
    os.makedirs(FRAME_DIR, exist_ok=True)

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        raise Exception(f"Could not open video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        raise Exception("Could not read video FPS")

    frame_interval = int(fps * SECONDS_BETWEEN_FRAMES)

    frame_items = []
    frame_count = 0
    saved_count = 0

    print("Extracting frames...")
    print(f"Video: {VIDEO_PATH}")
    print(f"FPS: {fps}")
    print(f"Saving one frame every {SECONDS_BETWEEN_FRAMES} seconds")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_count % frame_interval == 0:
            timestamp = round(frame_count / fps, 2)
            frame_path = f"{FRAME_DIR}/frame_{saved_count}_{timestamp}s.jpg"

            cv2.imwrite(frame_path, frame)

            frame_items.append({
                "timestamp": timestamp,
                "path": frame_path
            })

            print(f"Saved {frame_path}")

            saved_count += 1

        frame_count += 1

    cap.release()

    print(f"\nSending {len(frame_items)} frames in ONE Gemini request...")

    events = analyze_frames_batch(frame_items)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(events, f, indent=2)

    print(f"\nSaved {len(events)} events to {OUTPUT_PATH}")

    for event in events:
        print(event)


if __name__ == "__main__":
    main()