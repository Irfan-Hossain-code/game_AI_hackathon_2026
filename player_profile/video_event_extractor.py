import cv2
import json
import os
from vision_ai_extractor import analyze_frame


VIDEO_PATH = "gameplay1.mp4"
FRAME_DIR = "frames"
OUTPUT_PATH = "video_events.json"


def main():
    os.makedirs(FRAME_DIR, exist_ok=True)

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        raise Exception(f"Could not open video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * 5)  # analyze every 5 seconds

    events = []
    frame_count = 0
    analyzed_count = 0

    print("Processing video with vision AI...")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_count % frame_interval == 0:
            timestamp = round(frame_count / fps, 2)
            frame_path = f"{FRAME_DIR}/frame_{analyzed_count}_{timestamp}s.jpg"

            cv2.imwrite(frame_path, frame)

            print(f"Analyzing {frame_path} at {timestamp}s")

            try:
                event = analyze_frame(frame_path, timestamp)

                if event is not None:
                    events.append(event)
                    print("Detected:", event)
                else:
                    print("No event detected.")

            except Exception as e:
                print("AI analysis failed:", e)

            analyzed_count += 1

        frame_count += 1

    cap.release()

    with open(OUTPUT_PATH, "w") as f:
        json.dump(events, f, indent=2)

    print(f"\nSaved {len(events)} events to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()