"""
player_profile/cv_radar_engine.py

The blazing-fast Mathematical "Eyes" of the Clash Royale AI Coach.
Uses YOLOv8n to track generic movement objects, translates them to Clash Royale 
board locations, and yields standard JSON state payloads directly into the Mistral pipeline.
"""

import cv2
from ultralytics import YOLO
from player_profile.zone_translator import get_zone_from_coords

class CVRadarEngine:
    def __init__(self, video_source="player_profile/gameplay1.mp4"):
        self.video_source = video_source
        self.cap = cv2.VideoCapture(video_source)
        
        print(f"[Radar] Instantiating YOLOv8 Engine on {video_source} ...")
        self.model = YOLO("yolov8n.pt")
        
        # State tracking (mocked hand/elixir since we aren't tracking UI here)
        self.current_elixir = 5
        self.current_hand = ["Hog Rider", "Log", "Ice Spirit", "Musketeer"]
        
        self.COCO_TO_CLASH = {
            "person": "Giant",
            "dog": "Hog Rider",
            "sports ball": "Fireball",
            "bird": "Baby Dragon",
            "truck": "Ice Golem",
            "tv": "UI Element",
            "train": "Battle Ram",
            "bus": "Pekka"
        }

    def run_radar_loop(self):
        """Simulates 60 FPS scanning, yielding discrete game state JSONs."""
        if not self.cap.isOpened():
            print(f"[Radar Error] Failed to open visual resource: {self.video_source}")
            return
            
        screen_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        screen_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        
        frame_count = 0
        
        # Prevent spamming the Mistral pipeline 60 times a second for the same event
        events_fired_this_second = [] 
        
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
                
            timestamp_sec = frame_count / fps
            timestamp_str = f"{int(timestamp_sec // 60)}:{int(timestamp_sec % 60):02d}"
            
            # Clear logic gate every real-life second
            if int(timestamp_sec) > (frame_count - 1) // fps:
                events_fired_this_second.clear()

            # YOLO inference
            results = self.model(frame, verbose=False)
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Generic -> Clash mapping
                    cls_id = int(box.cls[0].item())
                    coco_name = self.model.names[cls_id]
                    cr_troop = self.COCO_TO_CLASH.get(coco_name, "Unknown Troop")
                    
                    if cr_troop == "UI Element" or cr_troop == "Unknown Troop":
                        continue
                        
                    # Spam Blocker (Don't fire 30 times for a Pekka walking 1 inch)
                    if cr_troop in events_fired_this_second:
                        continue
                    events_fired_this_second.append(cr_troop)
                    
                    # Grid -> Math
                    coords = box.xyxy[0].tolist()
                    center_x = (coords[0] + coords[2]) / 2
                    center_y = (coords[1] + coords[3]) / 2
                    
                    # Math -> English
                    english_location = get_zone_from_coords(center_x, center_y, screen_width, screen_height)
                    
                    # Yield structured JSON array
                    state_json = {
                      "timestamp": timestamp_str,
                      "opponent_action": f"Placed {cr_troop}",
                      "location": english_location,
                      "user_hand": self.current_hand,
                      "user_elixir": self.current_elixir
                    }
                    
                    yield state_json
            
            frame_count += 1
            
        self.cap.release()

if __name__ == "__main__":
    # Test execution
    radar = CVRadarEngine()
    import json
    for event in radar.run_radar_loop():
        print(json.dumps(event, indent=2))
