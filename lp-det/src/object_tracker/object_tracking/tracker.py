# src/swe/object_tracking/tracker.py

import random
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
from deep_sort_realtime.overrides import TrackExtended
from deep_sort_realtime.deep_sort.track import Track
from src.object_tracker.object_tracking.utils import process_track

class ObjectTracker:
    def __init__(self, model, forward_url, forward_url_port, out_dir):
        self.model = model
        self.forward_url = forward_url
        self.forward_url_port = forward_url_port
        self.out_dir = out_dir
        self.tracker = DeepSort(max_age=5, override_track_class=TrackExtended, on_delete=process_track)

    def update_tracks(self, boxes, frame):
        return self.tracker.update_tracks(boxes, frame=frame)

    def draw_tracking(self, track, frame):
        track_id = int(track.track_id)
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = ltrb
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
        cv2.putText(frame, f"id: {track_id}", (int(x1), int(y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
