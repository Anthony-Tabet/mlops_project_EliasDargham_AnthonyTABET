# src/swe/object_tracking/tracker.py
""""""
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
from deep_sort_realtime.deep_sort.track import Track
from inference_tracking.realtime_tracking.overrides import TrackExtended
from inference_tracking.object_tracking.utils import process_track


class ObjectTracker:
    def __init__(self, model, forward_url: str, forward_url_port: int, out_dir: str) -> None:
        self.model = model
        self.forward_url = forward_url
        self.forward_url_port = forward_url_port
        self.out_dir = out_dir
        self.tracker = DeepSort(max_age=5, override_track_class=TrackExtended)

    def update_tracks(self, boxes: list, frame: cv2.Mat):
        return self.tracker.update_tracks(boxes, frame=frame)

    def draw_tracking(self, track: Track, frame: cv2.Mat) -> None:
        track_id = int(track.track_id)
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = ltrb
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
        cv2.putText(frame, f"id: {track_id}", (int(x1), int(y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
