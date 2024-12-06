# deep_sort_realtime/overrides.py

from deep_sort_realtime.deep_sort_tracker import Track

class TrackExtended(Track):
    def __init__(self, track_id, bbox, confidence, feature, class_id=None):
        super().__init__(track_id, bbox, confidence, feature)
        self.class_id = class_id  # Optional: add a class ID if necessary
        self.speed = None  # Optional: track speed if needed
        
    def update_speed(self, new_bbox):
        # Example of calculating speed based on bounding box movement (optional)
        pass
