# deep_sort_realtime/track.py

class Track:
    def __init__(self, track_id, bbox, confidence, feature):
        self.track_id = track_id  # Unique identifier for the track
        self.bbox = bbox  # Bounding box [x, y, w, h]
        self.confidence = confidence  # Detection confidence
        self.feature = feature  # Appearance feature for re-identification

    def to_tlwh(self):
        """Convert bounding box to top-left width-height format."""
        return self.bbox[0], self.bbox[1], self.bbox[2], self.bbox[3]
    
    def to_tlbr(self):
        """Convert bounding box to top-left bottom-right format."""
        x1, y1, w, h = self.bbox
        return x1, y1, x1 + w, y1 + h

    def update(self, bbox, confidence, feature):
        """Update the track with a new bounding box and feature."""
        self.bbox = bbox
        self.confidence = confidence
        self.feature = feature

    def get_track_info(self):
        """Return information about the track."""
        return {
            "track_id": self.track_id,
            "bbox": self.bbox,
            "confidence": self.confidence
        }
