# src/inference_tracking/object_tracking/tracker.py
"""
tracker.py
Author: Anthony Tabet (Anthony-Tabet)
Team members: Elias Dargham (edargham), Anthony Tabet (Anthony-Tabet) [#DreamTeam]
Date: 2024-12-01
Description: Object tracking class using the DeepSort algorithm with custom extensions.
"""
import cv2
import numpy as np
from loguru import logger

from deep_sort_realtime.deepsort_tracker import DeepSort
from deep_sort_realtime.deep_sort.track import Track
from inference_tracking.realtime_tracking.overrides import TrackExtended

logger.add("object_tracking.log", rotation="10 MB")   # Each log file is limited to 10 MB

logger.add("object_tracking.log", rotation="10 MB")   # Each log file is limited to 10 MB

class ObjectTracker:
    """
    ### Description
        A class to handle object tracking using the DeepSort algorithm with custom extensions.
    ### Attributes
        model (Model): The model used for detection or feature extraction.
        forward_url (str): The URL to forward tracking data.
        forward_url_port (int): The port number at the forward URL.
        out_dir (str): Directory where the output is stored.
        tracker (DeepSort): The DeepSort tracker instance with customized tracking class.
    """

    def __init__(self, model, forward_url: str, forward_url_port: int, out_dir: str) -> None:
        """
        ### Description
            Initialize the ObjectTracker class with model, forwarding URL, port,
            and output directory.
        ### Parameters
            model (Model): The model used for detection or feature extraction.
            forward_url (str): The URL to forward tracking data.
            forward_url_port (int): The port number at the forward URL.
            out_dir (str): Directory where the output is stored.
        """
        self.model = model
        self.forward_url = forward_url
        self.forward_url_port = forward_url_port
        self.out_dir = out_dir
        self.tracker = DeepSort(max_age=5, override_track_class=TrackExtended)
        logger.info("ObjectTracker initialized with DeepSort and TrackExtended.")

    def update_tracks(self, boxes: list, frame: np.ndarray = None) -> list[Track]:
        """
        ### Description
            Update tracks with the current frame and detection boxes.
        ### Parameters
            boxes (list): List of detection boxes.
            frame (numpy.ndarray): The current frame.
        ### Returns
            list[Track]: Updated tracks for the current frame.
        """
        updated_tracks = self.tracker.update_tracks(boxes, frame=frame)
        logger.debug(f"Tracks updated for frame: {updated_tracks}")
        return updated_tracks

    def draw_tracking(self, track: Track, frame: np.ndarray) -> None:
        """
        ### Description
            Draw the tracking information on the frame.
        ### Parameters
            track (Track): The track information to draw.
            frame (np.ndarray): The frame on which to draw.
        ### Returns
            None
        """
        track_id = int(track.track_id)
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = ltrb
        cv2.rectangle(
            frame,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (0, 255, 0),
            3
        )
        cv2.putText(
            frame,
            f"id: {track_id}", (int(x1), int(y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2
        )
        logger.debug(f"Track drawn on frame: ID {track_id} at [{x1}, {y1}, {x2}, {y2}]")
