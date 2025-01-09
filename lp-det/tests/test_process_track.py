# test/test_process_track.py
"""
test_process_track.py
Author: Anthony Tabet (Anthony-Tabet)
Team members: Elias Dargham (edargham), Anthony Tabet (Anthony-Tabet) [#DreamTeam]
Date: 2024-12-06
Description: Tests for the process_track function in the object tracking module.
"""
from src.inference_tracking.object_tracking.utils import process_track
from deep_sort_realtime.deep_sort.track import Track
import numpy as np


def test_process_track() -> None:
    """
    ### Description
        Test the process_track function.
    ### Parameters
        None
    ### Returns
        None
    """

    my_frame = np.random.rand(640, 640 ,3)
    mean = np.random.rand(4).astype(np.float64)
    covariance = np.eye(4).astype(np.float64)
    track_id = 1
    n_init = 3
    max_age = 5
    feature = np.array([0.1, 0.2, 0.3])

    # Instantiate the Track object
    track = Track(mean, covariance, track_id, n_init, max_age, feature)
    frame = my_frame
    out_dir = "./test_output"
    forward_url = "http://127.0.0.1"
    forward_url_port = 8080

    # Run the function
    process_track(track, frame, out_dir, forward_url, forward_url_port)
