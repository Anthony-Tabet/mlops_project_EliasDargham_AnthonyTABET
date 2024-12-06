# src/lp-det/object_tracker/deep_sort_realtime/overrides.py
"""
overrides.py
Author: Anthony Tabet (Anthony-Tabet)
Date: 2024-12-06
Description: Overrides the Track class to include additional attributes or methods.
"""

from deep_sort_realtime.deep_sort.track import Track


class TrackExtended(Track):
    """
    Extends the Track class to include additional attributes or methods.
    """
    def __init__(
        self,
        *args,
        class_id: str=None,
        **kwargs,
    ):
        """
        ### Description
            Constructor for the TrackExtended class.
        ### Parameters
            class_id (str): Optional class ID for the track.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.class_id = class_id  # Optional: add a class ID if necessary
        self.speed = None  # Optional: track speed if needed
