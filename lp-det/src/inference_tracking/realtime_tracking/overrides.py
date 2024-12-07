# src/lp-det/object_tracker/deep_sort_realtime/overrides.py
"""
overrides.py
Author: Anthony Tabet (Anthony-Tabet)
Date: 2024-12-06
Description: Overrides the Track class to include additional attributes or methods.
"""
from loguru import logger
from deep_sort_realtime.deep_sort.track import Track


logger.add("track_extension.log", rotation="10 MB")  # Log file setup

class TrackExtended(Track):
    """
    Extends the Track class to include additional attributes or methods.
    """

    def __init__(
        self,
        *args,
        class_id: str = None,
        **kwargs,
    ):
        """
        ### Description
            Constructor for the TrackExtended class, enhancing Track with additional properties.
        ### Parameters
            class_id (str): Optional class ID for the track, used for additional classification.
            *args: Variable length argument list for base class initialization.
            **kwargs: Arbitrary keyword arguments for base class initialization.
        """
        super().__init__(*args, **kwargs)
        self.class_id = class_id  # Optional: add a class ID if necessary
        self.speed = None  # Optional: track speed if needed
        logger.debug(f"TrackExtended initialized with class_id: {class_id} and speed: {self.speed}")
