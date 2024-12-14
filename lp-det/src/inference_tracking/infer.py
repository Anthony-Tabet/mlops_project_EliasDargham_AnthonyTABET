# src/inference_tracking/infer.py
"""
infer.py
Author: Anthony Tabet (Anthony-Tabet)
Date: 2024-12-06
Description: Runs inference and tracking on a video source.
"""

import argparse
import cv2

from ultralytics import YOLO
from loguru import logger

from inference_tracking.object_tracking.tracker import ObjectTracker
from .monitoring import inference_counter, inference_time, processed_frames, active_tracks, errors

logger.add("tracking.log", rotation="10 MB")  # Log file setup

def run(
    source: str,
    out_dir: str,
    forward_url: str,
    forward_url_port: int
) -> None:
    """
    ### Description
        Execute the object tracking process on a given video source.
    ### Parameters
        source (str): Path or URL to the video source.
        out_dir (str): Directory to save any outputs.
        forward_url (str): URL to send detected object data for processing.
        forward_url_port (int): Port number of the forward URL.
    ### Returns
        None
    """
    logger.debug(f"Attempting to open video source: {source}")
    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        logger.error("Failed to open video source")
        errors.inc()
        return

    logger.info("Initializing YOLO model.")
    model = YOLO("yolov10x.pt")
    model.classes = [0, 1, 2, 3, 4, 5, 6, 7, 15, 16, 17]
    tracker = ObjectTracker(model, forward_url, forward_url_port, out_dir)
    try:
        ret, frame = cap.read()
        while ret:
            with inference_time.time():
                logger.debug("Processing a new frame.")
                results = model(frame)
                inference_counter.inc()
                boxes = []
                for result in results:
                    for r in result.boxes.data.tolist():
                        x1, y1, x2, y2, score, class_id = r
                        if score > 0.5:
                            boxes.append([[x1, y1, x2 - x1, y2 - y1], score, class_id])

                tracks = tracker.update_tracks(boxes, frame)
                active_tracks.set(len(tracks))
                for track in tracks:
                    if track.is_confirmed():
                        tracker.draw_tracking(track, frame)

                cv2.imshow('Footage', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("Quitting the tracking loop.")
                    break
            processed_frames.inc()
            ret, frame = cap.read()
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        errors.inc()  # Increment error counter on exceptions

    cap.release()
    cv2.destroyAllWindows()
    logger.info("Tracking process completed.")

def main() -> None:
    """
    ### Description
        Parse arguments and run the object tracking process.
    ### Parameters
        None
    ### Returns
        None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='./data/Test.mp4', help='source')
    parser.add_argument('--output', type=str, default='./output', help='output')
    parser.add_argument('--forward_url', type=str, default='http://127.0.0.1')
    parser.add_argument('--port', type=int, default=8080)

    args = parser.parse_args()
    logger.info(f"Running tracking with source: {args.source}")

    run(args.source, args.output, args.forward_url, args.port)
