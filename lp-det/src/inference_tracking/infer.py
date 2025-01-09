# src/inference_tracking/infer.py
"""
infer.py
Author: Anthony Tabet (Anthony-Tabet)
Date: 2024-12-06
Description: Runs inference and tracking on a video source.
"""

import argparse
from dotenv import load_dotenv
import cv2
import mlflow
import numpy as np
from loguru import logger
import onnxruntime as ort
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
    # Load YOLO model from mlflow
    logger.info("Loading latest model.")
    latest_model = 'models:/yolov10l/latest'
    model_proto = mlflow.onnx.load_model(latest_model)
    model_path = "temp_model.onnx"
    with open(model_path, "wb") as f:
        f.write(model_proto.SerializeToString())
    session = ort.InferenceSession(model_path)
    logger.info("Model loaded successfully.")

    # use gpu for onnxruntime if available
    if 'CUDAExecutionProvider' in session.get_providers():
        session.set_providers(['CUDAExecutionProvider'])
 
    logger.debug(f"Attempting to open video source: {source}")
    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        logger.error("Failed to open video source")
        errors.inc()
        return
    logger.info("Video source opened successfully.")

    tracker = ObjectTracker(session, forward_url, forward_url_port, out_dir)
    try:
        ret, frame = cap.read()
        input_name = session.get_inputs()[0].name
        while ret:
            with inference_time.time():
                logger.debug("Processing a new frame.")
                input_data = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                input_data = cv2.resize(input_data, (640, 640))
                input_data = input_data.transpose(2, 0, 1)
                input_data = np.expand_dims(input_data, axis=0).astype(np.float32)
                results = session.run(None, {input_name: input_data})
                inference_counter.inc()
                boxes = []
                for result in results[0]:
                    for r in result:
                        print(r)
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
    load_dotenv('.env')
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, help='source')
    parser.add_argument('--output', type=str, default='./output', help='output')
    parser.add_argument('--forward_url', type=str, default='http://127.0.0.1')
    parser.add_argument('--port', type=int, default=8080)

    args = parser.parse_args()
    logger.info(f"Running tracking with source: {args.source}")

    run(args.source, args.output, args.forward_url, args.port)