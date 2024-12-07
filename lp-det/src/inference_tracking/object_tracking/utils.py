# src/swe/object_tracking/utils.py
"""
utils.py
Author: Anthony Tabet (Anthony-Tabet)
Date:
Description:
"""

import base64
import os
import uuid
import textwrap
import argparse
import requests
import cv2
import numpy as np
from ultralytics import YOLO
from loguru import logger
from deep_sort_realtime.deep_sort.track import Track
from deep_sort_realtime.deepsort_tracker import DeepSort


logger.add("tracking_process.log", rotation="10 MB")  # Log file setup

def process_track(track: Track, frame: cv2.Mat, out_dir: str, forward_url: str, forward_url_port: int, forward_url_path: str = "/") -> None:
    """
    Process each track by cropping from the frame, encoding it, sending it to a server, and saving the response.

    Args:
        track (Track): The track information containing the bounding box.
        frame (cv2.Mat): The frame from which to crop the tracked object.
        out_dir (str): Directory where the outputs are stored.
        forward_url (str): The URL to forward the cropped image.
        forward_url_port (int): The port number for the URL.
        forward_url_path (str): The path at the URL to send the post request.

    Returns:
        None
    """
    x, y, w, h = track.to_tlwh()
    if frame is not None and frame.shape[0] != 0 and frame.shape[1] != 0:
        crop = frame[int(y):int(y + h), int(x):int(x + w)]
        if crop.shape[0] == 0 or crop.shape[1] == 0:
            logger.warning(f"Empty bounding box for track {track.track_id}. Skipping...")
            return

        _, encoded_image = cv2.imencode('.jpg', crop)
        crop_base64 = base64.b64encode(encoded_image).decode()

        try:
            endpoint_url = f'{forward_url}:{forward_url_port}{forward_url_path}'
            payload = {'image': crop_base64}
            response = requests.post(endpoint_url, json=payload)
            logger.debug(f"Request sent to {endpoint_url}. Status code: {response.status_code}")

            if response.status_code == 200:
                caption = response.content.decode()
                wrapped_text = textwrap.wrap(caption, width=35)
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)

                uuid_str = str(uuid.uuid4())
                os.makedirs(os.path.join(out_dir, uuid_str), exist_ok=True)

                crop_path = os.path.join(out_dir, uuid_str, 'image.jpg')
                cv2.imwrite(crop_path, crop)

                y = 0
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1
                thickness = 2
                color = (255, 0, 0)
                (_, line_height), _ = cv2.getTextSize('A', font, font_scale, thickness)

                for line in wrapped_text:
                    y += line_height + 10

                crop = cv2.putText(crop, caption, (10, y), font, font_scale, color, thickness, cv2.LINE_AA)
                cv2.imshow('Image', crop)

                caption_path = os.path.join(out_dir, uuid_str, 'caption.txt')
                with open(caption_path, 'w', encoding='utf8') as f:
                    f.write(caption)
                logger.info(f"Caption saved to {caption_path}")

            else:
                logger.error(f"Received non-200 response: {response.status_code}")

        except Exception as e:
            logger.error(f"Failed to send image: {e}")

    else:
        logger.warning(f"Invalid frame or empty bounding box for track {track.track_id}")

def run_object_tracking(input_video: str, output_dir: str, forward_url: str, forward_url_port: int) -> None:
    """
    Run the object tracking pipeline using YOLOv5 and DeepSort on a video input.

    Args:
        input_video (str): Path to the input video file.
        output_dir (str): Directory to save cropped images and captions.
        forward_url (str): URL to send cropped object images for further processing.
        forward_url_port (int): Port for the forward URL.
    """
    logger.info("Initializing YOLOv5 model.")
    # Initialize the YOLOv5 model
    model = YOLO("yolov5s.pt")  # Make sure the model path is correct for your use case

    logger.info("Initializing DeepSort tracker.")
    # Initialize the DeepSort tracker
    deepsort = DeepSort()

    # Open the input video stream
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        logger.error("Could not open video stream.")
        return

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            logger.info("No more frames to read, exiting loop.")
            break

        frame_id += 1
        logger.debug(f"Processing frame {frame_id}.")

        # Perform object detection using YOLOv5
        results = model(frame)

        # Extract detected boxes and confidences
        boxes = results.xywh[0][:, :-1].cpu().numpy()
        confidences = results.xywh[0][:, -1].cpu().numpy()

        # Convert detections to the format required by DeepSORT
        detections = []
        for box, confidence in zip(boxes, confidences):
            if confidence > 0.5:  # Set a confidence threshold
                detections.append(np.array([box[0], box[1], box[2], box[3], confidence]))

        # Update the DeepSort tracker
        tracks = deepsort.update_tracks(detections)

        # Process each track
        for track in tracks:
            process_track(
                track=track,
                frame=frame,
                out_dir=output_dir,
                forward_url=forward_url,
                forward_url_port=forward_url_port
            )

        # Optional: Display the frame with tracked objects
        for track in tracks:
            x1, y1, x2, y2 = track.to_tlbr()  # Get bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"ID: {track.track_id}",
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

        # Show the frame with tracking annotations
        cv2.imshow('Object Tracking', frame)

        # Press 'q' to quit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            logger.info("Exiting tracking loop.")
            break

    cap.release()
    cv2.destroyAllWindows()
    logger.info("Object tracking finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run object tracking pipeline using YOLOv5 and DeepSORT.")
    parser.add_argument('--input_video', type=str, required=True, help="Path to the input video file.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save cropped images and captions.")
    parser.add_argument('--forward_url', type=str, required=True,
                        help="URL to send cropped object images for further processing.")
    parser.add_argument('--forward_url_port', type=int, required=True, help="Port for the forward URL.")

    args = parser.parse_args()

    run_object_tracking(
        input_video=args.input_video,
        output_dir=args.output_dir,
        forward_url=args.forward_url,
        forward_url_port=args.forward_url_port
    )