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

from inference_tracking.object_tracking.tracker import ObjectTracker
from inference_tracking.object_tracking.utils import process_track

def run(
    source: str,
    out_dir: str,
    forward_url: str,
    forward_url_port: int
):
    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print("Failed to open video source")
        return

    model = YOLO("yolov10x.pt")
    model.classes = [0, 1, 2, 3, 4, 5, 6, 7, 15, 16, 17]
    tracker = ObjectTracker(model, forward_url, forward_url_port, out_dir)

    ret, frame = cap.read()
    while ret:
        results = model(frame)
        boxes = []
        for result in results:
            for r in result.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = r
                if score > 0.5:
                    boxes.append([[x1, y1, x2 - x1, y2 - y1], score, class_id])

        tracks = tracker.update_tracks(boxes, frame)
        for track in tracks:
            if track.is_confirmed():
                tracker.draw_tracking(track, frame)

        cv2.imshow('Footage', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        ret, frame = cap.read()

    cap.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='./data/Test.mp4', help='source')
    parser.add_argument('--output', type=str, default='./output', help='output')
    parser.add_argument('--forward_url', type=str, default='http://127.0.0.1')
    parser.add_argument('--port', type=int, default=8080)

    args = parser.parse_args()

    run(args.source, args.output, args.forward_url, args.port)
