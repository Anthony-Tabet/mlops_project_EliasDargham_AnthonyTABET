# src/swe/object_tracking/main.py

import argparse
import os
import random
import base64
import uuid
import cv2
import requests
from deep_sort_realtime.deep_sort.track import Track
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO
from src.swe.object_tracking.tracker import ObjectTracker
from src.swe.object_tracking.utils import process_track

def main(
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='./data/Test.mp4', help='source')
    parser.add_argument('--output', type=str, default='./output', help='output')
    parser.add_argument('--forward_url', type=str, default='http://127.0.0.1')
    parser.add_argument('--port', type=int, default=8080)

    args = parser.parse_args()

    main(args.source, args.output, args.forward_url, args.port)
