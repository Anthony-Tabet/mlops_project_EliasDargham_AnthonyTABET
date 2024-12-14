# src/inference_tracking/monitoring.py
"""
monitoring.py
Author: Anthony Tabet (Anthony-Tabet)
Date: 2024-12-09
Description: Prometheus monitoring setup for the inference tracking system.
"""
from prometheus_client import Counter, Histogram, Gauge, start_http_server


# A counter to track the number of inferences
inference_counter = Counter('inferences', 'Number of inference operations')

inference_time = Histogram('inference_time', 'Time spent on inference per frame')

# A histogram to track the processing time of each frame
processed_frames = Counter('processed_frames', 'Total number of processed video frames')

# A gauge to monitor the number of active tracks
active_tracks = Gauge('active_tracks', 'Number of active tracks at any time')

errors = Counter('errors', 'Number of errors encountered')

# Start up a server to expose these metrics to Prometheus
start_http_server(8001)  # Make sure this port is free or set to another free port
