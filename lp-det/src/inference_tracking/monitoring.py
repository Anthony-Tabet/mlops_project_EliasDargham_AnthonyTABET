from prometheus_client import Counter, Histogram, Gauge, start_http_server

# A counter to track the number of inferences
inference_counter = Counter('inferences', 'Number of inference operations')

# A histogram to track the processing time of each frame
processing_time = Histogram('processing_time', 'Time spent processing each frame')

# A gauge to monitor the number of active tracks
active_tracks = Gauge('active_tracks', 'Number of active tracks at any time')

errors = Counter('errors', 'Number of errors encountered')

# Start up a server to expose these metrics to Prometheus
start_http_server(8001)  # Make sure this port is free or set to another free port
