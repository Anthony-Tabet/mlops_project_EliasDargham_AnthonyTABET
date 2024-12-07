from inference_tracking.object_tracking.utils import process_track

def test_process_track():
    # Mock inputs
    track = MockTrack()
    frame = MockFrame()
    out_dir = "./test_output"
    forward_url = "http://127.0.0.1"
    forward_url_port = 8080

    # Run the function
    result = process_track(track, frame, out_dir, forward_url, forward_url_port)

    # Assert expected behavior
    assert result is None  # Replace with actual assertions
