# from src.video_processing import extract_frames
# extract_frames("sample_video.mp4", "data/frames")
from src.video_processing import extract_frames
from src.face_extraction import extract_faces

# Step 1: Extract frames
extract_frames("sample_video.mp4", "data/frames")

# Step 2: Extract faces from frames
extract_faces("data/frames", "data/faces")