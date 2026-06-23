from flask import Flask, render_template, request, jsonify
import os

# IMPORT BOTH
from src.inference import predict_image
from video_predict import predict_video

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------- HOME ----------------

@app.route("/")
def landing():
    return render_template("home.html")


@app.route("/detection")
def detection():
    return render_template("index.html")

@app.route("/login")
def login():
    return render_template("login.html")
# ---------------- PREDICT ----------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"})

        file = request.files["file"]

        if file.filename == "":
            return jsonify({"error": "Empty file"})

        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        filename = file.filename.lower()

        # ---------------- IMAGE ----------------
        if filename.endswith((".jpg", ".jpeg", ".png")):
            label, confidence, fake, real = predict_image(filepath)

            return jsonify({
                "type": "image",
                "result": label,
                "confidence": float(confidence),   # already 0–1
                "real_conf": float(real),
                "fake_conf": float(fake)
            })

        # ---------------- VIDEO ----------------
        elif filename.endswith((".mp4", ".avi", ".mov")):
            output = predict_video(filepath)

            return jsonify({
                "type": "video",
                "result": output["label"],
                "confidence": float(output["confidence"]),  # ⚠️ must be 0–1
                "real_conf": float(output["real_prob"]),
                "fake_conf": float(output["fake_prob"])
            })

        else:
            return jsonify({"error": "Unsupported file format"})

    except Exception as e:
        print("ERROR:", e)  # 🔥 shows error in terminal
        return jsonify({"error": str(e)})

# ---------------- FRAME EXTRACTION ----------------
@app.route("/extract_frames", methods=["POST"])
def extract_frames_api():
    try:
        from src.video_processing import extract_frames

        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"})

        file = request.files["file"]

        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        output_folder = "static/frames"
        os.makedirs(output_folder, exist_ok=True)

        # clear old frames
        for f in os.listdir(output_folder):
            os.remove(os.path.join(output_folder, f))

        # extract frames
        extract_frames(filepath, output_folder, frame_rate=10)

        frame_files = os.listdir(output_folder)

        frame_urls = [
            f"/static/frames/{f}" for f in frame_files
        ]

        return jsonify({"frames": frame_urls})

    except Exception as e:
        print("FRAME ERROR:", e)
        return jsonify({"error": str(e)})

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
    

# “My system supports both image and video deepfake detection.
# For images, I use a ResNet18-based CNN.
# For videos, I extract frames, detect faces, and average predictions.
# I also extract audio and convert it into spectrograms for classification.

# Finally, I combine video and audio predictions using weighted fusion.”
#Earlier the model was overfitting and giving extreme confidence. After 
# improving the pipeline and fusion, predictions became more calibrated and
# reliable.”

#"Initially, image model output probabilities in [0,1] while video model 
# returned percentage values [0,100]. This mismatch caused incorrect UI scaling. I standardized
# outputs to probability scale (0–1) across all modalities before rendering."