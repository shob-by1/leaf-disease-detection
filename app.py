from flask import Flask, request, jsonify, send_file, render_template
from ultralytics import YOLO
import os
import uuid

app = Flask(__name__)

# Create required folders
os.makedirs("static/uploads", exist_ok=True)
os.makedirs("static/results", exist_ok=True)

# Load trained model
model = YOLO("models/best.pt")


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "YOLO API running"})


# =========================================================
#                   DETECT (Images + Videos)
# =========================================================
@app.route("/detect", methods=["POST"])
def detect():
    files = request.files.getlist("images")
    results_output = []

    for file in files:
        if file:
            uid = uuid.uuid4().hex[:8]
            filename = file.filename.lower()
            ext = os.path.splitext(filename)[1]

            is_video = ext in [".mp4", ".avi", ".mov", ".mkv"]
            is_image = ext in [".jpg", ".jpeg", ".png"]

            saved_path = os.path.join("static/uploads", uid + ext)
            file.save(saved_path)

            # -----------------------------
            # VIDEO PROCESSING
            # -----------------------------
            if is_video:
                model.predict(
                    source=saved_path,
                    save=True,
                    project="static/results",
                    name=uid,
                    conf=0.25
                )

                annotated_video = f"static/results/{uid}/{uid}.mp4"

                results_output.append({
                    "type": "video",
                    "input": saved_path,
                    "annotated_video": annotated_video
                })

                continue

            # -----------------------------
            # IMAGE PROCESSING
            # -----------------------------
            results = model.predict(
                source=saved_path,
                imgsz=512,
                conf=0.25
            )

            r = results[0]

            detections = []
            total_area = 0
            diseased_area = 0

            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()

                w = xyxy[2] - xyxy[0]
                h = xyxy[3] - xyxy[1]
                area = w * h
                total_area += area

                label = r.names[cls]
                if label == "diseased":
                    diseased_area += area

                detections.append({
                    "class": label,
                    "confidence": round(conf, 3),
                    "bbox": xyxy
                })

            disease_percent = (
                round((diseased_area / total_area) * 100, 2)
                if total_area > 0 else 0
            )

            # Save output
            output_filename = f"{uid}_out.jpg"
            output_path = os.path.join("static/results", output_filename)
            r.plot(save=True, filename=output_path)

            results_output.append({
                "type": "image",
                "input": saved_path,
                "output": output_path,
                "detections": detections,
                "disease_percent": disease_percent
            })

    return jsonify(results_output)


@app.route("/output/<path:filename>")
def output(filename):
    path = os.path.join("static/results", filename)
    if os.path.exists(path):
        return send_file(path)
    return "Not found", 404


@app.route("/ui")
def ui():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
