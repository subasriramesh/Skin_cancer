# =============================================================
# Flask Backend — Melanoma Skin Cancer Detection API
# =============================================================
import os
import io
import base64
import numpy as np
import cv2
import tensorflow as tf

from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image

# ----- Config -----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "efficientnet_b4_model.h5")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
IMG_SIZE = 380

os.makedirs(UPLOAD_DIR, exist_ok=True)

# ----- App -----
app = Flask(__name__)
CORS(app)

# ----- Load Model -----
model = None

def get_model():
    """Lazy-load model so server starts even if model file is missing."""
    global model
    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. "
                "Please run train_local.py first to generate the model."
            )
        print(f"Loading model from {MODEL_PATH} ...")
        model = load_model(MODEL_PATH)
        print("Model loaded successfully!")
    return model

# ----- Helpers -----
def preprocess_image(image_bytes):
    """Read image bytes, resize, and preprocess for EfficientNet."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img, dtype=np.float32)
    img_array = preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)

def generate_gradcam(img_array, mdl):
    """Generate Grad-CAM heatmap and return as base64 JPEG."""
    try:
        last_conv = mdl.get_layer("top_conv")
        grad_model = tf.keras.Model(mdl.inputs, [last_conv.output, mdl.output])

        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(img_array)
            loss = preds[:, 0]

        grads = tape.gradient(loss, conv_out)
        weights = tf.reduce_mean(grads, axis=(0, 1, 2))
        cam = tf.reduce_sum(weights * conv_out[0], axis=-1)
        cam = np.maximum(cam.numpy(), 0)
        cam = cam / (np.max(cam) + 1e-8)

        heatmap = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
        heatmap = np.uint8(255 * heatmap)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # Original image (undo preprocess for display)
        orig = img_array[0].copy()
        orig = ((orig - orig.min()) / (orig.max() - orig.min() + 1e-8) * 255).astype(np.uint8)
        orig = cv2.cvtColor(orig, cv2.COLOR_RGB2BGR)

        overlay = cv2.addWeighted(orig, 0.6, heatmap_color, 0.4, 0)
        _, buffer = cv2.imencode(".jpg", overlay)
        return base64.b64encode(buffer).decode("utf-8")
    except Exception as e:
        print(f"Grad-CAM error: {e}")
        return None

# ----- Routes -----
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})

@app.route("/api/predict", methods=["POST"])
def predict():
    # Validate file
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    allowed = {"jpg", "jpeg", "png", "bmp", "webp"}
    ext = file.filename.rsplit(".", 1)[-1].lower()
    if ext not in allowed:
        return jsonify({"error": f"Invalid file type .{ext}"}), 400

    try:
        # Read & preprocess
        image_bytes = file.read()
        img_array = preprocess_image(image_bytes)

        # Predict
        mdl = get_model()
        prediction = mdl.predict(img_array, verbose=0)
        confidence = float(prediction[0][0])

        # Classify
        threshold = 0.5
        is_malignant = confidence > threshold
        class_name = "Malignant" if is_malignant else "Benign"
        display_confidence = confidence if is_malignant else (1 - confidence)

        # Grad-CAM
        gradcam_b64 = generate_gradcam(img_array, mdl)

        # Save upload (optional history)
        save_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(save_path, "wb") as f:
            f.write(image_bytes)

        return jsonify({
            "prediction": class_name,
            "confidence": round(display_confidence * 100, 2),
            "raw_probability": round(confidence, 4),
            "threshold": threshold,
            "gradcam": gradcam_b64,
            "message": (
                "⚠️ High risk of Malignant Melanoma detected. Please consult a dermatologist immediately."
                if is_malignant
                else "✅ The lesion appears Benign. Regular monitoring is still recommended."
            )
        })

    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

# ----- Main -----
if __name__ == "__main__":
    print("=" * 50)
    print("  Melanoma Detection API Server")
    print("  http://localhost:5000")
    print("=" * 50)
    app.run(host="0.0.0.0", port=5000, debug=True)
