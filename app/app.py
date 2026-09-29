import os
import sys
import time
import io
import zipfile
import h5py
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageChops
from scipy.ndimage import label
import streamlit as st
import tensorflow as tf

# Set environment variable for protobuf compatibility

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "handwritten_digit_cnn.keras"

st.set_page_config(
    page_title="Handwritten Digit Recognition",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
    /* Force overall app light background */
    html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"], [data-testid="stToolbar"] {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
    }

    /* Sidebar Light Theme */
    [data-testid="stSidebar"], [data-testid="stSidebar"] > div {
        background-color: #F8FAFC !important;
        color: #0F172A !important;
        border-right: 1px solid #E2E8F0 !important;
    }

    /* All Headings, Labels, and Text */
    h1, h2, h3, h4, h5, h6, 
    .stMarkdown, .stMarkdown p, .stMarkdown span,
    label, [data-testid="stWidgetLabel"] p, [data-testid="stSidebar"] * {
        color: #0F172A !important;
    }

    /* Professional Header */
    .app-header {
        border-bottom: 2px solid #E2E8F0;
        padding-bottom: 1rem;
        margin-bottom: 1.5rem;
    }

    .app-title {
        font-size: 2.25rem;
        font-weight: 700;
        color: #0F172A !important;
        margin: 0;
    }

    .app-subtitle {
        font-size: 1.05rem;
        color: #475569 !important;
        margin-top: 0.25rem;
    }

    /* Result Box */
    .result-card {
        background-color: #F8FAFC !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 8px;
        padding: 1.5rem;
        text-align: center;
        margin-bottom: 1.5rem;
    }

    .predicted-digit {
        font-size: 4.5rem;
        font-weight: 800;
        color: #0F172A !important;
        line-height: 1;
        margin: 0.5rem 0;
    }

    .confidence-score {
        font-size: 1.15rem;
        font-weight: 600;
        color: #2563EB !important;
    }

    /* Visible High-Contrast Buttons (Sidebar Presets & File Upload Buttons) */
    .stButton > button, 
    button[kind="secondary"],
    [data-testid="stBaseButton-secondary"],
    [data-testid="stFileUploader"] button,
    button {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
        border: 1.5px solid #CBD5E1 !important;
        border-radius: 6px !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1) !important;
        transition: all 0.15s ease-in-out !important;
    }

    .stButton > button:hover, 
    button[kind="secondary"]:hover,
    [data-testid="stFileUploader"] button:hover {
        background-color: #2563EB !important;
        color: #FFFFFF !important;
        border-color: #2563EB !important;
    }

    /* File Uploader styling */
    [data-testid="stFileUploader"] {
        background-color: #F8FAFC !important;
        border: 2px dashed #CBD5E1 !important;
        border-radius: 8px !important;
        padding: 12px !important;
    }

    [data-testid="stFileUploader"] section {
        background-color: #F8FAFC !important;
    }

    [data-testid="stFileUploader"] span, [data-testid="stFileUploader"] small, [data-testid="stFileUploader"] p {
        color: #0F172A !important;
        font-weight: 600 !important;
    }

    /* Chart Background & Legend Styling */
    [data-testid="stVegaLiteChart"] {
        background-color: #FFFFFF !important;
        border-radius: 6px !important;
        padding: 8px !important;
        border: 1px solid #E2E8F0 !important;
    }

    /* Explanation Info Box */
    .info-panel {
        background-color: #F1F5F9 !important;
        border-left: 4px solid #2563EB !important;
        padding: 1rem 1.25rem;
        border-radius: 4px;
        color: #1E293B !important;
        font-size: 0.95rem;
        line-height: 1.5;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Cached Model Loader
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_cnn_model(model_path: Path):
    """
    Loads trained Keras CNN model from disk.
    Supports native TensorFlow Keras loader and Keras 3 format fallbacks.
    """
    if not model_path.exists():
        st.error(f"Model file not found at path: {model_path}")
        return None

    try:
        return tf.keras.models.load_model(model_path)
    except Exception:
        pass

    try:
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1), name="conv2d"),
            tf.keras.layers.MaxPooling2D((2, 2), name="max_pooling2d"),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu", name="conv2d_1"),
            tf.keras.layers.MaxPooling2D((2, 2), name="max_pooling2d_1"),
            tf.keras.layers.Flatten(name="flatten"),
            tf.keras.layers.Dense(128, activation="relu", name="dense"),
            tf.keras.layers.Dropout(0.5, name="dropout"),
            tf.keras.layers.Dense(10, activation="softmax", name="dense_1")
        ])
        _ = model(tf.zeros((1, 28, 28, 1)))

        if zipfile.is_zipfile(model_path):
            with zipfile.ZipFile(model_path) as z:
                if "model.weights.h5" in z.namelist():
                    weights_bytes = z.read("model.weights.h5")
                    with h5py.File(io.BytesIO(weights_bytes), "r") as f:
                        model.get_layer("conv2d").set_weights([f["layers/conv2d/vars/0"][()], f["layers/conv2d/vars/1"][()]])
                        model.get_layer("conv2d_1").set_weights([f["layers/conv2d_1/vars/0"][()], f["layers/conv2d_1/vars/1"][()]])
                        model.get_layer("dense").set_weights([f["layers/dense/vars/0"][()], f["layers/dense/vars/1"][()]])
                        model.get_layer("dense_1").set_weights([f["layers/dense_1/vars/0"][()], f["layers/dense_1/vars/1"][()]])
                    return model
    except Exception as err:
        st.error(f"Error loading model weights: {err}")
        return None

    return None

# -----------------------------------------------------------------------------
# Preprocessing helpers
# -----------------------------------------------------------------------------
def _otsu_threshold(gray_uint8: np.ndarray) -> int:
    """Otsu's method: picks the threshold that best separates ink from
    background for THIS image, instead of a fixed percentage-of-max guess."""
    hist, _ = np.histogram(gray_uint8.flatten(), bins=256, range=(0, 256))
    total = gray_uint8.size
    sum_total = np.dot(np.arange(256), hist)
    sum_b, w_b, max_var, threshold = 0.0, 0, 0.0, 0
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        var_between = w_b * w_f * (m_b - m_f) ** 2
        if var_between > max_var:
            max_var = var_between
            threshold = t
    return threshold

def _dilate(arr: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Simple 3x3 max-filter dilation (no scipy/cv2 dependency needed) so thin
    single-stroke lines get thickened toward MNIST's thicker pen strokes."""
    result = arr.copy()
    for _ in range(max(0, iterations)):
        padded = np.pad(result, 1, mode="constant", constant_values=0)
        neighborhood = np.stack([
            padded[1:-1, 1:-1], padded[:-2, 1:-1], padded[2:, 1:-1],
            padded[1:-1, :-2], padded[1:-1, 2:], padded[:-2, :-2],
            padded[:-2, 2:], padded[2:, :-2], padded[2:, 2:],
        ])
        result = np.max(neighborhood, axis=0)
    return result

# Robust MNIST Preprocessing Pipeline

def preprocess_digit_image(raw_image: Image.Image, thicken_strokes: bool = False):
    """
    Preprocesses uploaded handwritten digit images into standard MNIST format.
    Supports both Single Digit and Multi-Digit Recognition (e.g. '41', '852'):
    1. Estimate local background via Gaussian blur to handle paper lighting gradients & shadows.
    2. Extract local ink contrast independently of paper color or desk borders.
    3. Perform connected component segmentation to isolate individual digit strokes.
    4. Group horizontally adjacent strokes left-to-right across the image.
    5. Crop each digit stroke cluster tightly and scale to fit a 20x20 box.
    6. Center each digit on a 28x28 black canvas using Center of Mass (Centroid) alignment.
    7. Normalize pixel values to [0.0, 1.0] for model input tensor shape (1, 28, 28, 1).

    Returns a list of dicts: [{'tensor': input_tensor, 'canvas': canvas_28x28, 'cropped': cropped_pil, 'bbox': (x1,y1,x2,y2)}, ...]
    """
    gray = raw_image.convert("L")
    w_orig, h_orig = gray.size

    # Downscale for processing if image is high-res photo (max dim 600)
    max_dim = max(w_orig, h_orig)
    if max_dim > 600:
        scale = 600.0 / max_dim
        gray_proc = gray.resize((int(round(w_orig * scale)), int(round(h_orig * scale))), Image.Resampling.BILINEAR)
    else:
        gray_proc = gray

    arr = np.array(gray_proc, dtype=np.float32)
    h, w = arr.shape

    # 1. Local background estimation to remove page shadows & lighting gradients
    blur_r = max(10, min(h, w) // 15)
    bg_local = np.array(gray_proc.filter(ImageFilter.GaussianBlur(radius=blur_r)), dtype=np.float32)

    diff_dark = bg_local - arr
    diff_light = arr - bg_local

    # Determine polarity (dark ink on light paper vs light ink on dark paper)
    if diff_dark.max() >= diff_light.max():
        diff = diff_dark
    else:
        diff = diff_light

    diff = np.maximum(0.0, diff)

    # 2. Thresholding and connected component filtering
    thresh = max(18.0, diff.max() * 0.22)
    binary_mask = (diff > thresh).astype(np.int32)

    labeled_array, num_features = label(binary_mask)
    margin_h = max(2, int(h * 0.04))
    margin_w = max(2, int(w * 0.04))
    digit_comps = []

    for i in range(1, num_features + 1):
        comp_pixels = np.where(labeled_array == i)
        comp_size = len(comp_pixels[0])
        if comp_size < 30:  # Filter out small noise dots & dust specks
            continue

        y_min, y_max = int(np.min(comp_pixels[0])), int(np.max(comp_pixels[0]))
        x_min, x_max = int(np.min(comp_pixels[1])), int(np.max(comp_pixels[1]))
        c_w = x_max - x_min + 1
        c_h = y_max - y_min + 1

        if c_h < 10 or c_w < 4:  # Drop tiny noise lines
            continue

        # Drop outer photo border frame/shadows (spans large width/height near outer margins)
        touches_border = (
            y_min <= margin_h or y_max >= h - margin_h or
            x_min <= margin_w or x_max >= w - margin_w
        )
        is_large_frame = (c_w > 0.35 * w or c_h > 0.7 * h)
        if touches_border and is_large_frame:
            continue

        cy = (y_min + y_max) / 2.0
        digit_comps.append({"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max, "w": c_w, "h": c_h, "cy": cy, "size": comp_size})

    if not digit_comps:
        digit_crops = [{"x_min": 0, "x_max": w - 1, "y_min": 0, "y_max": h - 1, "size": h * w}]
    else:
        # Text Line Alignment Filter:
        # Digits in a handwritten text line share similar height and vertical center
        heights = [c["h"] for c in digit_comps]
        median_h = np.median(heights)
        
        digit_candidates = [c for c in digit_comps if c["h"] >= 0.3 * median_h and c["w"] < 0.45 * w]
        if digit_candidates:
            cys = [c["cy"] for c in digit_candidates]
            median_cy = np.median(cys)
            filtered_comps = [c for c in digit_candidates if abs(c["cy"] - median_cy) <= max(30.0, 0.4 * h)]
        else:
            filtered_comps = digit_comps

        if not filtered_comps:
            filtered_comps = digit_comps

        # Sort left-to-right & merge sub-strokes of the SAME digit
        filtered_comps = sorted(filtered_comps, key=lambda c: c["x_min"])
        merged = []
        for box in filtered_comps:
            if not merged:
                merged.append(box)
            else:
                prev = merged[-1]
                x_gap = box["x_min"] - prev["x_max"]
                y_overlap = min(prev["y_max"], box["y_max"]) - max(prev["y_min"], box["y_min"])
                
                # Merge sub-strokes of the SAME digit (horizontal overlap or gap <= 2px with vertical overlap)
                if x_gap <= 2 and y_overlap > 0:
                    prev["x_min"] = min(prev["x_min"], box["x_min"])
                    prev["x_max"] = max(prev["x_max"], box["x_max"])
                    prev["y_min"] = min(prev["y_min"], box["y_min"])
                    prev["y_max"] = max(prev["y_max"], box["y_max"])
                else:
                    merged.append(box)
        digit_crops = merged

    results = []
    for crop_info in digit_crops:
        y_min, y_max = crop_info["y_min"], crop_info["y_max"]
        x_min, x_max = crop_info["x_min"], crop_info["x_max"]

        crop_h, crop_w = y_max - y_min + 1, x_max - x_min + 1
        pad_h = max(2, int(crop_h * 0.14))
        pad_w = max(2, int(crop_w * 0.14))

        y_min_pad = max(0, y_min - pad_h)
        y_max_pad = min(h - 1, y_max + pad_h)
        x_min_pad = max(0, x_min - pad_w)
        x_max_pad = min(w - 1, x_max + pad_w)

        cropped = diff[y_min_pad : y_max_pad + 1, x_min_pad : x_max_pad + 1]
        if cropped.max() > 0:
            cropped = (cropped / cropped.max()) * 255.0
        cropped_pil = Image.fromarray(cropped.astype(np.uint8))

        cw, ch = cropped_pil.size
        scale = 20.0 / max(cw, ch)
        new_w = max(1, int(round(cw * scale)))
        new_h = max(1, int(round(ch * scale)))

        resized_pil = cropped_pil.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)

        # Apply stroke dilation ONLY if user explicitly enabled it for thin ballpoint lines
        if thicken_strokes:
            resized_pil = resized_pil.filter(ImageFilter.MaxFilter(3))

        canvas_28x28 = Image.new("L", (28, 28), color=0)
        paste_x = (28 - new_w) // 2
        paste_y = (28 - new_h) // 2
        canvas_28x28.paste(resized_pil, (paste_x, paste_y))

        # Center of Mass (Centroid) Alignment to (14, 14)
        canvas_arr = np.array(canvas_28x28, dtype=np.float32)
        total_mass = canvas_arr.sum()
        if total_mass > 0:
            y_indices, x_indices = np.indices((28, 28))
            cy = (y_indices * canvas_arr).sum() / total_mass
            cx = (x_indices * canvas_arr).sum() / total_mass

            shift_x = int(round(14.0 - cx))
            shift_y = int(round(14.0 - cy))

            if abs(shift_x) <= 4 and abs(shift_y) <= 4 and (shift_x != 0 or shift_y != 0):
                canvas_28x28 = ImageChops.offset(canvas_28x28, shift_x, shift_y)

        canvas_28x28 = canvas_28x28.filter(ImageFilter.GaussianBlur(radius=0.3))
        tensor_arr = np.array(canvas_28x28, dtype=np.float32) / 255.0
        input_tensor = tensor_arr.reshape(1, 28, 28, 1)

        results.append({
            "tensor": input_tensor,
            "canvas": canvas_28x28,
            "cropped": cropped_pil,
            "bbox": (x_min, y_min, x_max, y_max),
            "low_signal": bool(total_mass < 200)
        })

    return results

def generate_sample_digit(digit: int) -> Image.Image:
    """Generates synthetic handwritten digit sample for demonstration."""
    img = Image.new("L", (220, 220), color=220)
    draw = ImageDraw.Draw(img)
    stroke = 18
    if digit == 0:
        draw.ellipse([45, 35, 175, 185], outline=30, width=stroke)
    elif digit == 1:
        draw.line([(110, 35), (110, 185)], fill=30, width=stroke)
        draw.line([(75, 60), (110, 35)], fill=30, width=stroke)
    elif digit == 2:
        draw.arc([45, 35, 175, 120], start=180, end=0, fill=30, width=stroke)
        draw.line([(175, 75), (45, 185)], fill=30, width=stroke)
        draw.line([(45, 185), (175, 185)], fill=30, width=stroke)
    elif digit == 3:
        draw.arc([45, 35, 175, 110], start=210, end=90, fill=30, width=stroke)
        draw.arc([45, 105, 175, 185], start=270, end=150, fill=30, width=stroke)
    elif digit == 4:
        draw.line([(155, 35), (55, 120)], fill=30, width=stroke)
        draw.line([(55, 120), (175, 120)], fill=30, width=stroke)
        draw.line([(155, 35), (155, 185)], fill=30, width=stroke)
    elif digit == 5:
        draw.line([(165, 35), (65, 35)], fill=30, width=stroke)
        draw.line([(65, 35), (65, 100)], fill=30, width=stroke)
        draw.arc([65, 90, 165, 185], start=270, end=90, fill=30, width=stroke)
    elif digit == 6:
        draw.arc([55, 35, 165, 185], start=45, end=270, fill=30, width=stroke)
        draw.ellipse([55, 100, 165, 185], outline=30, width=stroke)
    elif digit == 7:
        draw.line([(55, 35), (165, 35)], fill=30, width=stroke)
        draw.line([(165, 35), (75, 185)], fill=30, width=stroke)
    elif digit == 8:
        draw.ellipse([60, 35, 160, 105], outline=30, width=stroke)
        draw.ellipse([55, 100, 165, 185], outline=30, width=stroke)
    elif digit == 9:
        draw.ellipse([55, 35, 165, 120], outline=30, width=stroke)
        draw.line([(165, 75), (165, 185)], fill=30, width=stroke)
        draw.arc([55, 100, 165, 185], start=270, end=0, fill=30, width=stroke)
    return img

# Main Application Interface

def main():
    # Application Header
    st.markdown("""
    <div class="app-header">
        <div class="app-title">Handwritten Digit Recognition</div>
        <div class="app-subtitle">Handwritten character recognition system supporting single & multi-digit numbers</div>
    </div>
    """, unsafe_allow_html=True)

    # Load Model
    model = load_cnn_model(MODEL_PATH)

    if model is None:
        st.error("Unable to load CNN model from models/handwritten_digit_cnn.keras")
        st.stop()

    # Sidebar Controls
    with st.sidebar:
        st.header("Controls")
        toggle_fn = getattr(st, "toggle", st.checkbox)
        thicken_strokes = toggle_fn(
            "Thicken Thin Strokes (for ballpoint pens)",
            value=False,
            help="Enable ONLY if your handwriting is drawn with an extremely thin ballpoint pen line. Keep OFF for normal/marker strokes (like digits 4 and 1) to prevent open strokes from closing."
        )

        st.markdown("---")
        st.header("Sample Digit Presets")
        st.write("Click a digit below to test sample images:")

        preset_cols = st.columns(5)
        selected_preset = None
        for i in range(10):
            col_idx = i % 5
            if preset_cols[col_idx].button(f"{i}", key=f"preset_{i}"):
                selected_preset = i

        st.markdown("---")
        st.header("Model Specifications")
        st.write("""
        - Input Tensor: 28x28x1 Grayscale
        - Architecture: Sequential 2D CNN
        - Multi-Digit Mode: Supported
        - Target Classes: 0 to 9
        """)

    active_image = None
    uploaded_file = st.file_uploader(
        "Upload handwritten digit image (PNG, JPG, JPEG) - Single or Multi-Digit (e.g., '41', '852')",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:
        try:
            active_image = Image.open(uploaded_file)
        except Exception as err:
            st.error(f"Unable to read image file: {err}")
            active_image = None
    elif selected_preset is not None:
        active_image = generate_sample_digit(selected_preset)

    if active_image is not None:
        digit_items = preprocess_digit_image(active_image, thicken_strokes=thicken_strokes)
        num_detected = len(digit_items)

        # Process predictions for all detected digits
        predictions_list = []
        for item in digit_items:
            preds = model.predict(item["tensor"], verbose=0)[0]
            pred_digit = int(np.argmax(preds))
            confidence = float(np.max(preds) * 100)
            predictions_list.append({
                "digit": pred_digit,
                "confidence": confidence,
                "probabilities": preds,
                "canvas": item["canvas"],
                "cropped": item["cropped"],
                "low_signal": item["low_signal"]
            })

        full_sequence = "".join([str(p["digit"]) for p in predictions_list])

        # Main Page Layout
        col_input, col_result = st.columns([1, 1], gap="large")

        with col_input:
            st.subheader("Original Upload & Preprocessing")
            st.image(active_image, use_column_width=True)
            if num_detected > 1:
                st.info(f"Detected **{num_detected} digits** in uploaded image!")

        with col_result:
            st.subheader("Prediction Output")

            if num_detected == 1:
                p = predictions_list[0]
                st.markdown(f"""
                <div class="result-card">
                    <div style="font-size: 0.9rem; font-weight: 600; color: #475569; text-transform: uppercase;">Predicted Digit</div>
                    <div class="predicted-digit">{p['digit']}</div>
                    <div class="confidence-score">Confidence: {p['confidence']:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)

                st.subheader("28x28 Processed MNIST Input")
                st.image(p["canvas"], width=180, caption="Centered 20x20 digit on 28x28 canvas")

                st.subheader("Class Probability Distribution")
                chart_data = {f"Digit {i}": float(p["probabilities"][i] * 100) for i in range(10)}
                st.bar_chart(chart_data, height=200)

            else:
                # Multi-Digit Banner Display Card
                st.markdown(f"""
                <div class="result-card">
                    <div style="font-size: 0.9rem; font-weight: 600; color: #475569; text-transform: uppercase;">Recognized Number Sequence</div>
                    <div class="predicted-digit" style="letter-spacing: 0.25rem;">{full_sequence}</div>
                    <div class="confidence-score">Detected {num_detected} digits</div>
                </div>
                """, unsafe_allow_html=True)

                st.subheader("Individual Digit Breakdown")
                digit_cols = st.columns(min(num_detected, 4))
                for idx, p in enumerate(predictions_list):
                    c_idx = idx % min(num_detected, 4)
                    with digit_cols[c_idx]:
                        st.markdown(f"**Digit #{idx+1}: {p['digit']}**")
                        st.image(p["canvas"], width=120)
                        st.caption(f"Conf: {p['confidence']:.1f}%")
                        chart_data = {f"D{i}": float(p["probabilities"][i] * 100) for i in range(10)}
                        st.bar_chart(chart_data, height=130)

        st.markdown("""
        <div class="info-panel">
            <strong>MNIST Preprocessing Summary:</strong> Image was analyzed via connected component segmentation, 
            local background illumination removal, centroid (Center of Mass) 14x14 alignment, 
            and normalized to 28x28 grayscale tensors matching MNIST data format.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Upload an image or select a sample digit from the sidebar.")

if __name__ == "__main__":
    main()