import json
from pathlib import Path

import numpy as np
import gradio as gr
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from matplotlib import colormaps

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "best_baseline_model.keras"
CLASSES_PATH = BASE_DIR / "models" / "class_names.json"
IMG_SIZE = (224, 224)
LAST_CONV_LAYER = None  # set to the layer chosen in notebook 08, or None for auto

model = keras.models.load_model(MODEL_PATH)
with open(CLASSES_PATH) as f:
    class_names = json.load(f)

conv_name = LAST_CONV_LAYER or [
    l.name for l in model.layers if isinstance(l, keras.layers.Conv2D)
][-1]

_inputs = keras.Input(shape=(224, 224, 3))
_x, _target = _inputs, None
for layer in model.layers:
    if isinstance(layer, keras.layers.InputLayer):
        continue
    _x = layer(_x)
    if layer.name == conv_name:
        _target = _x
grad_model = keras.models.Model(_inputs, [_target, _x])


def gradcam(arr, class_idx):
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(arr)
        score = preds[:, class_idx]
    grads = tape.gradient(score, conv_out)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.reduce_sum(conv_out[0] * weights, axis=-1)
    cam = tf.maximum(cam, 0)
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    return cam.numpy()


def overlay(img_u8, cam, alpha=0.5, threshold=0.2):
    h, w = img_u8.shape[:2]
    cam_img = Image.fromarray(np.uint8(cam * 255)).resize((w, h), Image.BICUBIC)
    cam_r = np.asarray(cam_img, dtype="float32") / 255.0
    cam_r[cam_r < threshold] = 0
    colored = (colormaps["jet"](cam_r)[..., :3] * 255).astype("uint8")
    blended = (img_u8 * (1 - alpha) + colored * alpha).astype("uint8")
    return np.where((cam_r > 0)[..., None], blended, img_u8), cam_r


def pretty(name):
    crop, _, disease = name.partition("___")
    return f"{crop.replace('_', ' ')} - {disease.replace('_', ' ')}"


def predict(image):
    if image is None:
        return None, {}, "Please upload a leaf image."

    img = image.convert("RGB").resize(IMG_SIZE)
    img_u8 = np.array(img)
    arr = (img_u8.astype("float32") / 255.0)[None]

    probs = model.predict(arr, verbose=0)[0]
    top = np.argsort(probs)[::-1][:3]
    pred = int(top[0])

    cam = gradcam(tf.constant(arr), pred)
    blended, cam_r = overlay(img_u8, cam)

    border = np.ones_like(cam_r, dtype=bool)
    border[20:-20, 20:-20] = False
    edge_share = cam_r[border].sum() / (cam_r.sum() + 1e-8)

    notes = []
    if probs[pred] < 0.7:
        notes.append("Low confidence. Try a clearer, closer photo of one leaf.")
    if edge_share > 0.5:
        notes.append("Most attention is on the image edges or background, so treat this result with caution.")
    note = " ".join(notes) or "Red/yellow areas in the heatmap influenced the prediction most."

    labels = {pretty(class_names[i]): float(probs[i]) for i in top}
    return blended, labels, note


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Leaf image"),
    outputs=[
        gr.Image(label="Grad-CAM overlay"),
        gr.Label(num_top_classes=3, label="Prediction"),
        gr.Textbox(label="Note"),
    ],
    title="Crop Disease Detection with Grad-CAM",
    description=(
        "Upload a clear photo of a single leaf. Trained on PlantVillage (lab-style photos), "
        "so accuracy may drop on field photos. For academic demonstration only."
    ),
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()