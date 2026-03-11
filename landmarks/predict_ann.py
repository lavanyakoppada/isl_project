import argparse
import json
import os
import numpy as np
import tensorflow as tf
import mediapipe as mp
import cv2
import math
import importlib

def load_artifacts(models_dir):
    model = tf.keras.models.load_model(os.path.join(models_dir, "ann_landmarks.keras"))
    with open(os.path.join(models_dir, "label_map.json"), "r") as f:
        label_to_id = json.load(f)
    inv = {int(v): k for k, v in label_to_id.items()}
    norm = np.load(os.path.join(models_dir, "normalization.npz"))
    return model, inv, norm["mean"], norm["std"]

def extract_landmarks(img_path, landmarker):
    bgr = cv2.imread(img_path)
    if bgr is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)
    left_vec = None
    right_vec = None
    def norm_vec(pts):
        cx = pts[0].x
        cy = pts[0].y
        s = 0.0
        for q in pts:
            dx = q.x - cx
            dy = q.y - cy
            d = math.sqrt(dx * dx + dy * dy)
            if d > s:
                s = d
        if s < 1e-8:
            s = 1.0
        out = []
        for q in pts:
            out.extend([(q.x - cx) / s, (q.y - cy) / s, 0.0])
        return out
    hands_lms = getattr(result, "hand_landmarks", [])
    if getattr(result, "handedness", None) and hands_lms:
        for idx, handed_list in enumerate(result.handedness):
            if idx >= len(hands_lms):
                continue
            name = handed_list[0].category_name.lower() if handed_list else ""
            pts = hands_lms[idx]
            vec = norm_vec(pts)
            if name == "left":
                left_vec = vec
            elif name == "right":
                right_vec = vec
    def zeros63():
        return [0.0] * (21 * 3)
    if left_vec is None and right_vec is None and getattr(result, "hand_landmarks", None):
        pts = result.hand_landmarks[0]
        left_vec = norm_vec(pts)
    if left_vec is None and right_vec is None:
        return None
    arr = (left_vec if left_vec is not None else zeros63()) + (right_vec if right_vec is not None else zeros63())
    return np.asarray(arr, dtype=np.float32)

def build_tts(args):
    try:
        pyttsx3 = importlib.import_module("pyttsx3")
    except Exception:
        return None
    eng = pyttsx3.init()
    try:
        eng.setProperty("rate", args.speak_rate)
        eng.setProperty("volume", 1.0)
    except Exception:
        pass
    return eng


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image")
    p.add_argument("--models-dir", default="models")
    p.add_argument("--dataset-dir", default="images for phrases")
    p.add_argument("--speak", action="store_true")
    p.add_argument("--speak-rate", type=int, default=180)
    args = p.parse_args()

    model, id_to_label, mean, std = load_artifacts(args.models_dir)
    eng = None
    if args.speak:
        eng = build_tts(args)
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    model_path = os.path.join(args.models_dir, "hand_landmarker.task")
    if not os.path.exists(model_path):
        model_path = os.path.join("models", "hand_landmarker.task")
    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        num_hands=2,
        running_mode=vision.RunningMode.IMAGE
    )
    landmarker = vision.HandLandmarker.create_from_options(options)
    img_path = args.image
    if not img_path:
        base = args.dataset_dir
        subdirs = [d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]
        subdirs.sort()
        for d in subdirs:
            cand = [f for f in os.listdir(os.path.join(base, d)) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if cand:
                img_path = os.path.join(base, d, cand[0])
                break
    x = extract_landmarks(img_path, landmarker) if img_path else None
    landmarker.close()
    if x is None:
        print("No landmarks detected")
        return
    x = (x - mean) / std
    probs = model.predict(x.reshape(1, -1), verbose=0)[0]
    pred_id = int(np.argmax(probs))
    label = id_to_label[pred_id]
    print("Pred:", label, "Conf:", float(np.max(probs)))
    if eng is not None:
        try:
            eng.say(label)
            eng.runAndWait()
        except Exception:
            pass

if __name__ == "__main__":
    main()
