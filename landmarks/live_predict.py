import argparse
import json
import os
import time
import numpy as np
import tensorflow as tf
import mediapipe as mp
import cv2
import math

from tts_module import create_synthesizer

def load_artifacts(models_dir):
    model_path = os.path.join(models_dir, "ann_landmarks.keras")
    label_map_path = os.path.join(models_dir, "label_map.json")
    norm_path = os.path.join(models_dir, "normalization.npz")
    
    if not all(os.path.exists(p) for p in [model_path, label_map_path, norm_path]):
        print(f"Error: Missing artifacts in {models_dir}")
        return None, None, None, None
        
    model = tf.keras.models.load_model(model_path)
    with open(label_map_path, "r") as f:
        label_to_id = json.load(f)
    id_to_label = {int(v): k for k, v in label_to_id.items()}
    norm = np.load(norm_path)
    return model, id_to_label, norm["mean"], norm["std"]

def load_tflite(models_dir):
    tflite_path = os.path.join(models_dir, "ann_landmarks.tflite")
    if not os.path.exists(tflite_path):
        return None, None, None
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    in_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]
    return interpreter, in_det, out_det

def build_landmarker(models_dir):
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    
    p = os.path.join(models_dir, "hand_landmarker.task")
    if not os.path.exists(p):
        p = os.path.join("models", "hand_landmarker.task")
        
    base_options = mp_python.BaseOptions(model_asset_path=p)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        running_mode=vision.RunningMode.VIDEO
    )
    return vision.HandLandmarker.create_from_options(options)

def norm_vec(pts):
    if not pts:
        return None
    
    points = np.array([[lm.x, lm.y, lm.z] for lm in pts])
    
    wrist = points[0]
    normalized = points - wrist
    
    hand_size = np.linalg.norm(normalized[9])  
    if hand_size > 0:
        normalized = normalized / hand_size
    
    return normalized.flatten()

def landmarks_to_vector(result):
    left_hand = None
    right_hand = None
    
    if hasattr(result, 'hand_landmarks') and result.hand_landmarks:
        for idx, handedness in enumerate(result.handedness):
            label = handedness[0].category_name.lower()
            lms = result.hand_landmarks[idx]
            vec = norm_vec(lms)
            
            if label == 'left':
                left_hand = vec
            elif label == 'right':
                right_hand = vec
    else:
        return None

    # If only one hand is detected, the training data ALWAYS puts it in the first slot (left_hand variable)
    # regardless of whether it was actually a left or right hand. We must match this EXACTLY.
    if left_hand is None and right_hand is not None:
        left_hand = right_hand
        right_hand = None
    
    # If no hands at all, return None so the caller treats it as idle
    if left_hand is None and right_hand is None:
        return None
        
    # Create feature vector (126 dimensions for 2 hands)
    features = []
    if left_hand is not None:
        features.extend(left_hand)
    else:
        features.extend([0.0] * 63)
        
    if right_hand is not None:
        features.extend(right_hand)
    else:
        features.extend([0.0] * 63)
        
    return np.array(features, dtype=np.float32)

def draw_landmarks(frame, result):
    if hasattr(result, 'hand_landmarks') and result.hand_landmarks:
        h, w = frame.shape[:2]
        for hand_lms in result.hand_landmarks:
            for pt in hand_lms:
                x, y = int(pt.x * w), int(pt.y * h)
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera-index", type=int, default=0)
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--min-conf", type=float, default=0.45)
    ap.add_argument("--no-mirror", dest="mirror", action="store_false", help="Disable horizontal mirroring")
    ap.set_defaults(mirror=True)
    ap.add_argument("--smooth-window", type=int, default=15)
    ap.add_argument("--stable-frames", type=int, default=12)
    ap.add_argument("--use-tflite", action="store_true")
    ap.add_argument("--hold-sec", type=float, default=1.5)
    ap.add_argument("--no-hand-frames", type=int, default=5)
    ap.add_argument("--speak", action="store_true")
    args = ap.parse_args()

    model, id_to_label, mean, std = load_artifacts(args.models_dir)
    if model is None: return
    
    tflite_interpreter = None
    in_det = None
    out_det = None
    if args.use_tflite:
        tflite_interpreter, in_det, out_det = load_tflite(args.models_dir)
        if tflite_interpreter is None:
            print("Warning: TFLite model not found, using Keras model.")
            args.use_tflite = False
    
    landmarker = build_landmarker(args.models_dir)
    synth = create_synthesizer() if args.speak else None

    cap = cv2.VideoCapture(args.camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"Error: Could not open camera {args.camera_index}")
        return

    # Stabilization buffers
    prob_buffer = []
    稳定_prediction = None # Currently locked-in word
    稳定_count = 0        # How many frames the current top has been stable
    last_top_id = -1
    
    last_spoken = ""
    last_speak_time = 0
    
    print("\n" + "="*40)
    print("LIVE PREDICTION - STABILITY MODE")
    print("Mirroring:", "ENABLED" if args.mirror else "DISABLED")
    print("Press 'q' to exit.")
    print("="*40 + "\n")
    
    start_time = time.time()
    last_print_time = 0
    capturing = False
    capture_start = 0.0
    no_hand_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        if args.mirror:
            frame = cv2.flip(frame, 1)
            
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        current_ms = int((time.time() - start_time) * 1000)
        result = landmarker.detect_for_video(mp_image, current_ms)
        
        vector = landmarks_to_vector(result)
        
        display_text = "Searching for hands..."
        conf_val = 0.0
        
        if vector is not None:
            no_hand_count = 0
            if not capturing:
                capturing = True
                capture_start = time.time()
                prob_buffer.clear()
            vector = (vector - mean) / std
            if args.use_tflite and tflite_interpreter is not None:
                x = vector.reshape(1, -1).astype(np.float32)
                tflite_interpreter.set_tensor(in_det["index"], x)
                tflite_interpreter.invoke()
                raw_probs = tflite_interpreter.get_tensor(out_det["index"])[0]
            else:
                raw_probs = model.predict(vector.reshape(1, -1), verbose=0)[0]
            
            prob_buffer.append(raw_probs)
            if len(prob_buffer) > args.smooth_window:
                prob_buffer.pop(0)
            
            avg_probs = np.mean(prob_buffer, axis=0)
            top_idx = np.argsort(avg_probs)[-3:][::-1]
            current_top_id = top_idx[0]
            current_conf = avg_probs[current_top_id]
            
            if current_top_id == last_top_id:
                稳定_count += 1
            else:
                稳定_count = 0
            last_top_id = current_top_id
            
            ready = (time.time() - capture_start) >= args.hold_sec
            if ready and 稳定_count >= args.stable_frames and current_conf >= args.min_conf:
                稳定_prediction = id_to_label[current_top_id]
                conf_val = current_conf
            
            if time.time() - last_print_time > 0.5:
                top_labels = [f"{id_to_label[i]}: {avg_probs[i]:.2f}" for i in top_idx]
                print(f"Top 3: {' | '.join(top_labels)} [Stable: {稳定_count}]")
                last_print_time = time.time()

            if 稳定_prediction:
                display_text = f"PREDICTION: {稳定_prediction}"
                if synth and 稳定_prediction != last_spoken:
                    if time.time() - last_speak_time > 2.0:
                        synth.speak(稳定_prediction)
                        last_spoken = 稳定_prediction
                        last_speak_time = time.time()
            else:
                if capturing:
                    display_text = "Analyzing..."
                else:
                    display_text = "NO SIGN PERFORMED"
                
            draw_landmarks(frame, result)
        else:
            no_hand_count += 1
            capturing = False
            prob_buffer.clear()
            稳定_prediction = None
            稳定_count = 0
            last_top_id = -1
            last_spoken = ""
            if no_hand_count >= args.no_hand_frames:
                display_text = "NO SIGN PERFORMED"
            else:
                display_text = "Searching for hands..."

        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 60), (40, 40, 40), -1)
        cv2.putText(frame, display_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        
        if conf_val > 0:
            bar_w = int(conf_val * 200)
            cv2.rectangle(frame, (20, 65), (20 + bar_w, 75), (0, 255, 0), -1)
            cv2.putText(frame, f"{conf_val*100:.0f}%", (230, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow("Sign Translation - High Stability", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()

if __name__ == "__main__":
    main()
