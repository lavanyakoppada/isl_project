import argparse
import json
import os
import time
import numpy as np
import tensorflow as tf
import mediapipe as mp
import cv2
import math

# Hand Landmark Connections (Indices)
HAND_CONNECTIONS = [
    (0, 5), (5, 9), (9, 13), (13, 17), (17, 0),  # Palm
    (0, 1), (1, 2), (2, 3), (3, 4),               # Thumb
    (5, 6), (6, 7), (7, 8),                       # Index
    (9, 10), (10, 11), (11, 12),                 # Middle
    (13, 14), (14, 15), (15, 16),                # Ring
    (17, 18), (18, 19), (19, 20)                 # Pinky
]

def draw_landmarks_manual(image, landmarks):
    if not landmarks: return
    h, w = image.shape[:2]
    # Draw connections first
    for (start_idx, end_idx) in HAND_CONNECTIONS:
        if start_idx < len(landmarks) and end_idx < len(landmarks):
            p1 = (int(landmarks[start_idx].x * w), int(landmarks[start_idx].y * h))
            p2 = (int(landmarks[end_idx].x * w), int(landmarks[end_idx].y * h))
            cv2.line(image, p1, p2, (0, 255, 0), 2)
    # Draw points
    for lm in landmarks:
        px, py = int(lm.x * w), int(lm.y * h)
        cv2.circle(image, (px, py), 4, (0, 0, 255), -1)

def load_artifacts(models_dir):
    model = tf.keras.models.load_model(os.path.join(models_dir, "ann_landmarks.keras"))
    with open(os.path.join(models_dir, "label_map.json"), "r") as f:
        label_to_id = json.load(f)
    id_to_label = {int(v): k for k, v in label_to_id.items()}
    norm = np.load(os.path.join(models_dir, "normalization.npz"))
    return model, id_to_label, norm["mean"], norm["std"]

def build_landmarker(models_dir):
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    
    model_path = os.path.join(models_dir, "hand_landmarker.task")
    if not os.path.exists(model_path):
        model_path = os.path.join("models", "hand_landmarker.task")
        
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        running_mode=vision.RunningMode.VIDEO
    )
    return vision.HandLandmarker.create_from_options(options)

def norm_vec(pts):
    """Normalize landmarks exactly as in landmark_output.py."""
    if not pts:
        return None
    
    # Convert to numpy array (x, y, z)
    points = np.array([[lm.x, lm.y, lm.z] for lm in pts])
    
    # Normalize relative to wrist (landmark 0)
    wrist = points[0]
    normalized = points - wrist
    
    # Scale by hand size (distance between wrist and middle finger base - index 0 and 9)
    hand_size = np.linalg.norm(normalized[9])  
    if hand_size > 0:
        normalized = normalized / hand_size
    
    return normalized.flatten()

def landmarks_to_vector(result):
    """Convert MediaPipe result to exactly 126 features (63 Hand1, 63 Hand2)."""
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

    # Static fallback: if only one hand, MUST go to the first slot
    if left_hand is None and right_hand is not None:
        left_hand = right_hand
        right_hand = None
    
    # If no hands at all, return None to indicate idle
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
        
    # Return as float32 array instead of concatenating None objects
    return np.array(features, dtype=np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera-index", type=int, default=0)
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--min-conf", type=float, default=0.45)
    ap.add_argument("--no-mirror", dest="mirror", action="store_false")
    ap.set_defaults(mirror=True)
    ap.add_argument("--smooth-window", type=int, default=15)
    ap.add_argument("--stable-frames", type=int, default=12)
    ap.add_argument("--hold-sec", type=float, default=1.5)
    ap.add_argument("--no-hand-frames", type=int, default=5)
    ap.add_argument("--output-file", default="recorded_predictions.txt")
    args = ap.parse_args()

    print("--- Loading Model and Artifacts ---")
    model, id_to_label, mean, std = load_artifacts(args.models_dir)
    landmarker = build_landmarker(args.models_dir)
    
    cap = cv2.VideoCapture(args.camera_index, cv2.CAP_DSHOW)
    
    prob_buffer = []
    recorded_signs = []
    last_stable_label = None
    
    稳定_count = 0
    last_top_id = -1
    
    print("\n" + "="*40)
    print("RECORDING MODE - STABILITY LOCK")
    print(f"File: {args.output_file}")
    print("Press 'q' or 'ESC' to finish.")
    print("="*40 + "\n")

    start_time = time.time()
    last_print_time = 0
    capturing = False
    capture_start = 0.0
    no_hand_count = 0
    
    while True:
        ok, frame = cap.read()
        if not ok: break
        
        if args.mirror:
            frame = cv2.flip(frame, 1)
            
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        current_ms = int((time.time() - start_time) * 1000)
        r = landmarker.detect_for_video(mp_image, current_ms)
        x = landmarks_to_vector(r)
        
        display_text = "Searching for hands..."
        conf_val = 0.0

        # DRAW LANDMARKS ON SCREEN
        if hasattr(r, 'hand_landmarks') and r.hand_landmarks:
            for hand_landmarks in r.hand_landmarks:
                draw_landmarks_manual(frame, hand_landmarks)
        
        if x is not None:
            # Start/continue capture window
            no_hand_count = 0
            if not capturing:
                capturing = True
                capture_start = time.time()
                prob_buffer.clear()
            # Predict
            x = (x - mean) / std
            probs = model.predict(x.reshape(1, -1), verbose=0)[0]
            
            prob_buffer.append(probs)
            if len(prob_buffer) > args.smooth_window:
                prob_buffer.pop(0)
                
            avg_probs = np.mean(prob_buffer, axis=0)
            top_idx = np.argsort(avg_probs)[-3:][::-1]
            current_top_id = top_idx[0]
            current_conf = avg_probs[current_top_id]
            
            # Stability tracking
            if current_top_id == last_top_id:
                稳定_count += 1
            else:
                稳定_count = 0
            last_top_id = current_top_id
            
            # Terminal Debug (Top 3)
            if time.time() - last_print_time > 0.5:
                top_labels = [f"{id_to_label[i]}: {avg_probs[i]:.2f}" for i in top_idx]
                print(f"Top 3 (Rec): {' | '.join(top_labels)} [Stable: {稳定_count}]")
                last_print_time = time.time()

            # Lock only after hold time + stability + confidence
            ready = (time.time() - capture_start) >= args.hold_sec
            if ready and 稳定_count >= args.stable_frames and current_conf >= args.min_conf:
                label_name = id_to_label[current_top_id]
                conf_val = current_conf
                display_text = f"LOCKED: {label_name}"
                
                # Auto-record if it's a new unique sign in the sequence
                if label_name != last_stable_label:
                    recorded_signs.append(label_name)
                    last_stable_label = label_name
                    print(f"--- Captured Word: {label_name} ---")
            else:
                display_text = "Analyzing..."
        else:
            # No hands visible
            no_hand_count += 1
            capturing = False
            prob_buffer.clear()
            稳定_count = 0
            last_top_id = -1
            # DO NOT reset last_stable_label here
            # This prevents recording the same word twice if tracking stutters for 1 frame.
            if no_hand_count >= args.no_hand_frames:
                display_text = "NO SIGN PERFORMED"
            else:
                display_text = "Searching for hands..."

        # UI
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 60), (40, 40, 40), -1)
        cv2.putText(frame, display_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        if conf_val > 0:
            bw = int(conf_val * 200)
            cv2.rectangle(frame, (20, 65), (20+bw, 75), (0, 255, 0), -1)

        signs_str = " | ".join(recorded_signs[-4:])
        cv2.putText(frame, f"Stored Seq: {signs_str}", (20, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        cv2.imshow("Sign Sequence Recorder", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()
    
    if recorded_signs:
        output_content = " ".join(recorded_signs)
        with open(args.output_file, "w") as f:
            f.write(output_content)
        print(f"\nSaved {len(recorded_signs)} signs to {args.output_file}")
        print(f"Full Sequence: {output_content}")
    else:
        print("\nNo signs were captured.")

if __name__ == "__main__":
    main()
