import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import pickle

# ===== LOAD MODEL AND CLASSES =====
try:
    model = tf.keras.models.load_model("landmark_model.h5")
    with open("classes.pkl", "rb") as f:
        labels = pickle.load(f)
except FileNotFoundError:
    print("Error: landmark_model.h5 or classes.pkl not found.")
    print("Run train.py first to generate the models.")
    import sys; sys.exit(1)

# ===== MEDIAPIPE HAND DETECTION =====
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ===== START WEBCAM =====
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Draw skeleton on screen
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Extract normalized landmarks
        xs = [lm.x for lm in hand_landmarks.landmark]
        ys = [lm.y for lm in hand_landmarks.landmark]
        zs = [lm.z for lm in hand_landmarks.landmark]
        
        center_x = sum(xs) / len(xs)
        center_y = sum(ys) / len(ys)
        center_z = sum(zs) / len(zs)
        
        translated_x = [x - center_x for x in xs]
        translated_y = [y - center_y for y in ys]
        translated_z = [z - center_z for z in zs]
        
        max_value = max(
            max(map(abs, translated_x)),
            max(map(abs, translated_y)),
            max(map(abs, translated_z))
        )
        if max_value == 0:
            max_value = 1e-4
            
        features = []
        for tx, ty, tz in zip(translated_x, translated_y, translated_z):
            features.extend([
                tx / max_value,
                ty / max_value,
                tz / max_value
            ])
            
        # Reshape to (1, 63) for model
        input_data = np.array(features).reshape(1, -1)
        
        # Predict
        prediction = model.predict(input_data, verbose=0)
        class_id = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        
        # Identify bounding box from landmarks dynamically so we can put text above it
        h, w, _ = frame.shape
        x_min = w
        y_min = h
        
        for lm in hand_landmarks.landmark:
            x, y = int(lm.x * w), int(lm.y * h)
            x_min = min(x_min, x)
            y_min = min(y_min, y)

        if confidence > 50:
            text = f"{labels[class_id]} ({confidence:.1f}%)"
            color = (0, 255, 0) # Green for confident hit
        else:
            text = f"{labels[class_id]}? ({confidence:.1f}%)"
            color = (0, 165, 255) # Orange for low confidence

        # Draw the text smoothly near the hand
        cv2.putText(frame, text, (max(0, x_min - 10), max(30, y_min - 20)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    else:
        cv2.putText(frame, "No hand detected", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("ASL Landmark Prediction", frame)

    if cv2.waitKey(1) & 0xFF == 27: # ESC
        break

cap.release()
cv2.destroyAllWindows()
