import cv2
import mediapipe as mp
import csv
import os
import numpy as np
import random
from typing import List, Tuple, Optional

DATASET_DIR = "images for phrases"
OUTPUT_CSV = "hand_landmarks.csv"
MIN_SAMPLES_PER_CLASS = 40
MAX_SAMPLES_PER_CLASS = 50
RANDOM_SEED = 42

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(model_asset_path='models/hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
detector = vision.HandLandmarker.create_from_options(options)

def normalize_landmarks(landmarks):
    if not landmarks:
        return None
    
    points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    
    wrist = points[0]
    normalized = points - wrist
    
    hand_size = np.linalg.norm(normalized[9])  
    if hand_size > 0:
        normalized = normalized / hand_size
    
    return normalized.flatten()

def extract_landmarks_from_image(image_path: str) -> Optional[Tuple[List[float], str]]:
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    
    results = detector.detect(mp_image)
    
    if results.hand_landmarks:
        left_hand = None
        right_hand = None
        
        for hand_landmarks, handedness in zip(results.hand_landmarks, results.handedness):
            hand_label = handedness[0].category_name.lower()
            normalized = normalize_landmarks(hand_landmarks)
            
            if hand_label == 'left':
                left_hand = normalized
            elif hand_label == 'right':
                right_hand = normalized
        
        if left_hand is None and right_hand is not None:
            left_hand = right_hand
            right_hand = None
        
        features = []
        if left_hand is not None:
            features.extend(left_hand)
        else:
            features.extend([0.0] * 63)
            
        if right_hand is not None:
            features.extend(right_hand)
        else:
            features.extend([0.0] * 63)
            
        return features, 'detected'
    
    return None

def extract_landmarks():
    print("Starting landmark extraction...")
    class_counts = {}
    for word in os.listdir(DATASET_DIR):
        word_path = os.path.join(DATASET_DIR, word)
        if os.path.isdir(word_path):
            image_files = [f for f in os.listdir(word_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
            class_counts[word] = len(image_files)
    
    print(f"Found {len(class_counts)} classes")
    print("Class distribution:")
    for word, count in sorted(class_counts.items()):
        print(f"  {word}: {count} samples")
    
    header = ["label", "origin_file"]
    for hand in ["left", "right"]:
        for i in range(21):
            header.extend([f"{hand}_x{i}", f"{hand}_y{i}", f"{hand}_z{i}"])
    
    total_samples = 0
    
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        
        for word in sorted(os.listdir(DATASET_DIR)):
            word_path = os.path.join(DATASET_DIR, word)
            if not os.path.isdir(word_path):
                continue
            
            print(f"\nProcessing '{word}'...")
            written_samples = 0
            
            image_files = [f for f in os.listdir(word_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
            image_files.sort()
            random.seed(RANDOM_SEED)
            if len(image_files) > MAX_SAMPLES_PER_CLASS:
                target = random.randint(MIN_SAMPLES_PER_CLASS, MAX_SAMPLES_PER_CLASS)
                image_files = random.sample(image_files, target)
            
            for img_file in image_files:
                img_path = os.path.join(word_path, img_file)
                
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                result = extract_landmarks_from_image(img_path)
                if result:
                    features, hand_label = result
                    writer.writerow([word, img_file] + features)
                    written_samples += 1
                    total_samples += 1
            
            print(f"  Written: {written_samples}")
    
    print(f"\nLandmark extraction complete")
    print(f"Total samples: {total_samples}")
    print(f"Output saved to: {OUTPUT_CSV}")
    
    final_counts = {}
    with open(OUTPUT_CSV, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if row:
                label = row[0]
                final_counts[label] = final_counts.get(label, 0) + 1
    
    print(f"\nFinal class distribution:")
    for word, count in sorted(final_counts.items()):
        print(f"  {word}: {count} samples")

if __name__ == "__main__":
    extract_landmarks()
