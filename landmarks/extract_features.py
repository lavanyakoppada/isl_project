import cv2
import mediapipe as mp
import numpy as np
import os
import csv
import sys

# ===== MediaPipe Setup =====
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True, 
    max_num_hands=1, 
    min_detection_confidence=0.5
)

def extract_landmarks(image_path):
    """
    Reads an image, processes it with MediaPipe, and extracts normalized landmarks.
    Normalization is done relative to the wrist (landmark 0).
    """
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    if not results.multi_hand_landmarks:
        return None
    
    hand_landmarks = results.multi_hand_landmarks[0]
    
    # Find center of hand (centroid)
    xs = [lm.x for lm in hand_landmarks.landmark]
    ys = [lm.y for lm in hand_landmarks.landmark]
    zs = [lm.z for lm in hand_landmarks.landmark]
    
    center_x = sum(xs) / len(xs)
    center_y = sum(ys) / len(ys)
    center_z = sum(zs) / len(zs)
    
    # Translation: bring centroid exactly to (0,0,0)
    translated_x = [x - center_x for x in xs]
    translated_y = [y - center_y for y in ys]
    translated_z = [z - center_z for z in zs]
    
    # Scale: normalize all points so the max absolute extent is exactly 1.0 (bounding box size)
    max_value = max(
        max(map(abs, translated_x)),
        max(map(abs, translated_y)),
        max(map(abs, translated_z))
    )
    
    # Avoid division by zero
    if max_value == 0:
        max_value = 1e-4

    features = []
    for tx, ty, tz in zip(translated_x, translated_y, translated_z):
        features.extend([
            tx / max_value,
            ty / max_value,
            tz / max_value
        ])
        
    return features


def process_dataset_split(split_name, base_dir, output_csv):
    """
    Iterates through all classes in a train/test split, extracting features and saving to CSV.
    """
    split_dir = os.path.join(base_dir, split_name)
    if not os.path.exists(split_dir):
        print(f"Error: Directory {split_dir} does not exist.")
        return
    
    print(f"Processing {split_name} split...")
    
    classes = sorted([d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))])
    
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Create header: label, x0, y0, z0, ..., x20, y20, z20
        header = ['label']
        for i in range(21):
            header.extend([f'x{i}', f'y{i}', f'z{i}'])
        writer.writerow(header)
        
        total_images = 0
        successful_extractions = 0
        
        for class_name in classes:
            class_dir = os.path.join(split_dir, class_name)
            images = [img for img in os.listdir(class_dir) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            for img_name in images:
                total_images += 1
                img_path = os.path.join(class_dir, img_name)
                
                features = extract_landmarks(img_path)
                if features:
                    writer.writerow([class_name] + features)
                    successful_extractions += 1
                
                # Progress print output
                if total_images % 500 == 0:
                    print(f"Processed {total_images} images. Extracted {successful_extractions} hands.")
                    
        print(f"Finished {split_name} split: {successful_extractions}/{total_images} hands extracted successfully.")

if __name__ == "__main__":
    base_dataset_dir = r"d:\Project\ASL_Processed_Images\asl_processed"
    
    train_dest = "train_landmarks.csv"
    test_dest = "test_landmarks.csv"
    
    process_dataset_split("train", base_dataset_dir, train_dest)
    process_dataset_split("test", base_dataset_dir, test_dest)
