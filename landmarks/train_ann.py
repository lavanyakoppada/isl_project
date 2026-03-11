import argparse
import csv
import json
import math
import os
import random
import numpy as np
import tensorflow as tf

def load_csv(path):
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [r for r in reader if len(r) == len(header)]
    labels = [r[0] for r in rows]
    origins = [r[1] for r in rows]
    feats = [[float(x) for x in r[2:]] for r in rows] # Start from index 2 because of origin_file
    X = np.asarray(feats, dtype=np.float32)
    y = np.asarray(labels)
    return X, y, origins, header

def build_label_map(y):
    classes = sorted(list(set(y)))
    label_to_id = {c: i for i, c in enumerate(classes)}
    y_ids = np.array([label_to_id[v] for v in y], dtype=np.int64)
    return label_to_id, y_ids

def grouped_stratified_split(X, y, origins, train_ratio, val_ratio, seed):
    """Splits unique original images into Train/Val sets, then selects all associated rows."""
    rng = random.Random(seed)
    
    # 1. Group indices by Label -> Original Filename
    # Structure: { label_id: { origin_filename: [indices] } }
    grouped_data = {}
    for i, (label, origin) in enumerate(zip(y, origins)):
        label = int(label)
        if label not in grouped_data: grouped_data[label] = {}
        if origin not in grouped_data[label]: grouped_data[label][origin] = []
        grouped_data[label][origin].append(i)
        
    train_idx, val_idx, test_idx = [], [], []
    
    for label_id, origins_map in grouped_data.items():
        unique_origins = list(origins_map.keys())
        rng.shuffle(unique_origins)
        
        n = len(unique_origins)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        
        # Ensure at least 1 image for val if possible
        if n >= 2 and n_val == 0: n_val = 1
        if n_train + n_val > n: n_train = n - n_val
            
        train_origins = unique_origins[:n_train]
        val_origins = unique_origins[n_train:n_train + n_val]
        test_origins = unique_origins[n_train + n_val:]
        
        for o in train_origins: train_idx.extend(origins_map[o])
        for o in val_origins: val_idx.extend(origins_map[o])
        for o in test_origins: test_idx.extend(origins_map[o])
        
    return (
        X[train_idx], y[train_idx],
        X[val_idx], y[val_idx],
        X[test_idx], y[test_idx],
    )

def standardize_fit(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return mean, std

def standardize_apply(X, mean, std):
    return (X - mean) / std

def class_weights(y, num_classes):
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    weights = counts.max() / np.maximum(counts, 1.0)
    return {i: float(w) for i, w in enumerate(weights)}

def oversample_good(X, y, factor=2):
    if X.size == 0 or factor <= 1:
        return X, y
    left = X[:, :63]
    right = X[:, 63:126]
    eps = 1e-8
    good = (np.abs(left).sum(axis=1) > eps) & (np.abs(right).sum(axis=1) > eps)
    idx = np.where(good)[0]
    if idx.size == 0:
        return X, y
    X_dup = np.repeat(X[idx], factor - 1, axis=0)
    y_dup = np.repeat(y[idx], factor - 1, axis=0)
    X_out = np.concatenate([X, X_dup], axis=0)
    y_out = np.concatenate([y, y_dup], axis=0)
    return X_out, y_out

def build_model(input_dim, num_classes, hidden=128, hidden2=64, dropout=0.3, lr=1e-3):
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(hidden, use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(hidden2, use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(num_classes, activation="softmax")
    ])
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

def confusion_matrix(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="hand_landmarks.csv")
    p.add_argument("--outdir", default="models")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--hidden2", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-ratio", type=float, default=0.2)  # Set to 0.2 for 80/20 split
    p.add_argument("--test-ratio", type=float, default=0.0)  # Use all remaining for validation
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--oversample-good", action="store_true")
    p.add_argument("--oversample-factor", type=int, default=2)
    p.add_argument("--export-tflite", action="store_true", help="Export TFLite after training")
    p.add_argument("--export-tflite-only", action="store_true", help="Only export TFLite from existing model")
    args = p.parse_args()

    if args.export_tflite_only:
        os.makedirs(args.outdir, exist_ok=True)
        model_path = os.path.join(args.outdir, "ann_landmarks.keras")
        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}. Train first or adjust --outdir.")
            return
        model = tf.keras.models.load_model(model_path)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        tflite_path = os.path.join(args.outdir, "ann_landmarks.tflite")
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)
        print(f"Exported TFLite model to {tflite_path}")
        return

    X, y_str, origins, header = load_csv(args.csv)
    label_to_id, y = build_label_map(y_str)
    num_classes = len(label_to_id)
    
    train_ratio = 1.0 - args.val_ratio
    Xtr, ytr, Xv, yv, Xte, yte = grouped_stratified_split(X, y, origins, train_ratio, args.val_ratio, args.seed)
    
    mean, std = standardize_fit(Xtr)
    Xtr = standardize_apply(Xtr, mean, std)
    Xv = standardize_apply(Xv, mean, std)
    
    if args.oversample_good:
        Xtr, ytr = oversample_good(Xtr, ytr, args.oversample_factor)
    
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(X.shape[1],)),
        tf.keras.layers.Dense(args.hidden, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.Dropout(args.dropout),
        tf.keras.layers.Dense(args.hidden2, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.Dropout(args.dropout / 2),
        tf.keras.layers.Dense(num_classes, activation="softmax")
    ])
    
    opt = tf.keras.optimizers.Adam(learning_rate=args.lr)
    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    
    cw = class_weights(ytr, num_classes)
    es = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=15, restore_best_weights=True)
    rlrop = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6)
    
    print(f"Starting training on {Xtr.shape[0]} samples, validating on {Xv.shape[0]} samples...")
    hist = model.fit(
        Xtr, ytr,
        validation_data=(Xv, yv),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[es, rlrop],
        class_weight=cw,
        verbose=1
    )

    v_loss, v_acc = model.evaluate(Xv, yv, verbose=0)
    print(f"\nTraining Complete. Final Validation Accuracy: {v_acc:.4f}")

    os.makedirs(args.outdir, exist_ok=True)
    model_path = os.path.join(args.outdir, "ann_landmarks.keras")
    model.save(model_path)
    
    with open(os.path.join(args.outdir, "label_map.json"), "w") as f:
        json.dump(label_to_id, f, indent=2)
    
    np.savez(os.path.join(args.outdir, "normalization.npz"), mean=mean, std=std)
    
    metrics = {
        "val_accuracy": float(v_acc),
        "val_loss": float(v_loss),
        "history": {k: [float(x) for x in v] for k, v in hist.history.items()}
    }
    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Saved model to {model_path}")
    
    if args.export_tflite:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        tflite_path = os.path.join(args.outdir, "ann_landmarks.tflite")
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)
        print(f"Exported TFLite model to {tflite_path}")

if __name__ == "__main__":
    main()

