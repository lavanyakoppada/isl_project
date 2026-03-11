import argparse
import csv
import os
import json
import numpy as np
import tensorflow as tf

def load_csv(path):
    """Load landmarks CSV for evaluation."""
    if not os.path.exists(path):
        print(f"Error: {path} not found.")
        return None, None
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [r for r in reader if len(r) == len(header)]
    labels = [r[0] for r in rows]
    feats = [[float(x) for x in r[2:]] for r in rows]
    X = np.asarray(feats, dtype=np.float32)
    y = np.asarray(labels)
    return X, y

def load_artifacts(models_dir):
    """Load model, label map, and normalization parameters."""
    model_path = os.path.join(models_dir, "ann_landmarks.keras")
    label_map_path = os.path.join(models_dir, "label_map.json")
    norm_path = os.path.join(models_dir, "normalization.npz")
    
    if not all(os.path.exists(p) for p in [model_path, label_map_path, norm_path]):
        print("Error: Missing model artifacts in", models_dir)
        return None, None, None, None
        
    model = tf.keras.models.load_model(model_path)
    with open(label_map_path, "r") as f:
        label_to_id = json.load(f)
    id_to_label = {int(v): k for k, v in label_to_id.items()}
    norm = np.load(norm_path)
    return model, id_to_label, norm["mean"], norm["std"]

def build_label_map_from_id_to_label(id_to_label, y_str):
    """Map string labels to IDs based on the existing label map."""
    label_to_id = {v: int(k) for k, v in id_to_label.items()}
    y_ids = []
    for label in y_str:
        if label in label_to_id:
            y_ids.append(label_to_id[label])
        else:
            # Handle potentially missing labels in the evaluation set if any
            y_ids.append(-1)
    return np.array(y_ids, dtype=np.int64)

def metrics_from_confusion(cm):
    """Calculate accuracy, precision, recall, and F1 from confusion matrix."""
    tp = np.diag(cm)
    support = cm.sum(axis=1)
    pred_count = cm.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.where(support > 0, tp / support, 0.0)
        precision = np.where(pred_count > 0, tp / pred_count, 0.0)
        f1 = np.where((precision + recall) > 0, 2 * precision * recall / (precision + recall), 0.0)
    acc = tp.sum() / max(cm.sum(), 1)
    return acc, precision, recall, f1, support

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="hand_landmarks.csv")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--outdir", default="eval")
    args = ap.parse_args()

    # Load artifacts
    model, id_to_label, mean, std = load_artifacts(args.models_dir)
    if model is None:
        return

    # Load data
    X, y_str = load_csv(args.csv)
    if X is None:
        return
        
    y_ids = build_label_map_from_id_to_label(id_to_label, y_str)
    
    # Standardize
    X = (X - mean) / std
    
    # Predict
    print(f"Running predictions on {len(y_str)} samples...")
    probs = model.predict(X, verbose=1)
    y_pred = np.argmax(probs, axis=1)
    confidences = np.max(probs, axis=1)
    
    # Build confusion matrix
    num_classes = len(id_to_label)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_ids, y_pred):
        if t != -1:
            cm[int(t), int(p)] += 1
    
    acc, prec, rec, f1, sup = metrics_from_confusion(cm)

    os.makedirs(args.outdir, exist_ok=True)
    
    # Save predictions CSV
    preds_path = os.path.join(args.outdir, "predictions_detailed.csv")
    with open(preds_path, "w", encoding="utf-8") as fw:
        fw.write("index,true,pred,conf\n")
        for i, (true_lbl, pred_id, conf) in enumerate(zip(y_str, y_pred, confidences)):
            pred_lbl = id_to_label[pred_id]
            fw.write(f"{i},{true_lbl},{pred_lbl},{float(conf):.4f}\n")
    
    # Save confusion matrix
    np.savetxt(os.path.join(args.outdir, "confusion_matrix.txt"), cm, fmt="%d")
    
    # Save summary
    summary_path = os.path.join(args.outdir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "overall_accuracy": float(acc),
            "num_samples": int(len(y_str))
        }, f, indent=2)
    
    # Save per-class metrics
    per_class_path = os.path.join(args.outdir, "per_class_metrics.txt")
    with open(per_class_path, "w") as f:
        f.write("Label\tPrecision\tRecall\tF1-Score\tSupport\n")
        f.write("-" * 60 + "\n")
        for cid in range(num_classes):
            lbl = id_to_label[cid]
            f.write(f"{lbl}\t{float(prec[cid]):.44f}\t{float(rec[cid]):.4f}\t{float(f1[cid]):.4f}\t{int(sup[cid])}\n")
    
    print(f"\nEvaluation Complete!")
    print(f"Overall Accuracy: {acc:.4f}")
    print(f"Results saved to: {args.outdir}")

if __name__ == "__main__":
    main()
