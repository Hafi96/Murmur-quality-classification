#!/usr/bin/env python

import os
import sys
import numpy as np
from helper_code import load_patient_data, get_quality, load_challenge_outputs, compare_strings
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score

# ✅ Function to find label and output files
def find_challenge_files(label_folder, output_folder):
    label_files, output_files = [], []
    for label_file in sorted(os.listdir(label_folder)):
        label_file_path = os.path.join(label_folder, label_file)
        if os.path.isfile(label_file_path) and label_file.lower().endswith('.txt'):
            root, _ = os.path.splitext(label_file)
            output_file_path = os.path.join(output_folder, root + '.csv')
            if os.path.isfile(output_file_path):
                label_files.append(label_file_path)
                output_files.append(output_file_path)
            else:
                print(f"⚠️ Warning: Missing output file for {label_file}")
    return label_files, output_files

# ✅ Function to load quality labels
def load_qualities(label_files):
    valid_indices, labels = [], []
    for i, file in enumerate(label_files):
        data = load_patient_data(file)
        label = get_quality(data)
        if label in ["Blowing","Harsh"]:  
            labels.append([int(label == "Blowing"), int(label == "Harsh")])
            valid_indices.append(i)
    return np.array(labels, dtype=int), valid_indices

# ✅ Function to load classifier outputs
def load_classifier_outputs(output_files, valid_indices):
    binary_outputs, scalar_outputs = [], []
    filtered_output_files = [output_files[i] for i in valid_indices]
    for file in filtered_output_files:
        _, patient_classes, _, patient_scalar_outputs = load_challenge_outputs(file)
        binary_output, scalar_output = [0, 0], [0.0, 0.0]  # Default
        for j, x in enumerate(["Blowing","Harsh"]):
            for k, y in enumerate(patient_classes):
                if compare_strings(x, y):
                    scalar_output[j] = patient_scalar_outputs[k]
                    binary_output[j] = int(patient_scalar_outputs[k] >= 0.5)  # Default threshold
        binary_outputs.append(binary_output)
        scalar_outputs.append(scalar_output)
    return np.array(binary_outputs, dtype=int), np.array(scalar_outputs, dtype=np.float64)

# ✅ Compute the best threshold using F1-score

# ✅ Compute evaluation metrics
def compute_auc(labels, outputs):
    try:
        auroc_blowing = roc_auc_score(labels[:, 0], outputs[:, 0])
        auprc_blowing = average_precision_score(labels[:, 0], outputs[:, 0])
        auroc_harsh = roc_auc_score(labels[:, 1], outputs[:, 1])
        auprc_harsh = average_precision_score(labels[:, 1], outputs[:, 1])
    except ValueError:
        auroc_blowing, auprc_blowing, auroc_harsh, auprc_harsh = 0.5, 0.5, 0.5, 0.5
    return (auroc_blowing, auprc_blowing, auroc_harsh, auprc_harsh)

def compute_f_measure(labels, outputs):
    f1_Blowing = f1_score(labels[:, 0], outputs[:, 0])
    f1_Harsh = f1_score(labels[:, 1], outputs[:, 1])
    return np.mean([f1_Blowing, f1_Harsh]), [f1_Blowing, f1_Harsh]

def compute_accuracy(labels, outputs):
    accuracy_Blowing = accuracy_score(labels[:, 0], outputs[:, 0])
    accuracy_Harsh = accuracy_score(labels[:, 1], outputs[:, 1])
    return np.mean([accuracy_Blowing, accuracy_Harsh]), [accuracy_Blowing, accuracy_Harsh]

def compute_weighted_accuracy(labels, outputs):
    weights = np.array([[5, 1], [5, 1]])
    confusion = np.zeros((2, 2))
    for i in range(len(labels)):
        confusion[np.argmax(outputs[i]), np.argmax(labels[i])] += 1
    weighted_acc = np.trace(weights * confusion) / np.sum(weights * confusion)
    return weighted_acc

# ✅ Main evaluation function
def evaluate_model(label_folder, output_folder):
    print("🔍 Evaluating model...")

    # Load label & output files
    label_files, output_files = find_challenge_files(label_folder, output_folder)
    quality_labels, valid_indices = load_qualities(label_files)
    quality_binary_outputs, quality_scalar_outputs = load_classifier_outputs(output_files, valid_indices)

    # Find best threshold
    threshold = 0.5

    # Apply threshold
    quality_binary_outputs = (quality_scalar_outputs >= threshold).astype(int)

    # Compute evaluation metrics
    auroc_blowing, auprc_blowing, auroc_harsh, auprc_harsh = compute_auc(quality_labels, quality_scalar_outputs)
    quality_f_measure, quality_f_measure_classes = compute_f_measure(quality_labels, quality_binary_outputs)
    quality_accuracy, quality_accuracy_classes = compute_accuracy(quality_labels, quality_binary_outputs)
    quality_weighted_accuracy = compute_weighted_accuracy(quality_labels, quality_binary_outputs)

    return ["Blowing","Harsh"], [auroc_blowing, auroc_harsh], [auprc_blowing, auprc_harsh], \
           quality_f_measure, quality_f_measure_classes, quality_accuracy, quality_accuracy_classes, quality_weighted_accuracy

# ✅ Print & Save scores
def print_and_save_scores(filename, quality_scores):
    classes, auroc, auprc, f_measure, f_measure_classes, accuracy, accuracy_classes, weighted_accuracy = quality_scores
    total_auroc= np.mean([auroc[0],auroc[1]])
    total_auprc = np.mean([auprc[0], auprc[1]])
    output_string = f"""
#Quality scores
AUROC,AUPRC,F-measure,Accuracy,Weighted Accuracy
{total_auroc:.3f},{total_auprc:.3f},{f_measure:.3f},{accuracy:.3f},{weighted_accuracy:.3f}

#Quality  scores (per class)
Classes,Blowing,Harsh
AUROC,{auroc[0]:.3f},{auroc[1]:.3f}
AUPRC,{auprc[0]:.3f},{auprc[1]:.3f}
F-measure,{f_measure_classes[0]:.3f},{f_measure_classes[1]:.3f}
Accuracy,{accuracy_classes[0]:.3f},{accuracy_classes[1]:.3f}
"""

    # ✅ Print results to console
    print(output_string)

    # ✅ Save to file
    with open(filename, 'w') as f:
        f.write(output_string.strip())
    print(f"✅ Scores saved to {filename}")

# ✅ Run the evaluation script
if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: python evaluate_model.py <label_folder> <output_folder> <scores.csv>")
        sys.exit(1)

    quality_scores = evaluate_model(sys.argv[1], sys.argv[2])
    print_and_save_scores(sys.argv[3], quality_scores)

    print("✅ Model Evaluation Completed. Check scores.csv for detailed results.")
