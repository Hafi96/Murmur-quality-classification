#!/usr/bin/env python

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)

from helper_code import load_patient_data, get_quality, load_challenge_outputs, compare_strings

#  Function to find label and output files
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

#  Function to load quality labels
def load_qualities(label_files):
    valid_indices, labels = [], []
    for i, file in enumerate(label_files):
        data = load_patient_data(file)
        label = get_quality(data)
        if label in ["Blowing", "Harsh"]:
            labels.append([int(label == "Blowing"), int(label == "Harsh")])
            valid_indices.append(i)
    return np.array(labels, dtype=int), valid_indices

#  Function to load classifier outputs
def load_classifier_outputs(output_files, valid_indices):
    binary_outputs, scalar_outputs = [], []
    filtered_output_files = [output_files[i] for i in valid_indices]
    for file in filtered_output_files:
        _, patient_classes, _, patient_scalar_outputs = load_challenge_outputs(file)
        binary_output, scalar_output = [0, 0], [0.0, 0.0]
        for j, x in enumerate(["Blowing", "Harsh"]):
            for k, y in enumerate(patient_classes):
                if compare_strings(x, y):
                    scalar_output[j] = patient_scalar_outputs[k]
                    binary_output[j] = int(patient_scalar_outputs[k] >= 0.5)
        binary_outputs.append(binary_output)
        scalar_outputs.append(scalar_output)
    return np.array(binary_outputs, dtype=int), np.array(scalar_outputs, dtype=np.float64)

#  Compute evaluation metrics
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
    f1_blowing = f1_score(labels[:, 0], outputs[:, 0])
    f1_harsh = f1_score(labels[:, 1], outputs[:, 1])
    return np.mean([f1_blowing, f1_harsh]), [f1_blowing, f1_harsh]

def compute_accuracy(labels, outputs):
    acc_blowing = accuracy_score(labels[:, 0], outputs[:, 0])
    acc_harsh = accuracy_score(labels[:, 1], outputs[:, 1])
    return np.mean([acc_blowing, acc_harsh]), [acc_blowing, acc_harsh]

def compute_weighted_accuracy(labels, outputs):
    weights = np.array([[3, 1], [1, 2]])
    confusion = np.zeros((2, 2))
    for i in range(len(labels)):
        confusion[np.argmax(outputs[i]), np.argmax(labels[i])] += 1
    weighted_acc = np.trace(weights * confusion) / np.sum(weights * confusion)
    return weighted_acc

#  Visualization
def generate_visualizations_multiclass(true_onehot, predicted_probs, class_names=["Blowing", "Harsh"], output_dir='plots'):
    os.makedirs(output_dir, exist_ok=True)

    y_true = np.argmax(true_onehot, axis=1)
    y_pred = np.argmax(predicted_probs, axis=1)

    # ROC
    fpr, tpr, _ = roc_curve(true_onehot.ravel(), predicted_probs.ravel())
    plt.figure()
    plt.plot(fpr, tpr, label="Overall ROC")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Overall ROC Curve")
    plt.grid()
    plt.legend()
    plt.savefig(os.path.join(output_dir, "overall_roc.png"))
    plt.close()

    # PR
    precision, recall, _ = precision_recall_curve(true_onehot.ravel(), predicted_probs.ravel())
    plt.figure()
    plt.plot(recall, precision, label="Overall PR")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Overall Precision-Recall Curve")
    plt.grid()
    plt.legend()
    plt.savefig(os.path.join(output_dir, "overall_pr.png"))
    plt.close()

    # Confusion Matrix (Multiclass-style)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix - Multiclass')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='black')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(os.path.join(output_dir, "overall_confusion_matrix_multiclass.png"))
    plt.close()

#  Main evaluation function
def evaluate_model(label_folder, output_folder):
    print("🔍 Evaluating model...")
    label_files, output_files = find_challenge_files(label_folder, output_folder)
    quality_labels, valid_indices = load_qualities(label_files)
    quality_binary_outputs, quality_scalar_outputs = load_classifier_outputs(output_files, valid_indices)

    threshold = 0.5
    quality_binary_outputs = (quality_scalar_outputs >= threshold).astype(int)

    generate_visualizations_multiclass(quality_labels, quality_scalar_outputs)

    auroc_blowing, auprc_blowing, auroc_harsh, auprc_harsh = compute_auc(quality_labels, quality_scalar_outputs)
    quality_f_measure, quality_f_measure_classes = compute_f_measure(quality_labels, quality_binary_outputs)
    quality_accuracy, quality_accuracy_classes = compute_accuracy(quality_labels, quality_binary_outputs)
    quality_weighted_accuracy = compute_weighted_accuracy(quality_labels, quality_binary_outputs)

    return ["Blowing", "Harsh"], [auroc_blowing, auroc_harsh], [auprc_blowing, auprc_harsh], \
           quality_f_measure, quality_f_measure_classes, quality_accuracy, quality_accuracy_classes, quality_weighted_accuracy

#  Save scores
def print_and_save_scores(filename, quality_scores):
    classes, auroc, auprc, f_measure, f_measure_classes, accuracy, accuracy_classes, weighted_accuracy = quality_scores
    total_auroc = np.mean(auroc)
    total_auprc = np.mean(auprc)
    output_string = f"""
# Quality Scores
AUROC,AUPRC,F-measure,Accuracy,Weighted Accuracy
{total_auroc:.3f},{total_auprc:.3f},{f_measure:.3f},{accuracy:.3f},{weighted_accuracy:.3f}

# Per-class Quality Scores
Classes,Blowing,Harsh
AUROC,{auroc[0]:.3f},{auroc[1]:.3f}
AUPRC,{auprc[0]:.3f},{auprc[1]:.3f}
F-measure,{f_measure_classes[0]:.3f},{f_measure_classes[1]:.3f}
Accuracy,{accuracy_classes[0]:.3f},{accuracy_classes[1]:.3f}
"""
    print(output_string)
    with open(filename, 'w') as f:
        f.write(output_string.strip())
    print(f"✅ Scores saved to {filename}")

#  Run the evaluation script
if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: python evaluate_model_quality.py <label_folder> <output_folder> <scores.csv>")
        sys.exit(1)

    quality_scores = evaluate_model(sys.argv[1], sys.argv[2])
    print_and_save_scores(sys.argv[3], quality_scores)

    print(" Model Evaluation Completed. Check scores.csv and plots/ folder for visualizations.")
