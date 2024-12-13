from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support
import numpy as np
import matplotlib.pyplot as plt
import os

# Plot and save learning curve
def plot_learning_curve(history, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(output_dir, "loss_curve.png"))
    plt.close()

    plt.figure()
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(output_dir, "accuracy_curve.png"))
    plt.close()

# Plot and save class distribution
def plot_class_distribution(y, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    classes, counts = np.unique(y, return_counts=True)
    plt.bar(classes, counts)
    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.savefig(os.path.join(output_dir, "class_distribution.png"))
    plt.close()

# Plot and save precision, recall, F1 scores
def plot_metrics(y_true, y_pred, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
    classes = np.unique(y_true)
    x = np.arange(len(classes))

    plt.bar(x - 0.2, precision, width=0.2, label='Precision')
    plt.bar(x, recall, width=0.2, label='Recall')
    plt.bar(x + 0.2, f1, width=0.2, label='F1-Score')
    plt.xticks(x, labels=classes)
    plt.legend()
    plt.title('Precision, Recall, F1-Score')
    plt.savefig(os.path.join(output_dir, "metrics.png"))
    plt.close()

# Save confusion matrix
def save_confusion_matrix(y_test, y_pred, output_dir="plots"):
    cm = confusion_matrix(y_test, y_pred)
    os.makedirs(output_dir, exist_ok=True)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_test))
    disp.plot(cmap='viridis')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()