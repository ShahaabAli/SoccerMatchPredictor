import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from data_loader import load_data  # Assumes `data_loader.py` contains the `load_data` function
from tensorflow.keras.regularizers import l2

# Define the neural network model
def create_model(input_dim):
    model = Sequential([
        Dense(512, activation='relu', input_shape=(input_dim,), kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(256, activation='relu', kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation='relu', kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu', kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(3)  # Output layer for three classes
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    return model

# Preprocess data: feature engineering, scaling, and splitting
def preprocess_data():
    df = load_data()
    df['stat_diff'] = df['home_buildUpPlayPassing'] - df['away_buildUpPlayPassing']
    df['chance_creation_diff'] = df['home_chanceCreationPassing'] - df['away_chanceCreationPassing']
    df['goal_difference'] = df['home_avg_goal_diff'] - df['away_avg_goal_diff']
    df['win_percentage_diff'] = df['home_win_percentage'] - df['away_win_percentage']
    df = df[df['result'].isin([0, 1, 2])]  # Ensure valid target classes
    X = df.drop(columns=['result'])
    y = df['result']
    X_scaled = StandardScaler().fit_transform(X)
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

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
def save_confusion_matrix(cm, labels, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap='viridis')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()

# Train and evaluate the model
def train_model(output_dir="plots"):
    X_train, X_test, y_train, y_test = preprocess_data()

    plot_class_distribution(y_train, output_dir)

    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}

    model = create_model(X_train.shape[1])
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    ]

    history = model.fit(
        X_train, y_train,
        epochs=1000,
        batch_size=256,
        validation_split=0.2,
        class_weight=class_weights_dict,
        callbacks=callbacks
    )

    plot_learning_curve(history, output_dir)

    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_accuracy:.2f}")

    y_pred = np.argmax(model.predict(X_test), axis=1)
    cm = confusion_matrix(y_test, y_pred)
    save_confusion_matrix(cm, np.unique(y_test), output_dir)
    plot_metrics(y_test, y_pred, output_dir)

if __name__ == '__main__':
    train_model()