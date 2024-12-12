import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from data_loader import load_data  # Assuming `data_loader.py` contains the `load_data` function
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l2

# Define the updated neural network model
def create_improved_model(input_dim):
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
        Dense(3)  # Output layer with logits
    ])

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    return model

# Updated preprocess_data function for improved feature engineering
def preprocess_data():
    matches_df = load_data()

    # Feature engineering
    matches_df['stat_diff'] = matches_df['home_buildUpPlayPassing'] - matches_df['away_buildUpPlayPassing']
    matches_df['chance_creation_diff'] = matches_df['home_chanceCreationPassing'] - matches_df['away_chanceCreationPassing']
    matches_df['goal_difference'] = matches_df['home_avg_goal_diff'] - matches_df['away_avg_goal_diff']
    matches_df['win_percentage_diff'] = matches_df['home_win_percentage'] - matches_df['away_win_percentage']

    # Remove rows with invalid target values
    valid_classes = [0, 1, 2]
    matches_df = matches_df[matches_df['result'].isin(valid_classes)]

    # Separate features and target
    X = matches_df.drop(columns=['result'])
    y = matches_df['result']

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Plot learning curve and save the graphs
def plot_learning_curve(history, output_dir="plots"):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot training and validation loss
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Learning Curve (Loss)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(output_dir, "learning_curve_loss.png"))
    plt.close()

    # Plot training and validation accuracy
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Learning Curve (Accuracy)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(output_dir, "learning_curve_accuracy.png"))
    plt.close()

# Plot Class Distribution
def plot_class_distribution(y, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(8, 6))
    classes, counts = np.unique(y, return_counts=True)
    plt.bar(classes, counts, alpha=0.7, color='blue')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Class Distribution')
    plt.xticks(classes)
    plt.savefig(os.path.join(output_dir, "class_distribution.png"))
    plt.close()

from sklearn.metrics import precision_recall_fscore_support

# Plot Precision, Recall, F1-Score
def plot_precision_recall_f1(y_true, y_pred, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
    classes = np.unique(y_true)

    plt.figure(figsize=(12, 6))
    x = np.arange(len(classes))
    width = 0.3  # Bar width

    plt.bar(x - width, precision, width, label='Precision', alpha=0.7)
    plt.bar(x, recall, width, label='Recall', alpha=0.7)
    plt.bar(x + width, f1, width, label='F1-Score', alpha=0.7)

    plt.xlabel('Class')
    plt.ylabel('Score')
    plt.title('Precision, Recall, and F1-Score by Class')
    plt.xticks(x, labels=classes)
    plt.legend()
    plt.savefig(os.path.join(output_dir, "precision_recall_f1.png"))
    plt.close()

# Save confusion matrix
def save_confusion_matrix(cm, class_labels, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    disp.plot(cmap='viridis', xticks_rotation='vertical')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()

def train_improved_model(output_dir="plots"):
    X_train, X_test, y_train, y_test = preprocess_data()

    # Plot Class Distribution
    plot_class_distribution(y_train, output_dir=output_dir)

    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}

    input_dim = X_train.shape[1]
    model = create_improved_model(input_dim)

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

    history = model.fit(
        X_train, y_train,
        epochs=1000,
        batch_size=256,
        validation_split=0.2,
        callbacks=[early_stopping, reduce_lr],
        class_weight=class_weights_dict
    )

    plot_learning_curve(history, output_dir=output_dir)

    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_accuracy:.2f}")

    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Generate and save Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_classes)
    save_confusion_matrix(cm, class_labels=np.unique(y_test), output_dir=output_dir)

    # Plot Precision, Recall, and F1-Score
    plot_precision_recall_f1(y_test, y_pred_classes, output_dir=output_dir)

if __name__ == '__main__':
    train_improved_model()