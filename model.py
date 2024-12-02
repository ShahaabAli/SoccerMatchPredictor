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

# Load and preprocess the data
def preprocess_data():
    # Load the processed data
    matches_df = load_data()

    # Feature engineering
    matches_df['stat_diff'] = matches_df['home_buildUpPlayPassing'] - matches_df['away_buildUpPlayPassing']
    matches_df['chance_creation_diff'] = matches_df['home_chanceCreationPassing'] - matches_df['away_chanceCreationPassing']

    # Remove rows with invalid target values
    valid_classes = [0, 1, 2]  # Define the valid classes
    matches_df = matches_df[matches_df['result'].isin(valid_classes)]

    # Separate features (X) and target (y)
    X = matches_df.drop(columns=['result'])  # Drop the target column
    y = matches_df['result']  # Target column

    # Scale the features for better neural network performance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define the neural network model
def create_model(input_dim):
    model = Sequential([
        Dense(512, activation='relu', input_shape=(input_dim,)),
        Dropout(0.2),
        Dense(256, activation='relu'),
        Dropout(0.2),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(3)  # Output layer without activation
    ])

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    return model

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

# Save confusion matrix
def save_confusion_matrix(cm, class_labels, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    disp.plot(cmap='viridis', xticks_rotation='vertical')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()

# Train the model
def train_model(output_dir="plots"):
    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data()

    # Compute class weights to handle imbalanced data
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}

    # Create the model
    input_dim = X_train.shape[1]
    model = create_model(input_dim)

    # Add callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

    # Train the model and save the history
    history = model.fit(
        X_train, y_train,
        epochs=1000,
        batch_size=512,
        validation_split=0.2,
        callbacks=[early_stopping, reduce_lr],
        class_weight=class_weights_dict
    )

    # Save learning curves
    plot_learning_curve(history, output_dir=output_dir)

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_accuracy:.2f}")

    # Predict on test data
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Generate and save confusion matrix
    cm = confusion_matrix(y_test, y_pred_classes)
    save_confusion_matrix(cm, class_labels=np.unique(y_test), output_dir=output_dir)

if __name__ == '__main__':
    train_model()