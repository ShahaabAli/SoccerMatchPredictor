import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tensorflow.keras.regularizers import l2

from data_loader import load_data
from plotting import save_confusion_matrix, plot_metrics, plot_class_distribution, plot_learning_curve


# Neural Network Model
def create_model(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        Dense(32, activation='relu'),
        Dense(3)
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    return model


# Preprocess Data
def preprocess_data():
    df = load_data()
    X = df.drop(columns=['result'])
    y = df['result']
    X_scaled = StandardScaler().fit_transform(X)
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# Train and evaluate the model
def train_model(output_dir="plots"):
    X_train, X_test, y_train, y_test = preprocess_data()

    plot_class_distribution(y_train, output_dir)

    model = create_model(X_train.shape[1])

    # Training
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=512,
        validation_split=0.2,
    )

    plot_learning_curve(history, output_dir)

    # Testing
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_accuracy:.2f}")
    print(f"Test Loss: {test_loss:.2f}")

    # Plotting
    y_pred = np.argmax(model.predict(X_test), axis=1)
    save_confusion_matrix(y_test, y_pred)
    plot_metrics(y_test, y_pred, output_dir)


if __name__ == '__main__':
    train_model()