import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd
from data_loader import load_data

# Load the processed data
def preprocess_data():
    team_attributes_df, matches_df = load_data()

    # Encode team names into numerical values using LabelEncoder
    label_encoder = LabelEncoder()
    matches_df['home_team_name'] = label_encoder.fit_transform(matches_df['home_team_name'])
    matches_df['away_team_name'] = label_encoder.fit_transform(matches_df['away_team_name'])

    # Separate features (X) and target (y)
    X = matches_df[['home_team_name', 'away_team_name', 'year']]
    y = matches_df['result']

    # Scale the features for better neural network performance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define the neural network model
def create_model():
    model = Sequential([
        Dense(64, activation='relu', input_shape=(3,)),  # Input layer (3 features)
        Dropout(0.2),  # Dropout for regularization
        Dense(32, activation='relu'),  # Hidden layer
        Dropout(0.2),  # Dropout for regularization
        Dense(16, activation='relu'),  # Hidden layer
        Dense(3, activation='softmax')  # Output layer (3 classes: win, loss, draw)
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

# Main function to train and evaluate the model
def train_model():
    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data()

    # Create the model
    model = create_model()

    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=128, validation_split=0.2)

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_accuracy:.2f}")

if __name__ == '__main__':
    train_model()
