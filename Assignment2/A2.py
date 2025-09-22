# streamlit_app.py
import os
import random
import numpy as np
import pandas as pd
import streamlit as st
from itertools import combinations
from typing import List, Dict
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import pickle


script_dir = os.path.dirname(os.path.abspath(__file__))

# ==========================================
# Neural Network Wrapper
# ==========================================
class NeuralNetwork:
    """
    Neural network for predicting basketball team quality scores.
    """
    
    def __init__(self, input_dim=21):
        self.input_dim = input_dim
        self.model = None
        self.scaler = StandardScaler()
        self.history = None
        
    def build_model(self, hidden_layers=[64, 32, 16], dropout_rate=0.3, learning_rate=0.001):
        """
        Build the neural network architecture.
        
        Args:
            hidden_layers: List of neurons in each hidden layer
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
        """
        
        model = keras.Sequential()
        
        # Input layer
        model.add(layers.Input(shape=(self.input_dim,)))
        
        # Hidden layers with dropout for regularization
        for i, units in enumerate(hidden_layers):
            model.add(layers.Dense(units, activation='relu', name=f'hidden_{i+1}'))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(dropout_rate))
        
        # Output layer (single neuron for team score prediction)
        model.add(layers.Dense(1, activation='sigmoid', name='output'))
        
        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        self.model = model
        print("Neural Network Architecture:")
        print(self.model.summary())
        return model
    
    def prepare_data(self, features_file="team_features.csv", test_size=0.2, random_state=42):
        """
        Load and prepare data for training.
        
        Args:
            features_file: Path to team features CSV
            test_size: Fraction of data for testing
            random_state: Random seed for reproducible splits
        """
        
        # Load features
        df = pd.read_csv(os.path.join(script_dir, features_file))
        print(f"Loaded {len(df)} team examples")
        
        # Separate features and target
        feature_columns = [col for col in df.columns if col != 'team_score']
        X = df[feature_columns].values
        y = df['team_score'].values
        
        print(f"Features shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        print(f"Score range: {y.min():.3f} - {y.max():.3f}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.feature_columns = feature_columns
        
        print(f"Training samples: {len(X_train_scaled)}")
        print(f"Testing samples: {len(X_test_scaled)}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_model(self, X_train, y_train, X_test, y_test, 
                   epochs=30, batch_size=32, validation_split=0.2, patience=10):
        """
        Train the neural network.
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data  
            epochs: Maximum training epochs
            batch_size: Training batch size
            validation_split: Fraction for validation during training
            patience: Early stopping patience
        """
        
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Train model
        print(f"Training neural network...")
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate on test set
        test_loss, test_mae, test_mse = self.model.evaluate(X_test, y_test, verbose=0)
        
        # Make predictions
        y_pred = self.model.predict(X_test, verbose=0).flatten()
        r2 = r2_score(y_test, y_pred)
        
        print(f"\nFinal Test Results:")
        print(f"  Test Loss (MSE): {test_loss:.4f}")
        print(f"  Test MAE: {test_mae:.4f}")
        print(f"  Test R²: {r2:.4f}")
        
        return self.history
    
    def plot_training_history(self, save_plot=True):
        """Plot training history."""
        if self.history is None:
            print("No training history available.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(self.history.history['loss'], label='Training Loss', alpha=0.8)
        ax1.plot(self.history.history['val_loss'], label='Validation Loss', alpha=0.8)
        ax1.set_title('Model Loss During Training')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss (MSE)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # MAE plot
        ax2.plot(self.history.history['mae'], label='Training MAE', alpha=0.8)
        ax2.plot(self.history.history['val_mae'], label='Validation MAE', alpha=0.8)
        ax2.set_title('Model MAE During Training')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Mean Absolute Error')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig(os.path.join(script_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
            print("Training history plot saved as 'training_history.png'")
        
        plt.show()
    
    def save_model(self, model_path="basketball_team_model.keras", scaler_path="feature_scaler.pkl"):
        """Save trained model and scaler."""
        if self.model is None:
            print("No model to save.")
            return
        
        # Save model
        model_full_path = os.path.join(script_dir, model_path)
        self.model.save(model_full_path)
        
        # Save scaler
        scaler_full_path = os.path.join(script_dir, scaler_path)
        with open(scaler_full_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        print(f"Model saved to: {model_full_path}")
        print(f"Scaler saved to: {scaler_full_path}")
    
    def load_model(self, model_path="basketball_team_model.keras", scaler_path="feature_scaler.pkl"):
        """Load trained model and scaler."""
        
        # Load model
        model_full_path = os.path.join(script_dir, model_path)
        self.model = keras.models.load_model(model_full_path)
        
        # Load scaler
        scaler_full_path = os.path.join(script_dir, scaler_path)
        with open(scaler_full_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        print(f"Model loaded from: {model_full_path}")
        print(f"Scaler loaded from: {scaler_full_path}")
    
    def predict_team_score(self, team_features):
        """
        Predict team score from features.
        
        Args:
            team_features: List or array of team features
        
        Returns:
            Predicted team score (0-1)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() or train a model first.")
        
        # Ensure correct shape
        features = np.array(team_features).reshape(1, -1)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Predict
        prediction = self.model.predict(features_scaled, verbose=0)[0, 0]
        
        return prediction

# ==========================================
# Basketball Team Evaluator
# ==========================================
class BasketballTeamEvaluator:
    def __init__(self):
        self.weights = {
            "balance": 0.15,
            "strength": 0.15,
            "complementary": 0.70,
        }

    def evaluate_team(self, team_df: pd.DataFrame) -> Dict:
        balance_score = self._calculate_balance_score(team_df)
        strength_score = self._calculate_strength_score(team_df)
        complementary_score = self._calculate_complementary_score(team_df)

        total_score = (
            balance_score * self.weights["balance"]
            + strength_score * self.weights["strength"]
            + complementary_score * self.weights["complementary"]
        )

        features = self._generate_team_features(team_df)

        return {
            "total_score": total_score,
            "balance_score": balance_score,
            "strength_score": strength_score,
            "complementary_score": complementary_score,
            "features": features,
            "players": team_df["player_name"].tolist(),
        }

    def _calculate_balance_score(self, team_df):
        score = 0.0
        height_std = np.std(team_df["player_height"].values)
        score += min(height_std / 15.0, 1.0) * 0.4
        weight_std = np.std(team_df["player_weight"].values)
        score += min(weight_std / 20.0, 1.0) * 0.4
        gp_score = np.mean(team_df["gp"]) / 82.0
        score += gp_score * 0.2
        return score

    def _calculate_strength_score(self, team_df):
        score = 0.0
        avg_ts = team_df["ts_pct"].fillna(0.5).mean()
        score += min(max((avg_ts - 0.45) / 0.15, 0), 1) * 0.1
        score += min(team_df["pts"].mean() / 20.0, 1.0) * 0.3
        score += min(team_df["reb"].fillna(0).mean() / 10.0, 1.0) * 0.15
        score += min(team_df["oreb_pct"].fillna(0).mean() / 0.3, 1.0) * 0.15
        score += min(team_df["dreb_pct"].fillna(0).mean() / 0.8, 1.0) * 0.15
        avg_net = team_df["net_rating"].fillna(0).mean()
        score += min(max((avg_net + 10) / 20, 0), 1) * 0.15
        return score

    def _calculate_complementary_score(self, team_df):
        score = 0.0
        avg_usage = np.mean(team_df["usg_pct"].fillna(0.15).values)
        score += (1.0 - min(abs(avg_usage - 0.22) / 0.1, 1.0)) * 0.4
        playmakers_pct = sum(team_df["ast_pct"].fillna(0.1) > 0.2)
        score += min(playmakers_pct / 2.0, 1.0) * 0.3
        ast_score = min(team_df["ast"].fillna(0).sum() / 40.0, 1.0)
        score += ast_score * 0.3
        return score

    def _generate_team_features(self, team_df) -> List[float]:
        features = []
        features.extend([
            team_df["pts"].mean(),
            team_df["reb"].mean(),
            team_df["ast"].mean(),
            team_df["ts_pct"].fillna(0.5).mean(),
            team_df["net_rating"].fillna(0).mean(),
            team_df["usg_pct"].fillna(0.15).mean(),
            team_df["ast_pct"].fillna(0.1).mean(),
        ])
        features.extend([
            team_df["pts"].std(),
            team_df["reb"].std(),
            team_df["player_height"].std(),
            team_df["age"].std(),
            team_df["usg_pct"].fillna(0.15).std(),
        ])
        features.extend([
            team_df["player_height"].max() - team_df["player_height"].min(),
            team_df["age"].max() - team_df["age"].min(),
            sum(team_df["pts"] > 15),
            sum(team_df["ast"] > 4),
            sum(team_df["reb"] > 7),
            sum(team_df["ts_pct"].fillna(0.4) > 0.55),
        ])
        features.extend([
            team_df["player_height"].mean(),
            team_df["player_weight"].mean(),
            team_df["age"].mean(),
        ])
        return features

# ==========================================
# Optimal Team Finder
# ==========================================
def find_optimal_team(csv_file="Assignment2/relevant_data.csv", model=None, optional_max_teams=500):
    df = pd.read_csv(csv_file)
    evaluator = BasketballTeamEvaluator()

    best_score, best_team, best_features = 0, None, None
    total_combinations = len(list(combinations(range(len(df)), 5)))
    st.write(f"Total possible teams: {total_combinations:,}")

    if total_combinations > 100000:
        st.write(f"Too many combinations! Sampling {optional_max_teams} random teams...")
        team_combinations = [random.sample(range(len(df)), 5) for _ in range(optional_max_teams)]
    else:
        team_combinations = list(combinations(range(len(df)), 5))

    for i, team_indices in enumerate(team_combinations):
        team_df = df.iloc[list(team_indices)]
        features = evaluator._generate_team_features(team_df)
        predicted_score = model.predict_team_score(features)
        if predicted_score > best_score:
            best_score, best_team, best_features = predicted_score, team_df.copy(), features
    return best_team, best_score

# ==========================================
# Streamlit UI
# ==========================================
st.title("🏀 Optimal Basketball Team Finder")

if st.button("Find Best Team"):
    MODEL_PATH = "basketball_team_model.keras"
    nn = NeuralNetwork(input_dim=21)

    if os.path.exists(MODEL_PATH):
        st.write("Loading existing model...")
        nn.load_model(MODEL_PATH)
    else:
        st.write("Building and training new model...")
        nn.build_model()
        X_train, X_test, y_train, y_test = nn.prepare_data("team_features.csv")
        nn.train_model(X_train, y_train, X_test, y_test, epochs=30, batch_size=64, patience=10)
        nn.save_model(MODEL_PATH)

    best_team, score = find_optimal_team("Assignment2/relevant_data.csv", model=nn)
    st.subheader("Best Team Found")
    st.write(f"Predicted Score: {score:.4f}/1")
    st.dataframe(best_team)
