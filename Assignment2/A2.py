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

# ==========================================
# Neural Network Wrapper
# ==========================================
class NeuralNetwork:
    def __init__(self, input_dim: int):
        self.input_dim = input_dim
        self.model = None
        self.history = None

    def build_model(self, hidden_layers=[64, 32, 16], dropout_rate=0.3, learning_rate=0.001):
        model = models.Sequential()
        model.add(layers.Input(shape=(self.input_dim,)))
        for units in hidden_layers:
            model.add(layers.Dense(units, activation="relu"))
            model.add(layers.Dropout(dropout_rate))
        model.add(layers.Dense(1, activation="sigmoid"))
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss="mse",
            metrics=["mae"]
        )
        self.model = model

    def prepare_data(self, features_file="team_features.csv", test_size=0.2):
        df = pd.read_csv(features_file)
        X = df.drop(columns=["team_score"]).values
        y = df["team_score"].values
        from sklearn.model_selection import train_test_split
        return train_test_split(X, y, test_size=test_size, random_state=42)

    def train_model(self, X_train, y_train, X_test, y_test, epochs=30, batch_size=64, patience=10):
        es = callbacks.EarlyStopping(patience=patience, restore_best_weights=True)
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[es],
            verbose=1
        )

    def predict_team_score(self, features: List[float]) -> float:
        features = np.array(features).reshape(1, -1)
        return float(self.model.predict(features, verbose=0)[0][0])

    def save_model(self, path="basketball_team_model.keras"):
        self.model.save(path)

    def load_model(self, path="basketball_team_model.keras"):
        self.model = keras.models.load_model(path)

    def plot_training_history(self):
        if not self.history:
            return
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot(self.history.history["loss"], label="Train Loss")
        ax[0].plot(self.history.history["val_loss"], label="Val Loss")
        ax[0].legend()
        ax[0].set_title("Loss")
        ax[1].plot(self.history.history["mae"], label="Train MAE")
        ax[1].plot(self.history.history["val_mae"], label="Val MAE")
        ax[1].legend()
        ax[1].set_title("MAE")
        st.pyplot(fig)

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
def find_optimal_team(csv_file="relevant_data.csv", model=None, optional_max_teams=500):
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
        nn.plot_training_history()

    best_team, score = find_optimal_team("relevant_data.csv", model=nn)
    st.subheader("Best Team Found")
    st.write(f"Predicted Score: {score:.4f}/1")
    st.dataframe(best_team)
