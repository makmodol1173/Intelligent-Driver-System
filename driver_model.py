import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class DriverModel:
    def __init__(self):
        self.regression_model = LinearRegression()
        self.classification_model = LogisticRegression(max_iter=1000)
        self.clustering_model = KMeans(n_clusters=3, random_state=42, n_init=10)
        self.scaler = StandardScaler()
        self.features_trained_on = []

    def _prepare_features(self, df):
        required_cols = ['speed_avg', 'braking_intensity', 'jerk', 'ear', 'blink_rate']
        for col in required_cols:
            if col not in df.columns:
                if col == 'ear':
                    df[col] = np.random.rand(len(df)) * 0.4 + 0.1
                elif col == 'blink_rate':
                    df[col] = np.random.randint(0, 10, len(df))
                else:
                    df[col] = np.random.rand(len(df))

        numerical_features = ['speed_avg', 'braking_intensity', 'jerk', 'ear', 'blink_rate']
        available_features = [f for f in numerical_features if f in df.columns]

        if not available_features:
            raise ValueError("No numerical features available for model training/prediction.")

        X = df[available_features].copy()
        self.features_trained_on = available_features

        if not hasattr(self.scaler, 'scale_') or len(self.scaler.scale_) != len(available_features):
            self.scaler.fit(X)

        X_scaled = self.scaler.transform(X)
        return X_scaled, X.columns.tolist()

    def train_models(self, data):
        if data is None or data.empty:
            print("No data provided for training models.")
            return

        try:
            X_scaled, self.features_trained_on = self._prepare_features(data)
            X = pd.DataFrame(X_scaled, columns=self.features_trained_on)

            if 'risk_level' not in data.columns:
                data['risk_level'] = np.random.randint(0, 3, size=len(data))

            y_score = 100 - (data['risk_level'] * 30 + np.random.rand(len(data)) * 10)
            y_score = np.clip(y_score, 0, 100)
            y_risk = data['risk_level']

            self.regression_model.fit(X, y_score)
            print("Regression model trained.")

            self.classification_model.fit(X, y_risk)
            print("Classification model trained.")

            self.clustering_model.fit(X)
            print("Clustering model trained.")
        except Exception as e:
            print(f"Error during model training: {e}")
    def predict_score(self, features):
        if self.regression_model is None or not self.features_trained_on:
            print("Regression model not trained or features not set.")
            return 50.0

        if isinstance(features, dict):
            features_df = pd.DataFrame([features])
        else:
            features_df = features.copy()

        for col in self.features_trained_on:
            if col not in features_df.columns:
                features_df[col] = 0.0

        X_pred = features_df[self.features_trained_on]
        X_pred_scaled = self.scaler.transform(X_pred)

        score = self.regression_model.predict(X_pred_scaled)[0]
        return np.clip(score, 0, 100)

    def classify_risk(self, features):
        if self.classification_model is None or not self.features_trained_on:
            print("Classification model not trained or features not set.")
            return 1

        if isinstance(features, dict):
            features_df = pd.DataFrame([features])
        else:
            features_df = features.copy()

        for col in self.features_trained_on:
            if col not in features_df.columns:
                features_df[col] = 0.0

        X_pred = features_df[self.features_trained_on]
        X_pred_scaled = self.scaler.transform(X_pred)

        risk = self.classification_model.predict(X_pred_scaled)[0]
        return int(risk)

    def cluster_drivers(self, data):
        if self.clustering_model is None or data is None or data.empty:
            print("Clustering model not trained or no data for clustering.")
            return np.array([])

        try:
            X_scaled, _ = self._prepare_features(data)
            clusters = self.clustering_model.predict(X_scaled)
            return clusters
        except Exception as e:
            print(f"Error during clustering: {e}")
            return np.array([])

    def save_models(self, path="models"):
        joblib.dump(self.regression_model, f"{path}/regression_model.pkl")
        joblib.dump(self.classification_model, f"{path}/classification_model.pkl")
        joblib.dump(self.clustering_model, f"{path}/clustering_model.pkl")
        joblib.dump(self.scaler, f"{path}/scaler.pkl")
        print("Models saved.")

    def load_models(self, path="models"):
        try:
            self.regression_model = joblib.load(f"{path}/regression_model.pkl")
            self.classification_model = joblib.load(f"{path}/classification_model.pkl")
            self.clustering_model = joblib.load(f"{path}/clustering_model.pkl")
            self.scaler = joblib.load(f"{path}/scaler.pkl")
            print("Models loaded.")
            return True
        except FileNotFoundError:
            print("Model files not found. Models need to be trained.")
            return False
        except Exception as e:
            print(f"Error loading models: {e}")
            return False