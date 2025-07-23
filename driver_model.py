import pandas as pd
import numpy as np
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
