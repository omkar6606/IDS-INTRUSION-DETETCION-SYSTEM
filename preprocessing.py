import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder


class IDSPreprocessor:
    """Preprocessing for IDS with feature engineering"""

    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.categorical_cols = ["protocol_type", "service", "flag"]
        self.numerical_cols = None
        self.feature_cols = None

    def _ensure_numeric(self, df):
        """
        Force certain columns to numeric to avoid TypeError like
        'can only concatenate str (not "int") to str'.
        """
        df = df.copy()
        numeric_cols = ["src_bytes", "dst_bytes", "count", "duration"]

        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        return df

    def engineer_features(self, df):
        """Create engineered features (byte_ratio, packet_rate)"""
        df = self._ensure_numeric(df)

        # Avoid division issues with +1
        df["byte_ratio"] = df["src_bytes"] / (df["dst_bytes"] + 1)
        df["packet_rate"] = df["count"] / (df["duration"] + 1)

        return df

    def fit(self, X, y=None):
        """Fit encoders and scaler"""
        start_time = time.time()

        X = self.engineer_features(X)
        self.feature_cols = X.columns.tolist()

        # Fit label encoders for categorical columns
        for col in self.categorical_cols:
            le = LabelEncoder()
            le.fit(X[col].astype(str))
            self.label_encoders[col] = le

        # Encode categorical columns
        X_encoded = X.copy()
        for col in self.categorical_cols:
            X_encoded[col] = self.label_encoders[col].transform(X[col].astype(str))

        # Fit scaler on full encoded features
        self.scaler.fit(X_encoded)

        # Store numerical columns
        self.numerical_cols = [
            col for col in X_encoded.columns if col not in self.categorical_cols
        ]

        fit_time = time.time() - start_time
        print(f"  Preprocessing fit time: {fit_time:.3f}s")
        return self

    def transform(self, X):
        """Transform data using fitted encoders & scaler"""
        start_time = time.time()

        X = self.engineer_features(X)

        # Ensure same feature set/order as during fit
        for col in self.feature_cols:
            if col not in X.columns:
                X[col] = 0

        X = X[self.feature_cols]

        # Transform categorical columns
        X_encoded = X.copy()
        for col in self.categorical_cols:
            le = self.label_encoders[col]

            def map_value(v):
                v = str(v)
                if v in le.classes_:
                    return le.transform([v])[0]
                else:
                    # unseen category -> -1
                    return -1

            X_encoded[col] = X[col].map(map_value)

        # Scale all features
        X_scaled = self.scaler.transform(X_encoded)

        transform_time = time.time() - start_time
        latency_per_sample = (transform_time / len(X)) * 1000  # ms/sample

        return X_scaled, transform_time, latency_per_sample

    def fit_transform(self, X, y=None):
        """Fit + transform in one go"""
        self.fit(X, y)
        X_scaled, _, _ = self.transform(X)
        return X_scaled
