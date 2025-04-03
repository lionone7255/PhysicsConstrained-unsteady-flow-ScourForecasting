import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib


class ScourDataProcessor:
    """Handles data loading and preprocessing for scour prediction"""

    def __init__(self, config):
        self.config = config
        self.scaler = MinMaxScaler(feature_range=(-1, 1))

    def load_dataset(self):
        """Load and normalize raw measurement data"""
        raw_data = pd.read_excel(self.config['data_path'])
        return self._normalize_data(raw_data)

    def _normalize_data(self, data):
        """Apply min-max scaling to input features"""
        normalized = self.scaler.fit_transform(data)
        joblib.dump(self.scaler, self.config['scaler_path'])
        return normalized