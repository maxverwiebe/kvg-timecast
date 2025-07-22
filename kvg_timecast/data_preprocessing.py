import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

from config import MAX_DELAY_SECONDS, CATEGORICAL_COLS, EXCLUDE_COLS, CLEAN_DUPLICATED_TRIPS_AT_STOP


class DataPreprocessor:
    """Handles all data preprocessing tasks"""
    
    def __init__(self):
        self.label_encoders = {}
        
    def load_and_clean_data(self, file_path, max_records=None):
        """Load and perform basic cleaning on the dataset"""
        print("Loading and preprocessing data...")
        
        dataset = pd.read_csv(file_path)
        if max_records:
            dataset = dataset.head(max_records)
        print(f"Initial dataset: {dataset.shape[0]:,} records")
        
        dataset['timestamp'] = pd.to_datetime(dataset['timestamp'])
        dataset['actualTime'] = pd.to_datetime(
            dataset['timestamp'].dt.date.astype(str) + ' ' + dataset['actualTime'], 
            errors='coerce'
        )
        dataset['plannedTime'] = pd.to_datetime(
            dataset['timestamp'].dt.date.astype(str) + ' ' + dataset['plannedTime'], 
            errors='coerce'
        )
        
        dataset['delay_seconds'] = (dataset['actualTime'] - dataset['plannedTime']).dt.total_seconds()
        dataset['delay_minutes'] = dataset['delay_seconds'] / 60
        
        if CLEAN_DUPLICATED_TRIPS_AT_STOP:
            print("Dropping duplicated trips at stop...")
            dataset = dataset.sort_values('actualTime')
            dataset = dataset.drop_duplicates(subset=['busStopID', 'tripId', 'vehicleId'], keep='last')

        dataset = dataset[
            (dataset['delay_seconds'].abs() < MAX_DELAY_SECONDS) &
            (dataset['actualTime'].notna()) & 
            (dataset['plannedTime'].notna())
        ].copy()
        
        print(f"After basic cleaning: {dataset.shape[0]:,} records")
        return dataset
    
    def create_temporal_features(self, dataset):
        """Create temporal features from datetime columns"""
        print("Creating temporal features...")
        
        dataset = dataset.sort_values(['actualTime']).reset_index(drop=True)
        
        dataset['hour'] = dataset['actualTime'].dt.hour
        dataset['minute'] = dataset['actualTime'].dt.minute
        dataset['weekday'] = dataset['actualTime'].dt.dayofweek
        dataset['month'] = dataset['actualTime'].dt.month
        dataset['day_of_month'] = dataset['actualTime'].dt.day
        dataset['week_of_year'] = dataset['actualTime'].dt.isocalendar().week
        dataset['is_weekend'] = (dataset['weekday'] >= 5).astype(int)
        dataset['is_rush_hour'] = (
            (dataset['hour'].between(7, 9)) | 
            (dataset['hour'].between(16, 18))
        ).astype(int)
        dataset['is_late_night'] = (
            dataset['hour'].between(22, 24) | 
            dataset['hour'].between(0, 5)
        ).astype(int)
        
        dataset['hour_sin'] = np.sin(2 * np.pi * dataset['hour'] / 24)
        dataset['hour_cos'] = np.cos(2 * np.pi * dataset['hour'] / 24)
        dataset['minute_sin'] = np.sin(2 * np.pi * dataset['minute'] / 60)
        dataset['minute_cos'] = np.cos(2 * np.pi * dataset['minute'] / 60)
        dataset['weekday_sin'] = np.sin(2 * np.pi * dataset['weekday'] / 7)
        dataset['weekday_cos'] = np.cos(2 * np.pi * dataset['weekday'] / 7)
        dataset['month_sin'] = np.sin(2 * np.pi * dataset['month'] / 12)
        dataset['month_cos'] = np.cos(2 * np.pi * dataset['month'] / 12)
        
        cyclical_features = len([col for col in dataset.columns if col.endswith(('_sin', '_cos'))])
        print(f"Created {cyclical_features} cyclical features")
        
        return dataset
    
    def create_route_identifiers(self, dataset):
        """Create unique identifiers for route patterns"""
        dataset['route_stop_key'] = (
            dataset['patternText'].astype(str) + '_' + dataset['stopName'].astype(str)
        )
        dataset['route_dir_key'] = (
            dataset['patternText'].astype(str) + '_' + dataset['direction'].astype(str)
        )
        return dataset
    
    def encode_categorical_features(self, dataset, fit=True):
        """Encode categorical variables"""
        print("Encoding categorical features...")
        
        for col in CATEGORICAL_COLS:
            if col in dataset.columns:
                if fit:
                    le = LabelEncoder()
                    dataset[col] = le.fit_transform(dataset[col].astype(str))
                    self.label_encoders[col] = le
                else:
                    if col in self.label_encoders:
                        le = self.label_encoders[col]
                        mask = dataset[col].astype(str).isin(le.classes_)
                        dataset.loc[mask, col] = le.transform(dataset.loc[mask, col].astype(str))
                        dataset.loc[~mask, col] = -1  # unknown category
                    else:
                        dataset[col] = -1
        
        return dataset
    
    def prepare_feature_matrix(self, dataset):
        """Prepare final feature matrix and target"""
        print("Preparing feature matrix...")
        
        dataset_clean = dataset.dropna(subset=['delay_minutes']).copy()
        print(f"After removing NaN targets: {dataset_clean.shape[0]:,} records")
        
        feature_cols = [col for col in dataset_clean.columns if col not in EXCLUDE_COLS]
        
        X = dataset_clean[feature_cols].copy()
        y = dataset_clean['delay_minutes'].copy()
        
        for col in X.columns:
            if X[col].dtype in ['float64', 'int64']:
                X[col] = X[col].fillna(X[col].median())
            else:
                mode_val = X[col].mode()
                X[col] = X[col].fillna(mode_val[0] if len(mode_val) > 0 else 0)
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Features: {len(feature_cols)} total features")
        
        return X, y, feature_cols
    
    def get_encoders(self):
        """Get fitted label encoders"""
        return self.label_encoders
    
    def set_encoders(self, encoders):
        """Set label encoders (for loading saved encoders)"""
        self.label_encoders = encoders
