import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from config import ROLLING_WINDOWS, LAG_FEATURES, ANOMALY_THRESHOLD


class FeatureEngineer:
    """Handles advanced feature engineering for time series data"""
    
    def __init__(self):
        pass
    
    def create_rolling_features(self, dataset):
        """Create rolling statistics for historical delay patterns.
        Rolling features help capture trends and patterns over time."""
        print("Computing historical delay patterns...")
        
        def safe_rolling_stats(group, column, windows):
            """Compute rolling statistics for time series data"""
            group = group.sort_values('actualTime')
            result_dict = {}
            
            for window in windows:
                result_dict[f'{column}_rolling_mean_{window}'] = group[column].rolling(
                    window=window, min_periods=1
                ).mean().shift(1)
                
                result_dict[f'{column}_rolling_std_{window}'] = group[column].rolling(
                    window=window, min_periods=1
                ).std().shift(1)
                
                result_dict[f'{column}_rolling_max_{window}'] = group[column].rolling(
                    window=window, min_periods=1
                ).max().shift(1)
            
            return pd.DataFrame(result_dict, index=group.index)
        
        # compute rolling statistics for each route-stop combination
        rolling_features = []
        
        for route_stop in dataset['route_stop_key'].unique():
            mask = dataset['route_stop_key'] == route_stop
            group_data = dataset[mask]
            
            if len(group_data) >= 5:  # only compute for routes with enough data
                rolling_stats = safe_rolling_stats(group_data, 'delay_minutes', ROLLING_WINDOWS)
                rolling_features.append(rolling_stats)
        
        if rolling_features:
            rolling_df = pd.concat(rolling_features)
            dataset = dataset.join(rolling_df, how='left')
            print(f"Added {rolling_df.shape[1]} historical pattern features")
        else:
            print("No rolling features computed")
        
        return dataset
    
    def create_lag_features(self, dataset):
        """Create lag features for recent delay patterns. Lag features help capture temporal dependencies."""
        print("Creating lag features for recent delay patterns...")
        
        def create_lag_features_group(group, column, lags):
            """Create lag features for time series"""
            group = group.sort_values('actualTime')
            result_dict = {}
            
            for lag in lags:
                result_dict[f'{column}_lag_{lag}'] = group[column].shift(lag)
            
            return pd.DataFrame(result_dict, index=group.index)
        
        lag_features = []
        
        for route_stop in dataset['route_stop_key'].unique():
            mask = dataset['route_stop_key'] == route_stop
            group_data = dataset[mask]
            
            if len(group_data) >= 10:
                lag_stats = create_lag_features_group(group_data, 'delay_minutes', LAG_FEATURES)
                lag_features.append(lag_stats)
        
        if lag_features:
            lag_df = pd.concat(lag_features)
            dataset = dataset.join(lag_df, how='left')
            print(f"Added {lag_df.shape[1]} lag features")
        else:
            print("No lag features computed")
        
        return dataset
    
    def create_performance_features(self, dataset):
        """Create route and stop performance statistics"""
        print("Computing route performance statistics...")
        
        route_stats = dataset.groupby('patternText')['delay_minutes'].agg([
            'mean', 'std', 'median', 'count'
        ]).add_prefix('route_')
        
        stop_stats = dataset.groupby('stopName')['delay_minutes'].agg([
            'mean', 'std', 'median', 'count'
        ]).add_prefix('stop_')
        
        direction_stats = dataset.groupby('direction')['delay_minutes'].agg([
            'mean', 'std', 'median', 'count'
        ]).add_prefix('direction_')
        
        hour_stats = dataset.groupby('hour')['delay_minutes'].agg([
            'mean', 'std', 'count'
        ]).add_prefix('hour_')
        
        dataset = dataset.merge(route_stats, left_on='patternText', right_index=True, how='left')
        dataset = dataset.merge(stop_stats, left_on='stopName', right_index=True, how='left')
        dataset = dataset.merge(direction_stats, left_on='direction', right_index=True, how='left')
        dataset = dataset.merge(hour_stats, left_on='hour', right_index=True, how='left')
        
        print(f"Added route performance features")
        return dataset
    
    def create_anomaly_features(self, dataset):
        """Create anomaly detection features. Anomalies help identify unusual delays."""
        print("Creating anomaly detection features...")
        
        def detect_anomalies(series, threshold=ANOMALY_THRESHOLD):
            """Detect anomalies using z-score method"""
            if len(series) < 5:
                return pd.Series([0] * len(series), index=series.index)
            
            z_scores = np.abs((series - series.mean()) / (series.std() + 1e-8))
            return (z_scores > threshold).astype(int)
        
        dataset['delay_anomaly_global'] = detect_anomalies(dataset['delay_minutes'])
        dataset['delay_anomaly_route'] = dataset.groupby('patternText')['delay_minutes'].transform(detect_anomalies)
        dataset['delay_anomaly_stop'] = dataset.groupby('stopName')['delay_minutes'].transform(detect_anomalies)
        dataset['delay_anomaly_hour'] = dataset.groupby('hour')['delay_minutes'].transform(detect_anomalies)
        
        print(f"Added anomaly detection features")
        return dataset
    
    def create_interaction_features(self, dataset):
        """Create interaction features. Interaction features help capture relationships between different variables."""
        print("Creating interaction features...")
        
        dataset['route_hour_interaction'] = dataset['route_mean'] * dataset['hour_sin']
        dataset['stop_weekend_interaction'] = dataset['stop_mean'] * dataset['is_weekend']
        dataset['rush_hour_route_interaction'] = dataset['is_rush_hour'] * dataset['route_mean']
        
        print(f"Added interaction features")
        return dataset
    
    def engineer_all_features(self, dataset):
        """Apply all feature engineering steps"""
        print("Starting comprehensive feature engineering...")
        
        dataset = self.create_rolling_features(dataset)
        dataset = self.create_lag_features(dataset)
        dataset = self.create_performance_features(dataset)
        dataset = self.create_anomaly_features(dataset)
        dataset = self.create_interaction_features(dataset)
        
        print(f"Feature engineering complete! Dataset shape: {dataset.shape}")
        return dataset
