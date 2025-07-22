import numpy as np
import pandas as pd
from typing import Dict, List, Optional


class KVGPredictor:
    """Advanced prediction class"""
    
    def __init__(self, model, label_encoders, feature_names, global_stats):
        self.model = model
        self.label_encoders = label_encoders
        self.feature_names = feature_names
        self.global_stats = global_stats
    
    def predict_delay(self, stop_name: str, pattern_text: str, direction: str, 
                     weekday: int, month: int, hour: int, minute: int,
                     route_history: Optional[Dict] = None, 
                     recent_delays: Optional[List[float]] = None) -> Dict:
        """
        Advanced ensemble prediction with historical context
        
        Parameters:
        - Basic trip info: stop_name, pattern_text, direction, weekday, month, hour, minute
        - route_history: Recent performance stats for this route (optional)
        - recent_delays: Recent delays on this route (optional, list of delays)
        
        Returns:
        - Dictionary with prediction, individual model predictions, and confidence interval
        """
        
        sample = {
            'hour': hour,
            'minute': minute,
            'weekday': weekday,
            'month': month,
            'day_of_month': 15, #fallback value
            'week_of_year': 26,  #fallback value
            'is_weekend': 1 if weekday >= 5 else 0,
            'is_rush_hour': 1 if (7 <= hour <= 9) or (16 <= hour <= 18) else 0,
            'is_late_night': 1 if hour >= 22 or hour <= 5 else 0,
        }
        
        # cyclical features
        sample.update({
            'hour_sin': np.sin(2 * np.pi * hour / 24),
            'hour_cos': np.cos(2 * np.pi * hour / 24),
            'minute_sin': np.sin(2 * np.pi * minute / 60),
            'minute_cos': np.cos(2 * np.pi * minute / 60),
            'weekday_sin': np.sin(2 * np.pi * weekday / 7),
            'weekday_cos': np.cos(2 * np.pi * weekday / 7),
            'month_sin': np.sin(2 * np.pi * month / 12),
            'month_cos': np.cos(2 * np.pi * month / 12),
        })
        
        # encoding
        sample.update({
            'stopName': self._encode_categorical('stopName', stop_name),
            'patternText': self._encode_categorical('patternText', pattern_text),
            'direction': self._encode_categorical('direction', direction),
            'route_stop_key': self._encode_categorical('route_stop_key', f"{pattern_text}_{stop_name}"),
            'route_dir_key': self._encode_categorical('route_dir_key', f"{pattern_text}_{direction}"),
        })
        
        # add route performance features
        global_delay_mean = self.global_stats['mean']
        global_delay_std = self.global_stats['std']
        
        sample.update({
            'route_mean': route_history.get('mean', global_delay_mean) if route_history else global_delay_mean,
            'route_std': route_history.get('std', global_delay_std) if route_history else global_delay_std,
            'route_median': route_history.get('median', global_delay_mean) if route_history else global_delay_mean,
            'route_count': route_history.get('count', 100) if route_history else 100,
            'stop_mean': global_delay_mean,
            'stop_std': global_delay_std,
            'stop_median': global_delay_mean,
            'stop_count': 100,
            'direction_mean': global_delay_mean,
            'direction_std': global_delay_std,
            'direction_median': global_delay_mean,
            'direction_count': 100,
            'hour_mean': global_delay_mean,
            'hour_std': global_delay_std,
            'hour_count': 100,
        })
        
        # historical pattern features
        if recent_delays:
            recent_delays_array = np.array(recent_delays)
            sample.update({
                'delay_minutes_rolling_mean_3': np.mean(recent_delays_array[-3:]) if len(recent_delays_array) >= 3 else global_delay_mean,
                'delay_minutes_rolling_mean_5': np.mean(recent_delays_array[-5:]) if len(recent_delays_array) >= 5 else global_delay_mean,
                'delay_minutes_rolling_std_3': np.std(recent_delays_array[-3:]) if len(recent_delays_array) >= 3 else global_delay_std,
                'delay_minutes_lag_1': recent_delays_array[-1] if len(recent_delays_array) >= 1 else global_delay_mean,
                'delay_minutes_lag_2': recent_delays_array[-2] if len(recent_delays_array) >= 2 else global_delay_mean,
                'delay_minutes_lag_3': recent_delays_array[-3] if len(recent_delays_array) >= 3 else global_delay_mean,
            })
        else:
            # gloal averages
            sample.update({
                'delay_minutes_rolling_mean_3': global_delay_mean,
                'delay_minutes_rolling_mean_5': global_delay_mean,
                'delay_minutes_rolling_std_3': global_delay_std,
                'delay_minutes_lag_1': global_delay_mean,
                'delay_minutes_lag_2': global_delay_mean,
                'delay_minutes_lag_3': global_delay_mean,
            })
        
        # anomaly features and interactions
        sample.update({
            'delay_anomaly_global': 0,
            'delay_anomaly_route': 0,
            'delay_anomaly_stop': 0,
            'delay_anomaly_hour': 0,
            'route_hour_interaction': sample['route_mean'] * sample['hour_sin'],
            'stop_weekend_interaction': sample['stop_mean'] * sample['is_weekend'],
            'rush_hour_route_interaction': sample['is_rush_hour'] * sample['route_mean'],
        })
        
        sample_df = pd.DataFrame([sample])
        
        # fill missing features with zeros
        for col in self.feature_names:
            if col not in sample_df.columns:
                sample_df[col] = 0
        
        # reorder columns to match training data
        sample_df = sample_df[self.feature_names]
        
        predicted_delay = float(self.model.predict(sample_df)[0]) # python float
        
        individual_preds = {}
        if hasattr(self.model, 'models') and self.model.models:
            individual_preds = self.model.predict_with_individual_models(sample_df)
        
        return {
            'predicted_delay': max(predicted_delay, 0.0), # so its not negative
            'individual_predictions': individual_preds,
            'confidence_interval': {
                'lower': float(predicted_delay - 1.96 * global_delay_std),
                'upper': float(predicted_delay + 1.96 * global_delay_std)
            },
            'metadata': {
                'stop_name': stop_name,
                'pattern_text': pattern_text,
                'direction': direction,
                'hour': hour,
                'minute': minute,
                'weekday': weekday,
                'month': month
            }
        }
    
    def _encode_categorical(self, column_name: str, value: str) -> int:
        """Encode categorical value using saved encoders"""
        if column_name in self.label_encoders:
            le = self.label_encoders[column_name]
            if value in le.classes_:
                return le.transform([value])[0]
            else:
                return -1  # unknown category
        return 0  # default value
    
    def predict_batch(self, trips: List[Dict]) -> List[Dict]:
        """Predict delays for multiple trips"""
        predictions = []
        
        for trip in trips:
            try:
                prediction = self.predict_delay(**trip)
                predictions.append(prediction)
            except Exception as e:
                print(f"Error predicting for trip {trip}: {e}")
                # return default prediction
                predictions.append({
                    'predicted_delay': self.global_stats['mean'],
                    'error': str(e)
                })
        
        return predictions


def create_sample_predictions():
    """Create sample prediction cases for testing"""
    return [
        {
            'case': 'Rush hour downtown',
            'params': {
                'stop_name': 'Hauptbahnhof',
                'pattern_text': '11',
                'direction': 'Dietrichsdorf',
                'weekday': 1,
                'month': 8,
                'hour': 8,
                'minute': 30,
                'route_history': {'mean': 2.5, 'std': 1.8, 'median': 2.0, 'count': 150},
                'recent_delays': [1.5, 2.2, 3.1, 2.8, 2.0]
            }
        },
        {
            'case': 'Late night service',
            'params': {
                'stop_name': 'Rungholtplatz',
                'pattern_text': 'N22',
                'direction': 'Schwentinental',
                'weekday': 5,
                'month': 12,
                'hour': 0,
                'minute': 22,
                'route_history': {'mean': 4.2, 'std': 3.1, 'median': 3.5, 'count': 45},
                'recent_delays': [5.2, 3.8, 6.1, 4.5]
            }
        },
        {
            'case': 'Weekend regular service',
            'params': {
                'stop_name': 'Campus',
                'pattern_text': '100',
                'direction': 'Mettenhof',
                'weekday': 6,
                'month': 6,
                'hour': 14,
                'minute': 15,
                'recent_delays': [0.5, 1.2, 0.8]
            }
        }
    ]
