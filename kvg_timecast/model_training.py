import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import os

from config import LIGHTGBM_BASE_PARAMS, CV_FOLDS, TEST_SIZE, STACKING_LEARNER_MAX_SAMPLES


class KVGEnsembleModel:
    """VERY :) Advanced ensemble model for KVG delay prediction"""
    
    def __init__(self):
        self.models = {}
        self.stacking_ensemble = None
        self.feature_names = None
        self.global_stats = {}
        
    def create_base_models(self):
        """Create specialized LightGBM models"""
        print("Creating specialized ensemble models (LightGBM)...")
        
        # Model 1 general delay predictor
        self.models['lgb_general'] = lgb.LGBMRegressor(
            **LIGHTGBM_BASE_PARAMS,
            n_estimators=500,
            max_depth=7
        )
        
        # Model 2 high-frequency routes specialist  
        self.models['lgb_frequency'] = lgb.LGBMRegressor(
            **LIGHTGBM_BASE_PARAMS,
            n_estimators=300,
            max_depth=5,
            min_child_samples=20
        )
        
        # Model 3 time-sensitive model (focus on temporal patterns)
        params3 = LIGHTGBM_BASE_PARAMS.copy()
        params3['num_leaves'] = 63
        self.models['lgb_temporal'] = lgb.LGBMRegressor(
            **params3,
            n_estimators=400,
            max_depth=6
        )
        
        # Model 4 anomaly-aware model
        params4 = LIGHTGBM_BASE_PARAMS.copy()
        params4['feature_fraction'] = 0.9
        self.models['lgb_anomaly'] = lgb.LGBMRegressor(
            **params4,
            n_estimators=350,
            max_depth=8
        )
        
        print(f" Created {len(self.models)} specialized base models")
    
    def create_stacking_ensemble(self):
        """Create stacking ensemble with meta-learner"""
        print(" Building stacking ensemble...")
        
        # base models list
        base_models = [
            ('lgb_general', self.models['lgb_general']),
            ('lgb_temporal', self.models['lgb_temporal'])
        ]

        meta_learner = RidgeCV(alphas=[0.1, 1, 10], cv=5)

        self.stacking_ensemble = StackingRegressor(
            estimators=base_models,
            final_estimator=meta_learner,
            cv=5,
            n_jobs=1  # 1 to avoid memory overload on my machine :(
        )

        print(" Stacking ensemble created")

    def train(self, X_train, y_train):
        """Train the ensemble model with memory-efficient approach"""
        print("Starting ensemble training process...")
        
        # store feature names and global statistics
        self.feature_names = X_train.columns.tolist()
        self.global_stats = {
            'mean': y_train.mean(),
            'std': y_train.std(),
            'median': y_train.median(),
            'min': y_train.min(),
            'max': y_train.max()
        }
        
        if not self.models:
            self.create_base_models()
        
        # train individual models first (for feature importance analysis)
        print("Training individual models...")
        for name, model in self.models.items():
            print(f"  Training {name}...")
            model.fit(X_train, y_train)
        
        try:
            print("Attempting stacking ensemble training...")
            if self.stacking_ensemble is None:
                self.create_stacking_ensemble()
            
            # train with smaller batch if memory issues
            print(" Using reduced data size for stacking to prevent crashes...")
            
            sample_size = min(len(X_train), STACKING_LEARNER_MAX_SAMPLES)

            if len(X_train) > sample_size:
                indices = np.random.choice(len(X_train), sample_size, replace=False)
                X_stack = X_train.iloc[indices]
                y_stack = y_train.iloc[indices]
                print(f"  Using {sample_size:,} samples for stacking (instead of {len(X_train):,})")
            else:
                X_stack = X_train
                y_stack = y_train
            
            self.stacking_ensemble.fit(X_stack, y_stack)
            print(" Stacking ensemble training completed!")
            
        except Exception as e:
            print(f" Stacking ensemble training failed: {str(e)}")
            return
        
        print(" Ensemble training completed!")
    
    def predict(self, X):
        """Make predictions using the ensemble"""
        if self.stacking_ensemble is None:
            raise ValueError("Model not trained yet!")
        
        return self.stacking_ensemble.predict(X)
    
    def predict_with_individual_models(self, X):
        """Get predictions from both ensemble and individual models"""
        predictions = {}
        
        # ensemble prediction
        predictions['ensemble'] = self.predict(X)
        
        # individual model predictions
        for name, model in self.models.items():
            predictions[name] = model.predict(X)
        
        return predictions
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        print(" Evaluating ensemble performance...")
        
        # get predictions
        y_pred_ensemble = self.predict(X_test)
        
        # individual model predictions
        individual_predictions = {}
        for name, model in self.models.items():
            individual_predictions[name] = model.predict(X_test)
        
        def calculate_metrics(y_true, y_pred, model_name):
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)
            
            return {"MAE": mae, "RMSE": rmse, "R2": r2, "name": model_name}
        
        results = {}
        
        for name, pred in individual_predictions.items():
            results[name] = calculate_metrics(y_test, pred, name)
        
        results['stacking_ensemble'] = calculate_metrics(y_test, y_pred_ensemble, "Stacking Ensemble")
        
        # baseline comparison (to be deleted i guess?)
        baseline_pred = np.full(len(y_test), self.global_stats['mean'])
        results['baseline'] = calculate_metrics(y_test, baseline_pred, "Baseline")
        
        print("\n=== MODEL PERFORMANCE COMPARISON ===")
        for name, metrics in results.items():
            print(f"{metrics['name']}:")
            print(f"  MAE:  {metrics['MAE']:.3f} minutes")
            print(f"  RMSE: {metrics['RMSE']:.3f} minutes")
            print(f"  R²:   {metrics['R2']:.4f}")
            print()
        
        return results
    
    def get_feature_importance(self):
        """Get feature importance from the best individual model"""
        if not self.models:
            raise ValueError("Models not trained yet!")
        
        # use the general model for feature importance
        best_model = self.models['lgb_general']
        
        feature_importance = pd.DataFrame({
            "feature": self.feature_names,
            "importance": best_model.feature_importances_
        }).sort_values("importance", ascending=False)
        
        return feature_importance
    
    def save_model(self, model_path, encoders_path, features_path, label_encoders):
        """Save the trained model, encoders, and feature names"""
        print(f"Saving model to {model_path}...")
        
        # save the ensemble model
        model_data = {
            'stacking_ensemble': self.stacking_ensemble,
            'individual_models': self.models,
            'feature_names': self.feature_names,
            'global_stats': self.global_stats
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        # save label encoders
        with open(encoders_path, 'wb') as f:
            pickle.dump(label_encoders, f)
        
        # save feature names
        with open(features_path, 'wb') as f:
            pickle.dump(self.feature_names, f)
        
        print(f" Model saved successfully!")
        print(f"   Model: {model_path}")
        print(f"   Encoders: {encoders_path}")
        print(f"   Features: {features_path}")
    
    @classmethod
    def load_model(cls, model_path, encoders_path, features_path):
        """Load a trained model"""
        print(f" Loading model from {model_path}...")
        
        # create instance
        instance = cls()
        
        # load model data
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        instance.stacking_ensemble = model_data['stacking_ensemble']
        instance.models = model_data['individual_models']
        instance.feature_names = model_data['feature_names']
        instance.global_stats = model_data['global_stats']
        
        # load encoders
        with open(encoders_path, 'rb') as f:
            label_encoders = pickle.load(f)
        
        print(" Model loaded successfully!")
        return instance, label_encoders


def create_time_aware_split(X, y, test_size=TEST_SIZE):
    """Create time-aware train-test split"""
    print(" Creating time-aware train-test split...")
    
    # use the last portion as test set (most recent data)
    split_idx = int((1 - test_size) * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Training set: {X_train.shape[0]:,} samples")
    print(f"Test set: {X_test.shape[0]:,} samples")
    
    return X_train, X_test, y_train, y_test
