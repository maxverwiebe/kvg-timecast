import sys
import os
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import DATA_PATH, MODEL_PATH, ENCODERS_PATH, FEATURES_PATH, MAX_RECORDS
from data_preprocessing import DataPreprocessor
from feature_engineering import FeatureEngineer
from model_training import KVGEnsembleModel, create_time_aware_split


def main():
    """Main training pipeline"""
    print("Starting KVG Time Series Prediction Training Pipeline")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    try:
        print("\n STEP 1: Data Loading and Preprocessing")
        preprocessor = DataPreprocessor()
        
        dataset = preprocessor.load_and_clean_data(DATA_PATH, MAX_RECORDS)
        
        dataset = preprocessor.create_temporal_features(dataset)
        
        dataset = preprocessor.create_route_identifiers(dataset)
        
        # feature engineering
        print("\nSTEP 2: Feature Engineering")
        feature_engineer = FeatureEngineer()
        dataset = feature_engineer.engineer_all_features(dataset)
        
        # data prep
        print("\n STEP 3: Data Preparation")
        # encode categorical features
        dataset = preprocessor.encode_categorical_features(dataset, fit=True)
        
        # feature matrix preparation
        # feature matrix is a matrix of features used for training
        X, y, feature_names = preprocessor.prepare_feature_matrix(dataset)
        
        # create time-aware train-test split
        X_train, X_test, y_train, y_test = create_time_aware_split(X, y)
        
        print(f"Training delay stats: mean={y_train.mean():.2f}, std={y_train.std():.2f}")
        print(f"Test delay stats: mean={y_test.mean():.2f}, std={y_test.std():.2f}")
        
        # actual training
        print("\nSTEP 4: Ensemble Model Training")
        model = KVGEnsembleModel()
        model.train(X_train, y_train)
        
        # eval
        print("\n STEP 5: Model Evaluation")
        results = model.evaluate(X_test, y_test)
        
        # feature importance analysis
        print("\nSTEP 6: Feature Importance Analysis")
        feature_importance = model.get_feature_importance()
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10).to_string(index=False))
        
        # model export
        print("\nSTEP 7: Saving Model")
        model.save_model(
            MODEL_PATH, 
            ENCODERS_PATH, 
            FEATURES_PATH, 
            preprocessor.get_encoders()
        )
        
        # summary
        print("\n" + "=" * 60)
        print(" TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        best_r2 = results['stacking_ensemble']['R2']
        best_mae = results['stacking_ensemble']['MAE']
        
        print(f" Final Model Performance:")
        print(f"   R^2 Score: {best_r2:.4f}")
        print(f"   MAE: {best_mae:.3f} minutes")
        print(f"   Model saved to: {MODEL_PATH}")
        print(f"   Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return True
        
    except Exception as e:
        print(f"\n!ERROR in training pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
