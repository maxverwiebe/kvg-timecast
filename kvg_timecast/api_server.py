"""
API server for KVG Time Series Prediction
"""

import sys
import os
from datetime import datetime
from typing import Dict, List, Optional
import json

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from flask import Flask, request, jsonify
    from flask_cors import CORS
except ImportError:
    print("Flask not installed. Install with: pip install flask flask-cors")
    sys.exit(1)

from config import MODEL_PATH, ENCODERS_PATH, FEATURES_PATH
from model_training import KVGEnsembleModel
from prediction import KVGPredictor


# Global variables for model
predictor = None
app = Flask(__name__)
CORS(app)  # Enable CORS for web frontend


def load_model():
    """Load the trained model"""
    global predictor
    
    try:
        print("📁 Loading model for API...")
        model, label_encoders = KVGEnsembleModel.load_model(
            MODEL_PATH, 
            ENCODERS_PATH, 
            FEATURES_PATH
        )
        
        predictor = KVGPredictor(
            model=model,
            label_encoders=label_encoders,
            feature_names=model.feature_names,
            global_stats=model.global_stats
        )
        
        print(" Model loaded successfully for API!")
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return False


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': predictor is not None,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/predict', methods=['POST'])
def predict_delay():
    """Predict delay for a single trip"""
    if predictor is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['stop_name', 'pattern_text', 'direction', 'weekday', 'month', 'hour', 'minute']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Extract parameters
        prediction_params = {
            'stop_name': data['stop_name'],
            'pattern_text': str(data['pattern_text']),
            'direction': data['direction'],
            'weekday': int(data['weekday']),
            'month': int(data['month']),
            'hour': int(data['hour']),
            'minute': int(data['minute'])
        }
        
        # Optional parameters
        if 'route_history' in data:
            prediction_params['route_history'] = data['route_history']
        
        if 'recent_delays' in data:
            prediction_params['recent_delays'] = data['recent_delays']
        
        # Make prediction
        result = predictor.predict_delay(**prediction_params)
        
        # Format response
        response = {
            'success': True,
            'predicted_delay_minutes': round(result['predicted_delay'], 2),
            'confidence_interval': {
                'lower': round(result['confidence_interval']['lower'], 2),
                'upper': round(result['confidence_interval']['upper'], 2)
            },
            'metadata': result['metadata'],
            'timestamp': datetime.now().isoformat()
        }
        
        # Add individual predictions if available
        if result['individual_predictions']:
            response['individual_predictions'] = {
                k: round(v, 2) for k, v in result['individual_predictions'].items()
                if k != 'ensemble'
            }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 400


@app.route('/predict_batch', methods=['POST'])
def predict_delay_batch():
    """Predict delays for multiple trips"""
    if predictor is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        
        if 'trips' not in data:
            return jsonify({'error': 'Missing trips array'}), 400
        
        trips = data['trips']
        predictions = []
        
        for i, trip in enumerate(trips):
            try:
                # Validate trip data
                required_fields = ['stop_name', 'pattern_text', 'direction', 'weekday', 'month', 'hour', 'minute']
                for field in required_fields:
                    if field not in trip:
                        predictions.append({
                            'trip_index': i,
                            'success': False,
                            'error': f'Missing required field: {field}'
                        })
                        continue
                
                # Make prediction
                result = predictor.predict_delay(
                    stop_name=trip['stop_name'],
                    pattern_text=str(trip['pattern_text']),
                    direction=trip['direction'],
                    weekday=int(trip['weekday']),
                    month=int(trip['month']),
                    hour=int(trip['hour']),
                    minute=int(trip['minute']),
                    route_history=trip.get('route_history'),
                    recent_delays=trip.get('recent_delays')
                )
                
                predictions.append({
                    'trip_index': i,
                    'success': True,
                    'predicted_delay_minutes': round(result['predicted_delay'], 2),
                    'confidence_interval': {
                        'lower': round(result['confidence_interval']['lower'], 2),
                        'upper': round(result['confidence_interval']['upper'], 2)
                    },
                    'metadata': result['metadata']
                })
                
            except Exception as e:
                predictions.append({
                    'trip_index': i,
                    'success': False,
                    'error': str(e)
                })
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 400


@app.route('/model_info', methods=['GET'])
def model_info():
    """Get information about the loaded model"""
    if predictor is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify({
        'model_loaded': True,
        'feature_count': len(predictor.feature_names),
        'global_stats': predictor.global_stats,
        'available_encoders': list(predictor.label_encoders.keys()),
        'timestamp': datetime.now().isoformat()
    })


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


def main():
    """Start the API server"""
    print("🚀 Starting KVG Prediction API Server")
    print("=" * 40)
    
    # Load model
    if not load_model():
        print("❌ Failed to load model. Please run train_model.py first.")
        return False
    
    print("\n📡 API Endpoints:")
    print("  GET  /health        - Health check")
    print("  POST /predict       - Single prediction")
    print("  POST /predict_batch - Batch predictions")
    print("  GET  /model_info    - Model information")
    
    print(f"\n🌐 Starting server at http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=False)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
