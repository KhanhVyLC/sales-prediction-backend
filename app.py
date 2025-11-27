from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import json
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all origins

# Config from environment variables
AZURE_ML_ENDPOINT = os.environ.get(
    'AZURE_ML_ENDPOINT',
    'https://sales-sa-ml-gdjrr.southeastasia.inference.ml.azure.com/score'
)
AZURE_ML_API_KEY = os.environ.get('AZURE_ML_API_KEY', '')

@app.route('/')
def home():
    return jsonify({
        'status': 'ok',
        'message': 'Sales Prediction API is running',
        'endpoints': {
            'predict': '/predict',
            'test': '/test'
        }
    })

@app.route('/test', methods=['GET'])
def test():
    return jsonify({
        'status': 'ok',
        'message': 'Proxy server is running',
        'azure_endpoint': AZURE_ML_ENDPOINT
    })

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    # Handle preflight
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        client_data = request.get_json()
        
        print(f"Received request: {json.dumps(client_data, indent=2)}")
        
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {AZURE_ML_API_KEY}'
        }
        
        response = requests.post(
            AZURE_ML_ENDPOINT,
            headers=headers,
            json=client_data,
            timeout=30
        )
        
        print(f"Azure ML response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Azure ML raw result: {result}")
            
            # Parse if string
            if isinstance(result, str):
                try:
                    result = json.loads(result)
                except json.JSONDecodeError:
                    pass
            
            return jsonify(result), 200
        else:
            error_msg = response.text
            print(f"Azure ML error: {error_msg}")
            return jsonify({
                'error': f'Azure ML returned {response.status_code}',
                'details': error_msg
            }), response.status_code
            
    except Exception as e:
        print(f"Proxy error: {str(e)}")
        return jsonify({
            'error': 'Proxy server error',
            'details': str(e)
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)