from flask import Flask, request, jsonify, send_from_directory
import joblib
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the LSTM model and the scaler
model = load_model('lstm_model.h5')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def serve_index():
    return send_from_directory('static', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    size = data['input_data'][0]

    # Here, we would transform the size input as needed, similar to how the data was preprocessed
    size = np.array([[size]])
    scaled_size = scaler.transform(size)

    # Predict the future price
    scaled_prediction = model.predict(scaled_size)
    prediction = scaler.inverse_transform(scaled_prediction)
    
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
