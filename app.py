from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

model = tf.keras.models.load_model('model.keras')
preprocessor = joblib.load('preprocessor.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Create a single-row DataFrame matching training column names
    input_df = pd.DataFrame([{
        'Time (24hr)': float(data['time']),
        'Number of Vehicles': int(data['vehicles']),
        'Road Surface': int(data['road_surface']),
        'Lighting Conditions': int(data['lighting']),
        'Weather Conditions': int(data['weather']),
        'Type of Vehicle': int(data['vehicle_type']),
        'Age of Casualty': float(data['age']),
        'Sex of Casualty': int(data['sex'])
    }])

    # Preprocess using saved pipeline
    processed_input = preprocessor.transform(input_df).astype(np.float32)

    prob = float(model.predict(processed_input, verbose=0)[0][0] * 100)

    return jsonify({'prediction': f"Estimated Accident Risk: {prob:.1f}%", 'prob': prob})


if __name__ == '__main__':
    app.run(debug=True)