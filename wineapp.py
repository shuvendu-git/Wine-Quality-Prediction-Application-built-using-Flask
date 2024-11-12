from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the pre-trained model and scaler
model = pickle.load(open('wine_model_new.pkl', 'rb'))
scaler = StandardScaler()

# Home route to render the input form
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle prediction requests
@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from form
    features = [
        float(request.form['citric_acid']),
        float(request.form['residual_sugar']),
        float(request.form['chlorides']),
        float(request.form['free_sulfur_dioxide']),
        float(request.form['total_sulfur_dioxide']),
    ]

    # Scale input features
    scaled_features = scaler.fit_transform(np.array(features).reshape(1, -1))

    # Make prediction
    prediction = model.predict(scaled_features)
    
    # Return the prediction result
    return jsonify({'quality_prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
