import pandas as pd
import numpy as np
import pickle
from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

# Load models from the models directory
ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))

@app.route("/")
def index():
    # Typically routes to a landing page or index page
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        # 1. Extract values from the form inputs using the HTML 'name' attribute
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes')) # 1 for Fire, 0 for Not Fire
        Region = float(request.form.get('Region'))   # 1 for Bejaia, 0 for Sidi Bel-Abbes

        # 2. Arrange into a 2D array matching the exact feature order used during training
        # Order: [Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]
        new_data_scaled = standard_scaler.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
        
        # 3. Predict the FWI result
        result = ridge_model.predict(new_data_scaled)
        
        # 4. Return the prediction result back to your UI template
        return render_template('home.html', result=round(result[0], 2))
        
    else:
        # If it's a GET request, just display the empty form page
        return render_template('home.html')

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True) # Added debug=True for easier local debugging