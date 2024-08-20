from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

# Load the dataset
data = pd.read_csv('Cleaned_data.csv')

# Ensure the correct version of scikit-learn is used
from sklearn import __version__ as sklearn_version
assert sklearn_version == '1.4.2', "scikit-learn version mismatch. Expected 1.4.2."

# Load the model
with open('RidgeModel.pkl', 'rb') as file:
    pipe = pickle.load(file)

@app.route('/')
def index():
    locations = sorted(data['location'].unique())
    return render_template('index.html', locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    location = request.form.get('location')
    bhk = float(request.form.get('bhk'))
    bath = float(request.form.get('bath'))
    sqft = float(request.form.get('total_sqft'))

    input_data = pd.DataFrame([[location, sqft, bath, bhk]], columns=['location', 'total_sqft', 'bath', 'bhk'])
    prediction = pipe.predict(input_data)[0] * 1e5

    return str(np.round(prediction, 2))

if __name__ == '__main__':
    app.run(debug=True, port=5001)
