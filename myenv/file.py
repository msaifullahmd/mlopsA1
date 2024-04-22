from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


app = Flask(__name__)

# Load and prepare the dataset and model
data = pd.read_csv('healthcare-dataset-stroke-data.csv')
X = data[['age']].values
y = data['stroke'].values.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

@app.route('/predict', methods=['POST'])
def predict():
    # Get age from the request's body
    request_data = request.get_json()
    print(request_data)
    age = request_data['age']
    
    # Ensure age is in the form of a 2D array for sklearn
    age_array = np.array([[age]])
    
    # Make prediction
    prediction = model.predict(age_array)
    
    # Return prediction in a JSON format
    return jsonify({'prediction': prediction[0, 0]})

if __name__ == '__main__':
    app.run(debug=True)

