from flask import Flask, render_template, request
import pandas as pd
import pickle

# Load wine quality dataset
df = pd.read_csv('winequality_red.csv')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user inputs from form
    fixed_acidity = float(request.form['fixed_acidity'])
    volatile_acidity = float(request.form['volatile_acidity'])
    citric_acid = float(request.form['citric_acid'])
    residual_sugar = float(request.form['residual_sugar'])
    chlorides = float(request.form['chlorides'])
    free_sulfur_dioxide = float(request.form['free_sulfur_dioxide'])
    total_sulfur_dioxide = float(request.form['total_sulfur_dioxide'])
    density = float(request.form['density'])
    pH = float(request.form['pH'])
    sulphates = float(request.form['sulphates'])
    alcohol = float(request.form['alcohol'])

    # Preprocess user inputs
    X = pd.DataFrame({
        'fixed acidity': [fixed_acidity],
        'volatile acidity': [volatile_acidity],
        'citric acid': [citric_acid],
        'residual sugar': [residual_sugar],
        'chlorides': [chlorides],
        'free sulfur dioxide': [free_sulfur_dioxide],
        'total sulfur dioxide': [total_sulfur_dioxide],
        'density': [density],
        'pH': [pH],
        'sulphates': [sulphates],
        'alcohol': [alcohol]
    })

    # Load trained model
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Make wine quality prediction
    quality = model.predict(X)[0]

    # Return predicted wine quality to user
    return render_template('index.html', quality=quality)

if __name__ == '__main__':
    app.run(debug=True)
