from flask import Flask, render_template, request, jsonify
import pandas as pd
from joblib import load
import os

app = Flask(__name__)

# Load the trained Reguly model
# Adjust the path if PythonAnywhere requires a specific absolute path 
# (e.g., '/home/yourusername/mysite/fintech.joblib')
MODEL_PATH = 'fintech.joblib'
try:
    model = load(MODEL_PATH)
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

@app.route("/")
def index():
    """Serves the HTML Front End"""
    return render_template("index.html")

@app.route("/predict", methods=['GET'])
def predict():
    """API Endpoint: Receives inputs, applies model, returns JSON prediction."""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        # 1. Get parameters from the request
        late_filings = float(request.args.get('late_filings_count', 0))
        revenue = float(request.args.get('annual_gross_revenue', 0))
        unpaid_taxes = float(request.args.get('unpaid_taxes', 0))

        # 2. Put them into a DataFrame to match training data structure
        input_data = pd.DataFrame({
            'late_filings_count': [late_filings],
            'annual_gross_revenue': [revenue],
            'unpaid_taxes': [unpaid_taxes]
        })

        # 3. Predict compliance (1 = Compliant, 0 = Non-Compliant)
        prediction_class = int(model.predict(input_data)[0])
        # Get the probability of being class '1' (Compliant)
        probability_compliant = float(model.predict_proba(input_data)[0][1])

        # 4. Construct JSON response
        result = {
            "prediction_class": prediction_class,
            "prediction_label": "Compliant (No Fines)" if prediction_class == 1 else "Non-Compliant (High Risk of Fines)",
            "probability_compliant": round(probability_compliant, 4),
            "inputs_received": {
                "late_filings_count": late_filings,
                "annual_gross_revenue": revenue,
                "unpaid_taxes": unpaid_taxes
            }
        }
        
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Standard boilerplate to run the app locally (not strictly needed for PythonAnywhere WSGI)
if __name__ == "__main__":
    app.run(debug=True)