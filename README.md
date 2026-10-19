# Reguly - Tax Compliance Predictor (Fintech Impact)

This repository contains the machine-learning and deployment code for **Reguly**, a B2B Neobanking & Tax Compliance Fintech for the Spanish market.

## Objective
To predict the probability that a business account (solo entrepreneur, micro-business, or startup in Spain) will remain tax-compliant or receive an automated tax penalty from the Spanish Tax Agency (*Hacienda / AEAT*). This supports Reguly's mission to combine banking operations with automated regulatory compliance, triggering proactive alerts and auto-filing invoices before deadlines.

## Model Variables
* **`late_filings_count`** (Discrete): Number of late tax filings in the last 12 months (0 to 10).
* **`annual_gross_revenue`** (Continuous): Annual gross revenue in EUR (€10,000 to €500,000).
* **`unpaid_taxes`** (Continuous): Outstanding tax or social security debt in EUR (€0 to €50,000).
* **Target (`compliant`)**: Binary (1 = Compliant / Zero Fines, 0 = Non-Compliant / Tax Penalty Received).

## Project Structure
```
Reguly-ML/
│
├── fintech.py                 # Script that generates synthetic data, trains & validates the ML model
├── fintech.xlsx               # Generated synthetic dataset (1,000 Spanish business accounts)
├── fintech.joblib             # Trained Logistic Regression model
├── flask_app.py               # Flask web application and JSON API endpoint
├── requirements.txt           # Python dependencies
├── templates/
│     └── index.html           # HTML front-end form for live predictions
└── README.md                  # Project documentation
```

## Setup Instructions
1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd <your-repo-folder>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the model training script to generate the synthetic dataset and model:
   ```bash
   python fintech.py
   ```
4. Start the Flask application locally:
   ```bash
   python flask_app.py
   ```
5. Open `http://127.0.0.1:5000/` in your browser to use the HTML form.

## API Endpoint Documentation
* **Endpoint URL:** `/api/predict` (or `/predict`)
* **Method:** `GET` or `POST`
* **Description:** Receives the three input variables, loads the trained Logistic Regression model (`fintech.joblib`), and returns a JSON response containing the prediction and probability of compliance.

### Example Request (GET)
```http
GET /api/predict?late_filings_count=0&annual_gross_revenue=150000&unpaid_taxes=0
```

### Example JSON Response
```json
{
  "inputs_received": {
    "annual_gross_revenue": 150000.0,
    "late_filings_count": 0.0,
    "unpaid_taxes": 0.0
  },
  "prediction_class": 1,
  "prediction_label": "Compliant (Zero Missed-Deadline Fines)",
  "probability_compliant": 0.775,
  "probability_non_compliant": 0.225
}
```
