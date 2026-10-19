\# Reguly - Tax Compliance Predictor (Fintech Impact)



This repository contains the machine-learning and deployment code for \*\*Reguly\*\*, a B2B Neobanking \& Tax Compliance Fintech for the Spanish market.



\## Objective

To predict the probability that a business account (micro-business/freelancer) will remain tax-compliant or receive an automated tax penalty from the Spanish Tax Agency. 



\## Model Variables

\* \*\*`late\_filings\_count`\*\* (Discrete): Number of late tax filings in the last 12 months.

\* \*\*`annual\_gross\_revenue`\*\* (Continuous): Annual gross revenue in EUR.

\* \*\*`unpaid\_taxes`\*\* (Continuous): Outstanding tax or social security debt in EUR.

\* \*\*Target (`compliant`)\*\*: Binary (1 = Compliant/No Fines, 0 = Non-Compliant/Penalty Received).



\## Setup Instructions

1\. Clone the repository.

2\. Install dependencies: `pip install -r requirements.txt`

3\. Run the model training script to generate the synthetic data and model: `python fintech\_2.py`

4\. Start the Flask application locally: `python flask\_app.py`

5\. Open `http://127.0.0.1:5000/` in your browser.



\## API Endpoint Documentation

\* \*\*Endpoint URL:\*\* `/predict`

\* \*\*Method:\*\* `GET`

\* \*\*Description:\*\* Receives the three input variables, loads the trained Logistic Regression model (`fintech\_2.joblib`), and returns a JSON response containing the prediction and probability of compliance.



\### Example Request

```http

GET /predict?late\_filings\_count=0\&annual\_gross\_revenue=150000\&unpaid\_taxes=0

