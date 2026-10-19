"""
Logistic Regression – Reguly (Spain)
User-account-application-level compliance model.

Features
--------
- late_filings_count : discrete  – number of late tax filings
- annual_gross_revenue : continuous – yearly gross revenue (€)
- unpaid_taxes : continuous – outstanding unpaid taxes (€)

Target
------
- compliant : binary (1 = files on time, zero fines; 0 = tax penalty received)

Author  : Patricio De Leopoldo Peñas
Date    : 2026-10-19
"""

# ── Imports ──────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

# ── 1. Reproducible random data generation ───────────────────────────────
np.random.seed(42)

n_samples = 1_000

late_filings_count = np.random.randint(0, 11, size=n_samples)            # 0-10 filings
annual_gross_revenue = np.random.uniform(10_000, 500_000, size=n_samples) # €10 k – €500 k
unpaid_taxes = np.random.uniform(0, 50_000, size=n_samples)              # €0 – €50 k

# Simulate the target with a realistic latent relationship:
#   more late filings & more unpaid taxes → less likely compliant
#   higher revenue → slightly more likely compliant (better resources)
linear_combination = (
    -0.60 * late_filings_count
    + 0.000005 * annual_gross_revenue
    - 0.00005 * unpaid_taxes
)
probability = 1 / (1 + np.exp(-linear_combination))                      # sigmoid
compliant = (probability > np.random.rand(n_samples)).astype(int)

# ── 2. Build DataFrame & export to Excel ─────────────────────────────────
df = pd.DataFrame({
    "late_filings_count": late_filings_count,
    "annual_gross_revenue": annual_gross_revenue,
    "unpaid_taxes": unpaid_taxes,
    "compliant": compliant,
})

df.to_excel("fintech.xlsx", index=False)
print("✅  fintech.xlsx saved successfully.\n")
print(df.head(10))
print(f"\nDataset shape : {df.shape}")
print(f"Class balance :\n{df['compliant'].value_counts()}\n")

# ── 3. Train / Test split ────────────────────────────────────────────────
X = df[["late_filings_count", "annual_gross_revenue", "unpaid_taxes"]]
y = df["compliant"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"Train size : {X_train.shape[0]}")
print(f"Test  size : {X_test.shape[0]}\n")

# ── 4. Fit logistic regression ───────────────────────────────────────────
model = LogisticRegression(max_iter=1_000, random_state=42)
model.fit(X_train, y_train)

# ── 5. Validation – Classification reports ───────────────────────────────
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

print("=" * 55)
print("CLASSIFICATION REPORT – TRAIN SET")
print("=" * 55)
print(classification_report(y_train, y_train_pred, target_names=["Non-Compliant", "Compliant"]))

print("=" * 55)
print("CLASSIFICATION REPORT – TEST SET")
print("=" * 55)
print(classification_report(y_test, y_test_pred, target_names=["Non-Compliant", "Compliant"]))

# ── 6. Model coefficients (interpretability) ─────────────────────────────
print("Model Coefficients")
print("-" * 55)
for name, coef in zip(X.columns, model.coef_[0]):
    print(f"  {name:.<30s} {coef:+.6f}")
print(f"  {'intercept':.<30s} {model.intercept_[0]:+.6f}\n")

# ── 7. Persist the model with joblib ─────────────────────────────────────
joblib.dump(model, "fintech.joblib")
print("✅  Model saved to fintech.joblib")
