# ChurnX — Customer Churn Prediction System

A machine learning graduation project that predicts which telecom customers are likely to cancel their subscription, explains *why* using SHAP, and provides an interactive dashboard for business teams.

---

## Project Structure

```
ChurnX/
├── data/
│   ├── raw/          ← put the downloaded CSV here
│   └── processed/    ← auto-generated after running notebooks
├── notebooks/
│   ├── 01_EDA_and_Preprocessing.ipynb
│   ├── 02_Models_and_Evaluation.ipynb
│   └── 03_SHAP_Explainability.ipynb
├── models/           ← saved trained model (auto-generated)
├── app/
│   └── streamlit_app.py
├── reports/figures/  ← saved charts (auto-generated)
├── requirements.txt
└── README.md
```

---

## How to Run

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Download the dataset
1. Go to https://www.kaggle.com/datasets/blastchar/telco-customer-churn
2. Download `WA_Fn-UseC_-Telco-Customer-Churn.csv`
3. Put it inside `data/raw/`

### Step 3 — Run the notebooks in order
1. `01_EDA_and_Preprocessing.ipynb` — explore data, engineer features, handle imbalance
2. `02_Models_and_Evaluation.ipynb` — train and compare 3 models, tune the best one
3. `03_SHAP_Explainability.ipynb` — understand what drives churn predictions

### Step 4 — Launch the dashboard
```bash
cd app
streamlit run streamlit_app.py
```

---

## Models Used
| Model | Role |
|---|---|
| Logistic Regression | Baseline — simple and explainable |
| Random Forest | Strong ensemble model |
| XGBoost | Best performer — used in production |

## Key Techniques
- **SMOTE** — fixes class imbalance without losing real data
- **5-fold Cross Validation** — honest evaluation, not one lucky split
- **Optuna** — automatic hyperparameter tuning
- **SHAP** — explains every individual prediction

---

## Tech Stack
Python · pandas · scikit-learn · XGBoost · SHAP · Optuna · Streamlit · Plotly
