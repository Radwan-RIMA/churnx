"""
ChurnX — Customer Churn Prediction Dashboard
Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
import os
import io

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnX",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load model files ──────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

@st.cache_resource   # only loads once — not every time the user interacts
def load_model():
    model = joblib.load(os.path.join(MODELS_DIR, 'best_model.joblib'))

    # Fix compatibility issue with saved XGBoost model
    if not hasattr(model, "gpu_id"):
        model.gpu_id = None

    if not hasattr(model, "predictor"):
        model.predictor = None

    model.use_label_encoder = False

    scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.joblib'))
    feature_names = joblib.load(os.path.join(MODELS_DIR, 'feature_names.joblib'))

    explainer = shap.TreeExplainer(model)

    return model, scaler, feature_names, explainer


# ── Simple login ──────────────────────────────────────────────────────────────
def login_page():
    st.title("🔐 ChurnX Login")
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        st.markdown("### Sign in to continue")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login", use_container_width=True):
            if username == "admin" and password == "churnx123":
                st.session_state['logged_in'] = True
                st.rerun()
            else:
                st.error("Wrong username or password. Try: admin / churnx123")
        st.caption("Demo credentials: admin / churnx123")


# ── Helper: preprocess uploaded CSV to match model's expected input ───────────
def preprocess_csv(df_raw, scaler, feature_names):
    df = df_raw.copy()

    # Drop customerID if present
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])

    # Fix TotalCharges
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)

    # Drop Churn column if it exists in the upload
    if 'Churn' in df.columns:
        df = df.drop(columns=['Churn'])

    # Service count feature
    service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                    'TechSupport', 'StreamingTV', 'StreamingMovies']
    df['service_count']       = df[service_cols].apply(lambda r: sum(r == 'Yes'), axis=1)
    df['charge_per_tenure']   = df['MonthlyCharges'] / (df['tenure'] + 1)
    df['has_premium_services'] = (df['service_count'] >= 3).astype(int)

    # Binary encoding
    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService',
                   'PaperlessBilling'] + service_cols
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map(lambda x: 1 if x in ['Yes', 'Female', 1, True] else 0)

    # One-hot encoding
    multi_cat = ['MultipleLines', 'InternetService', 'Contract', 'PaymentMethod']
    df = pd.get_dummies(df, columns=[c for c in multi_cat if c in df.columns], drop_first=True)

    # Align columns to match what the model expects
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0   # add missing columns as 0
    df = df[feature_names]   # put columns in the right order

    # Scale
    df_scaled = scaler.transform(df)
    return df_scaled, df


# ── Risk label helper ─────────────────────────────────────────────────────────
def risk_label(prob):
    if prob >= 0.7:
        return "🔴 High"
    elif prob >= 0.4:
        return "🟡 Medium"
    else:
        return "🟢 Low"


# ── Main app ──────────────────────────────────────────────────────────────────
def main_app():
    model, scaler, feature_names, explainer = load_model()

    # Sidebar navigation
    st.sidebar.image("https://img.icons8.com/color/96/graph.png", width=60)
    st.sidebar.title("ChurnX")
    st.sidebar.markdown("---")
    page = st.sidebar.radio(
        "Navigate",
        ["🏠 Dashboard", "🔍 Customer Lookup", "🎛 What-If Simulator", "📤 Export Report"]
    )
    st.sidebar.markdown("---")
    if st.sidebar.button("Logout"):
        st.session_state['logged_in'] = False
        st.rerun()

    # ── Upload section (shared across pages) ─────────────────────────────────
    uploaded_file = st.sidebar.file_uploader("Upload customer CSV", type=["csv"])

    df_raw = None
    df_results = None

    if uploaded_file:
        df_raw = pd.read_csv(uploaded_file)
        X_scaled, X_aligned = preprocess_csv(df_raw, scaler, feature_names)

        probs = model.predict_proba(X_scaled)[:, 1]
        preds = (probs >= 0.5).astype(int)

        df_results = df_raw.copy()
        if 'customerID' not in df_results.columns:
            df_results.insert(0, 'customerID', [f'CUST-{i:04d}' for i in range(len(df_results))])
        df_results['Churn Probability'] = (probs * 100).round(1)
        df_results['Prediction']        = preds
        df_results['Risk Level']        = probs.apply(risk_label) if hasattr(probs, 'apply') else pd.Series(probs).apply(risk_label)

        # Store in session so all pages can access it
        st.session_state['df_results'] = df_results
        st.session_state['X_scaled']   = X_scaled
        st.session_state['X_aligned']  = X_aligned
        st.session_state['probs']      = probs

    # Retrieve from session if already uploaded
    if 'df_results' in st.session_state:
        df_results = st.session_state['df_results']
        X_scaled   = st.session_state['X_scaled']
        X_aligned  = st.session_state['X_aligned']
        probs      = st.session_state['probs']

    # ── PAGE: Dashboard ───────────────────────────────────────────────────────
    if page == "🏠 Dashboard":
        st.title("📊 ChurnX — Customer Churn Prediction Dashboard")

        if df_results is None:
            st.info("Upload a customer CSV file in the sidebar to get started.")
            st.markdown("""
            **What this app does:**
            - Predicts which customers are likely to churn
            - Explains *why* using SHAP values
            - Lets you simulate 'what if we change this plan?' scenarios
            - Exports predictions as a downloadable report

            **How to use:**
            1. Upload the Telco Churn CSV in the sidebar
            2. View predictions on this page
            3. Click a customer to see their SHAP explanation
            4. Use the What-If Simulator to test interventions
            """)
            return

        # KPI cards
        total     = len(df_results)
        churned   = int((df_results['Prediction'] == 1).sum())
        high_risk = int((probs >= 0.7).sum())
        avg_prob  = probs.mean() * 100

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Customers",  f"{total:,}")
        c2.metric("Predicted Churners", f"{churned:,}", delta=f"{churned/total*100:.1f}%", delta_color="inverse")
        c3.metric("High Risk Customers", f"{high_risk:,}")
        c4.metric("Avg Churn Probability", f"{avg_prob:.1f}%")

        st.markdown("---")

        # Filter by risk
        col_left, col_right = st.columns([1, 2])

        with col_left:
            risk_filter = st.selectbox("Filter by Risk Level", ["All", "High", "Medium", "Low"])

        filtered = df_results.copy()
        if risk_filter == "High":
            filtered = filtered[probs >= 0.7]
        elif risk_filter == "Medium":
            filtered = filtered[(probs >= 0.4) & (probs < 0.7)]
        elif risk_filter == "Low":
            filtered = filtered[probs < 0.4]

        with col_right:
            st.markdown(f"Showing **{len(filtered)}** customers")

        # Color-coded table
        display_cols = ['customerID', 'tenure', 'MonthlyCharges', 'Contract',
                        'Churn Probability', 'Risk Level']
        display_cols = [c for c in display_cols if c in filtered.columns]
        st.dataframe(
            filtered[display_cols].reset_index(drop=True),
            use_container_width=True,
            height=400
        )

        # Churn probability distribution chart
        st.markdown("### Churn Probability Distribution")
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.hist(probs, bins=40, color='#F44336', alpha=0.7, edgecolor='white')
        ax.axvline(0.5, color='black', linestyle='--', linewidth=1.5, label='Decision threshold (0.5)')
        ax.axvline(0.7, color='orange', linestyle='--', linewidth=1.5, label='High risk threshold (0.7)')
        ax.set_xlabel('Churn Probability')
        ax.set_ylabel('Number of Customers')
        ax.set_title('Distribution of Predicted Churn Probabilities')
        ax.legend()
        st.pyplot(fig)
        plt.close()

    # ── PAGE: Customer Lookup ─────────────────────────────────────────────────
    elif page == "🔍 Customer Lookup":
        st.title("🔍 Individual Customer Explanation")

        if df_results is None:
            st.warning("Please upload a CSV file first.")
            return

        customer_ids = df_results['customerID'].tolist()
        selected_id  = st.selectbox("Select a customer", customer_ids)

        idx = df_results.index[df_results['customerID'] == selected_id].tolist()[0]
        prob = probs[idx]

        # Customer summary
        col1, col2, col3 = st.columns(3)
        col1.metric("Customer ID", selected_id)
        col2.metric("Churn Probability", f"{prob*100:.1f}%")
        col3.metric("Risk Level", risk_label(prob))

        # Progress bar showing risk
        st.progress(float(prob), text=f"Churn Risk: {prob*100:.1f}%")

        st.markdown("---")

        # SHAP waterfall explanation
        st.markdown("### Why is this customer at risk?")
        st.caption("Red bars = features pushing toward churn | Blue bars = features reducing churn risk")

        shap_values = explainer.shap_values(X_scaled[idx:idx+1])

        explanation = shap.Explanation(
            values        = shap_values[0],
            base_values   = explainer.expected_value,
            data          = X_scaled[idx],
            feature_names = feature_names
        )

        fig_wf, ax_wf = plt.subplots(figsize=(10, 6))
        shap.waterfall_plot(explanation, show=False, max_display=12)
        plt.tight_layout()
        st.pyplot(plt.gcf())
        plt.close()

        # Show raw feature values
        st.markdown("### Customer Feature Values")
        if df_raw is not None:
            row_display = df_raw.iloc[idx].to_frame(name='Value')
            st.dataframe(row_display, use_container_width=True)

    # ── PAGE: What-If Simulator ───────────────────────────────────────────────
    elif page == "🎛 What-If Simulator":
        st.title("🎛 What-If Simulator")
        st.markdown("Change a customer's features and see how the predicted churn probability changes in real time.")

        if df_results is None:
            st.warning("Please upload a CSV file first.")
            return

        customer_ids = df_results['customerID'].tolist()
        selected_id  = st.selectbox("Select a customer to simulate", customer_ids)
        idx = df_results.index[df_results['customerID'] == selected_id].tolist()[0]

        original_prob = probs[idx]
        st.metric("Original Churn Probability", f"{original_prob*100:.1f}%", help="Before any changes")

        st.markdown("### Adjust Customer Features")
        st.markdown("Use the sliders and dropdowns to simulate changes, then click Predict.")

        col1, col2 = st.columns(2)

        with col1:
            new_tenure = st.slider(
                "Tenure (months)",
                min_value=0, max_value=72,
                value=int(df_raw.iloc[idx].get('tenure', 12)) if df_raw is not None else 12
            )
            new_monthly = st.slider(
                "Monthly Charges ($)",
                min_value=18.0, max_value=120.0, step=0.5,
                value=float(df_raw.iloc[idx].get('MonthlyCharges', 65)) if df_raw is not None else 65.0
            )
            new_contract = st.selectbox(
                "Contract Type",
                ["Month-to-month", "One year", "Two year"],
                index=0
            )

        with col2:
            new_internet = st.selectbox(
                "Internet Service",
                ["DSL", "Fiber optic", "No"],
                index=1
            )
            new_techsupport = st.selectbox(
                "Tech Support",
                ["Yes", "No", "No internet service"],
                index=1
            )
            new_paperless = st.selectbox(
                "Paperless Billing",
                ["Yes", "No"],
                index=0
            )

        if st.button("🔮 Predict with New Values", use_container_width=True):
            # Build a modified copy of this customer's row
            if df_raw is not None:
                modified_row = df_raw.iloc[idx:idx+1].copy()
                modified_row['tenure']          = new_tenure
                modified_row['MonthlyCharges']  = new_monthly
                modified_row['Contract']        = new_contract
                modified_row['InternetService'] = new_internet
                modified_row['TechSupport']     = new_techsupport
                modified_row['PaperlessBilling'] = new_paperless
                # Recompute TotalCharges as a rough estimate
                modified_row['TotalCharges'] = new_tenure * new_monthly

                X_mod_scaled, _ = preprocess_csv(modified_row, scaler, feature_names)
                new_prob = model.predict_proba(X_mod_scaled)[0, 1]

                st.markdown("---")
                c1, c2, c3 = st.columns(3)
                c1.metric("Before", f"{original_prob*100:.1f}%")
                c2.metric("After",  f"{new_prob*100:.1f}%",
                          delta=f"{(new_prob - original_prob)*100:+.1f}%",
                          delta_color="inverse")
                c3.metric("Risk Level (After)", risk_label(new_prob))

                if new_prob < original_prob:
                    st.success(f"✅ These changes would reduce churn probability by {(original_prob - new_prob)*100:.1f} percentage points.")
                else:
                    st.warning(f"⚠️ These changes would increase churn probability by {(new_prob - original_prob)*100:.1f} percentage points.")

    # ── PAGE: Export Report ───────────────────────────────────────────────────
    elif page == "📤 Export Report":
        st.title("📤 Export Predictions Report")

        if df_results is None:
            st.warning("Please upload a CSV file first.")
            return

        st.markdown(f"Ready to export predictions for **{len(df_results):,} customers**.")

        # Summary stats before downloading
        col1, col2, col3 = st.columns(3)
        col1.metric("High Risk",   int((probs >= 0.7).sum()))
        col2.metric("Medium Risk", int(((probs >= 0.4) & (probs < 0.7)).sum()))
        col3.metric("Low Risk",    int((probs < 0.4).sum()))

        # Build CSV in memory
        export_df = df_results[['customerID', 'Churn Probability', 'Prediction', 'Risk Level']].copy()
        csv_buffer = io.StringIO()
        export_df.to_csv(csv_buffer, index=False)

        st.download_button(
            label="⬇️ Download Predictions as CSV",
            data=csv_buffer.getvalue(),
            file_name="churnx_predictions.csv",
            mime="text/csv",
            use_container_width=True
        )

        # Show a preview
        st.dataframe(export_df.head(20), use_container_width=True)


# ── Entry point ───────────────────────────────────────────────────────────────
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if not st.session_state['logged_in']:
    login_page()
else:
    main_app()
