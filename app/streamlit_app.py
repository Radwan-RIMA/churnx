"""
ChurnX 2.0 — AI Customer Retention System
Run with: streamlit run app/streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import shap
import joblib
import os
import io

try:
    from openai import OpenAI
    OPENAI_SDK_AVAILABLE = True
except ImportError:
    OPENAI_SDK_AVAILABLE = False

# Provider config: Groq and DeepSeek are both OpenAI-compatible.
LLM_PROVIDERS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "model":    "llama-3.3-70b-versatile",
        "key_name": "GROQ_API_KEY",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "model":    "deepseek-chat",
        "key_name": "DEEPSEEK_API_KEY",
    },
}

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnX 2.0",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, '..', 'models')
DEMO_CSV   = os.path.join(BASE_DIR, '..', 'data', 'raw', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')


# ── Load model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model         = joblib.load(os.path.join(MODELS_DIR, 'best_model.joblib'))
    scaler        = joblib.load(os.path.join(MODELS_DIR, 'scaler.joblib'))
    feature_names = joblib.load(os.path.join(MODELS_DIR, 'feature_names.joblib'))
    # Compatibility shim: models saved with older XGBoost lack attributes the
    # newer installed version expects (e.g. use_label_encoder). Restore them.
    for attr, default in (('use_label_encoder', False),
                          ('gpu_id', None),
                          ('predictor', None),
                          ('enable_categorical', False)):
        if not hasattr(model, attr):
            setattr(model, attr, default)
    explainer     = shap.TreeExplainer(model)
    return model, scaler, feature_names, explainer


# ── Preprocessing ──────────────────────────────────────────────────────────────
def preprocess_csv(df_raw, scaler, feature_names):
    df = df_raw.copy()
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    if 'Churn' in df.columns:
        df = df.drop(columns=['Churn'])

    service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                    'TechSupport', 'StreamingTV', 'StreamingMovies']
    df['service_count']        = df[service_cols].apply(lambda r: sum(r == 'Yes'), axis=1)
    df['charge_per_tenure']    = df['MonthlyCharges'] / (df['tenure'] + 1)
    df['has_premium_services'] = (df['service_count'] >= 3).astype(int)

    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling'] + service_cols
    for col in binary_cols:
        # Encode whenever the column isn't already numeric (handles object AND
        # pyarrow/string dtypes, which the old `== object` check missed).
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].map(lambda x: 1 if x in ['Yes', 'Female'] else 0)

    multi_cat = ['MultipleLines', 'InternetService', 'Contract', 'PaymentMethod']
    # NOTE: do NOT use drop_first here. On a single-row input (What-If Simulator)
    # the chosen category would be the only value present and drop_first would
    # delete it, silently reverting the change to the baseline category. Instead
    # we create all dummy columns and let the feature_names alignment below drop
    # the baseline ones — correct for both single rows and full datasets.
    df = pd.get_dummies(df, columns=[c for c in multi_cat if c in df.columns])

    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]
    # Safety net: force everything numeric before scaling (dummies may be bool,
    # and any stray non-numeric value becomes 0 instead of crashing the scaler).
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0).astype(float)
    return scaler.transform(df), df


# Rendering every customerID as an <option> is what makes the dropdown feel
# unresponsive on datasets with thousands of rows (the widget isn't virtualized,
# so a click has to build the whole list in the DOM before it opens). Filtering
# down with a search box first keeps that list small.
MAX_UNFILTERED_OPTIONS = 500


def customer_selectbox(label, df_results, key):
    all_ids = df_results['customerID'].tolist()
    search = st.text_input(
        "Search customer ID",
        key=f"{key}_search",
        placeholder=f"Type to filter {len(all_ids):,} customers...",
    )
    if search:
        ids = [c for c in all_ids if search.lower() in str(c).lower()]
        if not ids:
            st.warning(f"No customers match '{search}'.")
            st.stop()
    elif len(all_ids) > MAX_UNFILTERED_OPTIONS:
        st.caption(
            f"Showing first {MAX_UNFILTERED_OPTIONS:,} of {len(all_ids):,} customers — "
            "search above to reach the rest."
        )
        ids = all_ids[:MAX_UNFILTERED_OPTIONS]
    else:
        ids = all_ids
    return st.selectbox(label, ids, key=key)


def risk_label(prob):
    if prob >= 0.7:   return "🔴 High"
    elif prob >= 0.4: return "🟡 Medium"
    else:             return "🟢 Low"


def get_llm_config():
    """Return (provider_name, api_key, base_url, model) for the configured LLM provider."""
    try:
        provider = st.secrets.get("LLM_PROVIDER", "groq").lower()
    except Exception:
        provider = os.environ.get("LLM_PROVIDER", "groq").lower()

    cfg = LLM_PROVIDERS.get(provider, LLM_PROVIDERS["groq"])
    key_name = cfg["key_name"]

    try:
        key = st.secrets.get(key_name, "")
    except Exception:
        key = ""
    if not key or key.startswith("your-"):
        key = os.environ.get(key_name, "")
    if key.startswith("your-"):
        key = ""

    return provider, key, cfg["base_url"], cfg["model"]


# ── Run predictions on a dataframe ─────────────────────────────────────────────
def run_predictions(df_raw, scaler, feature_names, model):
    X_scaled, X_aligned = preprocess_csv(df_raw, scaler, feature_names)
    probs = model.predict_proba(X_scaled)[:, 1]
    preds = (probs >= 0.5).astype(int)

    df_results = df_raw.copy()
    if 'customerID' not in df_results.columns:
        df_results.insert(0, 'customerID', [f'CUST-{i:04d}' for i in range(len(df_results))])
    df_results['Churn Probability'] = (probs * 100).round(1)
    df_results['Prediction']        = preds
    df_results['Risk Level']        = pd.Series(probs).apply(risk_label).values
    return df_results, X_scaled, X_aligned, probs


# ── Login ──────────────────────────────────────────────────────────────────────
def login_page():
    st.title("🔐 ChurnX 2.0")
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
                st.error("Wrong credentials. Try: admin / churnx123")
        st.caption("Demo credentials: admin / churnx123")


# ── Main app ───────────────────────────────────────────────────────────────────
def main_app():
    model, scaler, feature_names, explainer = load_model()

    # Sidebar
    st.sidebar.image("https://img.icons8.com/color/96/graph.png", width=60)
    st.sidebar.title("ChurnX 2.0")
    st.sidebar.markdown("---")
    page = st.sidebar.radio(
        "Navigate",
        ["🏠 Dashboard", "🔍 Customer Lookup", "🤖 Retention Copilot",
         "🎛 What-If Simulator", "📤 Export Report"]
    )
    st.sidebar.markdown("---")

    uploaded_file = st.sidebar.file_uploader("Upload customer CSV", type=["csv"])
    if st.sidebar.button("Logout"):
        st.session_state['logged_in'] = False
        st.rerun()

    # ── Load data: uploaded or demo ───────────────────────────────────────────
    is_demo = False

    if uploaded_file:
        df_raw = pd.read_csv(uploaded_file)
        df_results, X_scaled, X_aligned, probs = run_predictions(df_raw, scaler, feature_names, model)
        st.session_state.update({
            'df_results': df_results, 'X_scaled': X_scaled,
            'X_aligned': X_aligned, 'probs': probs, 'df_raw': df_raw, 'is_demo': False
        })

    if 'df_results' not in st.session_state and os.path.exists(DEMO_CSV):
        df_raw = pd.read_csv(DEMO_CSV)
        df_results, X_scaled, X_aligned, probs = run_predictions(df_raw, scaler, feature_names, model)
        st.session_state.update({
            'df_results': df_results, 'X_scaled': X_scaled,
            'X_aligned': X_aligned, 'probs': probs, 'df_raw': df_raw, 'is_demo': True
        })

    df_results = st.session_state.get('df_results')
    X_scaled   = st.session_state.get('X_scaled')
    X_aligned  = st.session_state.get('X_aligned')
    probs      = st.session_state.get('probs')
    df_raw     = st.session_state.get('df_raw')
    is_demo    = st.session_state.get('is_demo', False)

    if is_demo:
        st.info("🎯 **Demo mode** — showing Telco dataset. Upload your own CSV in the sidebar to replace it.")

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE: Dashboard
    # ══════════════════════════════════════════════════════════════════════════
    if page == "🏠 Dashboard":
        st.title("📊 ChurnX 2.0 — Customer Retention Intelligence")

        if df_results is None:
            st.info("Upload a customer CSV in the sidebar to get started.")
            return

        total     = len(df_results)
        churned   = int((df_results['Prediction'] == 1).sum())
        high_risk = int((probs >= 0.7).sum())
        avg_prob  = probs.mean() * 100

        churner_mask = df_results['Prediction'] == 1
        monthly_at_risk = 0.0
        if 'MonthlyCharges' in df_raw.columns:
            monthly_at_risk = df_raw.loc[churner_mask.values, 'MonthlyCharges'].sum()

        # KPI row
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Customers",      f"{total:,}")
        c2.metric("Predicted Churners",   f"{churned:,}",
                  delta=f"{churned/total*100:.1f}%", delta_color="inverse")
        c3.metric("High Risk",            f"{high_risk:,}")
        c4.metric("Avg Churn Probability", f"{avg_prob:.1f}%")
        c5.metric("Monthly Revenue at Risk",
                  f"${monthly_at_risk:,.0f}",
                  delta="per month", delta_color="off")

        st.markdown("---")

        col_l, col_r = st.columns(2)

        # Risk donut chart
        with col_l:
            st.markdown("#### Risk Distribution")
            n_high   = int((probs >= 0.7).sum())
            n_medium = int(((probs >= 0.4) & (probs < 0.7)).sum())
            n_low    = int((probs < 0.4).sum())
            fig_donut = go.Figure(go.Pie(
                labels=['High Risk', 'Medium Risk', 'Low Risk'],
                values=[n_high, n_medium, n_low],
                hole=0.55,
                marker_colors=['#F44336', '#FF9800', '#4CAF50'],
                textinfo='label+percent',
                hovertemplate='%{label}: %{value:,} customers<extra></extra>'
            ))
            fig_donut.update_layout(
                showlegend=False,
                margin=dict(t=10, b=10, l=10, r=10),
                height=280,
                annotations=[dict(text=f"{churned/total*100:.0f}%<br>churn", x=0.5, y=0.5,
                                  font_size=18, showarrow=False)]
            )
            st.plotly_chart(fig_donut, use_container_width=True)

        # Global SHAP bar chart (sample 400 for speed)
        with col_r:
            st.markdown("#### Top Churn Drivers (Global SHAP)")
            if 'global_shap' not in st.session_state:
                sample_n = min(400, len(X_scaled))
                idx_sample = np.random.choice(len(X_scaled), sample_n, replace=False)
                shap_sample = explainer.shap_values(X_scaled[idx_sample])
                mean_abs_shap = np.abs(shap_sample).mean(axis=0)
                st.session_state['global_shap'] = mean_abs_shap

            mean_abs_shap = st.session_state['global_shap']
            top_n = 10
            top_idx = np.argsort(mean_abs_shap)[-top_n:]
            top_features = [feature_names[i] for i in top_idx]
            top_values   = mean_abs_shap[top_idx]

            fig_bar = go.Figure(go.Bar(
                x=top_values,
                y=top_features,
                orientation='h',
                marker_color='#F44336',
            ))
            fig_bar.update_layout(
                xaxis_title="Mean |SHAP value|",
                margin=dict(t=10, b=10, l=10, r=10),
                height=280,
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("---")

        # Churn probability histogram
        st.markdown("#### Churn Probability Distribution")
        fig_hist, ax = plt.subplots(figsize=(10, 3))
        ax.hist(probs, bins=40, color='#F44336', alpha=0.7, edgecolor='white')
        ax.axvline(0.5, color='black',  linestyle='--', linewidth=1.5, label='Decision threshold (0.5)')
        ax.axvline(0.7, color='orange', linestyle='--', linewidth=1.5, label='High risk threshold (0.7)')
        ax.set_xlabel('Churn Probability')
        ax.set_ylabel('Number of Customers')
        ax.legend()
        st.pyplot(fig_hist)
        plt.close()

        # Top 10 highest risk customers
        st.markdown("#### Top 10 Customers to Act On Now")
        top10_idx  = np.argsort(probs)[-10:][::-1]
        top10      = df_results.iloc[top10_idx][['customerID', 'Churn Probability', 'Risk Level']].copy()
        if 'MonthlyCharges' in df_raw.columns:
            top10['Monthly Charges'] = df_raw.iloc[top10_idx]['MonthlyCharges'].values
        if 'Contract' in df_raw.columns:
            top10['Contract'] = df_raw.iloc[top10_idx]['Contract'].values
        st.dataframe(top10.reset_index(drop=True), use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE: Customer Lookup
    # ══════════════════════════════════════════════════════════════════════════
    elif page == "🔍 Customer Lookup":
        st.title("🔍 Individual Customer Explanation")

        if df_results is None:
            st.warning("Please upload a CSV file first.")
            return

        selected_id = customer_selectbox("Select a customer", df_results, key="lookup_customer")
        idx  = df_results.index[df_results['customerID'] == selected_id].tolist()[0]
        prob = probs[idx]

        col1, col2, col3 = st.columns(3)
        col1.metric("Customer ID",        selected_id)
        col2.metric("Churn Probability",  f"{prob*100:.1f}%")
        col3.metric("Risk Level",         risk_label(prob))

        st.progress(float(prob), text=f"Churn Risk: {prob*100:.1f}%")
        st.markdown("---")

        st.markdown("### Why is this customer at risk?")
        st.caption("Red bars = features pushing toward churn | Blue bars = features reducing churn risk")

        shap_values = explainer.shap_values(X_scaled[idx:idx+1])
        explanation = shap.Explanation(
            values        = shap_values[0],
            base_values   = explainer.expected_value,
            data          = X_scaled[idx],
            feature_names = feature_names
        )
        shap.waterfall_plot(explanation, show=False, max_display=12)
        plt.tight_layout()
        st.pyplot(plt.gcf())
        plt.close()

        st.markdown("### Customer Feature Values")
        if df_raw is not None:
            st.dataframe(df_raw.iloc[idx].astype(str).to_frame(name='Value'), use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE: Retention Copilot
    # ══════════════════════════════════════════════════════════════════════════
    elif page == "🤖 Retention Copilot":
        st.title("🤖 Retention Copilot")
        st.markdown("AI-generated, customer-specific retention strategies powered by an LLM.")

        if df_results is None:
            st.warning("Please upload a CSV file first.")
            return

        provider, api_key, base_url, model_name = get_llm_config()
        if not OPENAI_SDK_AVAILABLE:
            st.error("The `openai` package is not installed. Run: `pip install openai`")
            return
        st.caption(f"LLM provider: **{provider}** · model: `{model_name}`")

        col_sel, col_risk = st.columns([3, 1])
        with col_sel:
            selected_id = customer_selectbox("Select a customer", df_results, key="copilot_customer")
        idx  = df_results.index[df_results['customerID'] == selected_id].tolist()[0]
        prob = probs[idx]

        with col_risk:
            st.metric("Churn Probability", f"{prob*100:.1f}%")

        # Customer profile card
        row = df_raw.iloc[idx]
        st.markdown("### Customer Profile")
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Tenure",           f"{row.get('tenure', 'N/A')} months")
        p2.metric("Monthly Charges",  f"${row.get('MonthlyCharges', 0):.2f}")
        p3.metric("Contract",         str(row.get('Contract', 'N/A')))
        p4.metric("Internet Service", str(row.get('InternetService', 'N/A')))

        # Compute SHAP for this customer and extract top drivers
        shap_vals = explainer.shap_values(X_scaled[idx:idx+1])[0]
        shap_series = pd.Series(shap_vals, index=feature_names)
        top_positive = shap_series.nlargest(5)   # pushing toward churn
        top_negative = shap_series.nsmallest(3)  # reducing churn risk

        st.markdown("### Risk Drivers (SHAP)")
        d1, d2 = st.columns(2)
        with d1:
            st.markdown("**Pushing toward churn ↑**")
            for feat, val in top_positive.items():
                st.markdown(f"- `{feat}`: +{val:.3f}")
        with d2:
            st.markdown("**Reducing churn risk ↓**")
            for feat, val in top_negative.items():
                st.markdown(f"- `{feat}`: {val:.3f}")

        st.markdown("---")

        # Build SHAP driver text for the prompt
        shap_driver_lines = "\n".join(
            [f"  - {feat} (SHAP: +{val:.3f}, pushing toward churn)"
             for feat, val in top_positive.items()] +
            [f"  - {feat} (SHAP: {val:.3f}, reducing risk)"
             for feat, val in top_negative.items()]
        )

        prompt = f"""You are a customer retention specialist at a telecom company.

Customer profile:
- Tenure: {row.get('tenure', 'N/A')} months
- Monthly charges: ${row.get('MonthlyCharges', 0):.2f}
- Contract: {row.get('Contract', 'N/A')}
- Tech Support: {row.get('TechSupport', 'N/A')}
- Internet Service: {row.get('InternetService', 'N/A')}
- Churn probability: {prob*100:.1f}%

Top reasons they are predicted to churn (SHAP values):
{shap_driver_lines}

Write a retention strategy with:
1. Plain English explanation of why this customer is at risk
2. Three specific actions the retention team should take this week
3. Expected impact if actions are taken

Be specific. Use the actual numbers. Do not be generic."""

        if not api_key:
            key_name = LLM_PROVIDERS.get(provider, LLM_PROVIDERS["groq"])["key_name"]
            st.warning(
                f"No API key found for **{provider}**. Add it to `.streamlit/secrets.toml` as "
                f"`{key_name} = \"...\"` (and set `LLM_PROVIDER = \"{provider}\"`)."
            )
            with st.expander("Preview prompt that will be sent to the LLM"):
                st.code(prompt, language="text")
        else:
            if st.button("🚀 Generate Retention Strategy", use_container_width=True, type="primary"):
                with st.spinner(f"{provider.capitalize()} is analyzing this customer..."):
                    try:
                        client = OpenAI(api_key=api_key, base_url=base_url)
                        response = client.chat.completions.create(
                            model=model_name,
                            max_tokens=1024,
                            messages=[{"role": "user", "content": prompt}]
                        )
                        strategy = response.choices[0].message.content
                        st.session_state[f'strategy_{selected_id}'] = strategy
                    except Exception as e:
                        st.error(f"API call failed: {e}")

            if f'strategy_{selected_id}' in st.session_state:
                st.markdown("### Retention Strategy")
                st.markdown(st.session_state[f'strategy_{selected_id}'])

                col_copy, col_export = st.columns(2)
                with col_export:
                    strategy_text = st.session_state[f'strategy_{selected_id}']
                    st.download_button(
                        "⬇️ Download Strategy",
                        data=strategy_text,
                        file_name=f"retention_{selected_id}.txt",
                        mime="text/plain"
                    )

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE: What-If Simulator
    # ══════════════════════════════════════════════════════════════════════════
    elif page == "🎛 What-If Simulator":
        st.title("🎛 What-If Simulator")
        st.markdown("Change a customer's features and see how churn probability shifts in real time.")

        if df_results is None:
            st.warning("Please upload a CSV file first.")
            return

        selected_id   = customer_selectbox("Select a customer to simulate", df_results, key="whatif_customer")
        idx           = df_results.index[df_results['customerID'] == selected_id].tolist()[0]
        original_prob = probs[idx]

        st.metric("Original Churn Probability", f"{original_prob*100:.1f}%")
        st.markdown("### Adjust Customer Features")

        col1, col2 = st.columns(2)
        with col1:
            new_tenure = st.slider(
                "Tenure (months)", 0, 72,
                int(df_raw.iloc[idx].get('tenure', 12)) if df_raw is not None else 12
            )
            new_monthly = st.slider(
                "Monthly Charges ($)", 18.0, 120.0, step=0.5,
                value=float(df_raw.iloc[idx].get('MonthlyCharges', 65)) if df_raw is not None else 65.0
            )
            new_contract = st.selectbox(
                "Contract Type",
                ["Month-to-month", "One year", "Two year"]
            )

        with col2:
            new_internet = st.selectbox(
                "Internet Service", ["DSL", "Fiber optic", "No"], index=1
            )
            new_techsupport = st.selectbox(
                "Tech Support", ["Yes", "No", "No internet service"], index=1
            )
            new_paperless = st.selectbox("Paperless Billing", ["Yes", "No"])

        if st.button("🔮 Predict with New Values", use_container_width=True):
            if df_raw is not None:
                modified_row = df_raw.iloc[idx:idx+1].copy()
                modified_row['tenure']           = new_tenure
                modified_row['MonthlyCharges']   = new_monthly
                modified_row['Contract']         = new_contract
                modified_row['InternetService']  = new_internet
                modified_row['TechSupport']      = new_techsupport
                modified_row['PaperlessBilling'] = new_paperless
                modified_row['TotalCharges']     = new_tenure * new_monthly

                X_mod_scaled, _ = preprocess_csv(modified_row, scaler, feature_names)
                new_prob = model.predict_proba(X_mod_scaled)[0, 1]

                st.markdown("---")
                c1, c2, c3 = st.columns(3)
                c1.metric("Before", f"{original_prob*100:.1f}%")
                c2.metric("After",  f"{new_prob*100:.1f}%",
                          delta=f"{(new_prob - original_prob)*100:+.1f}%",
                          delta_color="inverse")
                c3.metric("Risk After", risk_label(new_prob))

                if new_prob < original_prob:
                    st.success(f"✅ These changes reduce churn probability by {(original_prob - new_prob)*100:.1f} pp.")
                else:
                    st.warning(f"⚠️ These changes increase churn probability by {(new_prob - original_prob)*100:.1f} pp.")

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE: Export Report
    # ══════════════════════════════════════════════════════════════════════════
    elif page == "📤 Export Report":
        st.title("📤 Export Predictions Report")

        if df_results is None:
            st.warning("Please upload a CSV file first.")
            return

        # Filter options
        risk_filter = st.selectbox("Filter by Risk Level", ["All", "High", "Medium", "Low"])
        if risk_filter == "High":
            mask = probs >= 0.7
        elif risk_filter == "Medium":
            mask = (probs >= 0.4) & (probs < 0.7)
        elif risk_filter == "Low":
            mask = probs < 0.4
        else:
            mask = np.ones(len(probs), dtype=bool)
        filtered       = df_results[mask]
        filtered_probs = probs[mask]

        # Summary
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Customers in Filter", f"{len(filtered):,}")
        col2.metric("Avg Churn Prob",       f"{filtered_probs.mean()*100:.1f}%" if len(filtered) else "—")

        revenue_at_risk = 0.0
        if 'MonthlyCharges' in df_raw.columns and len(filtered) > 0:
            revenue_at_risk = df_raw.loc[mask, 'MonthlyCharges'].sum()
        col3.metric("Monthly Revenue at Risk", f"${revenue_at_risk:,.0f}")
        col4.metric("Annual Revenue at Risk",  f"${revenue_at_risk*12:,.0f}")

        # Build export CSV
        export_df = filtered[['customerID', 'Churn Probability', 'Prediction', 'Risk Level']].copy()
        if 'MonthlyCharges' in df_raw.columns:
            export_df['Monthly Charges'] = df_raw.loc[mask, 'MonthlyCharges'].values
        if 'Contract' in df_raw.columns:
            export_df['Contract'] = df_raw.loc[mask, 'Contract'].values

        csv_buffer = io.StringIO()
        export_df.to_csv(csv_buffer, index=False)

        st.download_button(
            label=f"⬇️ Download {len(filtered):,} Records as CSV",
            data=csv_buffer.getvalue(),
            file_name=f"churnx_predictions_{risk_filter.lower()}.csv",
            mime="text/csv",
            use_container_width=True
        )

        st.dataframe(export_df.head(30), use_container_width=True)


# ── Entry point ────────────────────────────────────────────────────────────────
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if not st.session_state['logged_in']:
    login_page()
else:
    main_app()
