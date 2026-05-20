"""
ChurnX — Customer Churn Prediction Dashboard v4
Run with: streamlit run streamlit_app.py

Upgrades from v3:
- Live What-If simulator (probability updates instantly on slider change)
- SHAP values cached after first computation — no re-calculation on customer switch
- Dashboard: risk breakdown donut + top 10 customers to call table
- Export: filter by risk level before downloading
- Full design overhaul: dark premium theme, custom CSS, refined typography
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap
import joblib
import os
import io


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnX",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS — sharp premium dark design ────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&family=Inter:wght@300;400;500&display=swap');

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    font-size: 14px;
}

/* ── App background with subtle grid ── */
.stApp {
    background-color: #070b14;
    background-image:
        linear-gradient(rgba(59,130,246,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(59,130,246,0.03) 1px, transparent 1px);
    background-size: 40px 40px;
    color: #cbd5e1;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 2rem !important;
    max-width: 1400px !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0b1020 !important;
    border-right: 1px solid #1a2540 !important;
    padding-top: 0 !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.5rem;
}
/* Hide default radio dots */
[data-testid="stSidebar"] .stRadio > div {
    gap: 2px !important;
}
[data-testid="stSidebar"] .stRadio label {
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    font-weight: 400 !important;
    color: #64748b !important;
    padding: 8px 12px !important;
    border-radius: 8px !important;
    cursor: pointer !important;
    transition: all 0.15s !important;
    border: none !important;
    width: 100% !important;
    display: block !important;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background: #111827 !important;
    color: #94a3b8 !important;
}
[data-testid="stSidebar"] [aria-checked="true"] + label,
[data-testid="stSidebar"] .stRadio [data-checked="true"] label {
    background: #1e3a5f !important;
    color: #60a5fa !important;
}
/* Kill radio circle indicators */
[data-testid="stSidebar"] .stRadio [role="radio"] {
    display: none !important;
}
[data-testid="stSidebar"] p, [data-testid="stSidebar"] span,
[data-testid="stSidebar"] div { color: #64748b; }
[data-testid="stSidebar"] strong { color: #94a3b8 !important; }

/* ── Dividers ── */
hr { border: none !important; border-top: 1px solid #1a2540 !important; margin: 1.5rem 0 !important; }

/* ── Page titles ── */
h1 {
    font-family: 'Syne', sans-serif !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
    color: #f8fafc !important;
    letter-spacing: -0.5px !important;
    margin-bottom: 0.25rem !important;
}
h2 {
    font-family: 'Syne', sans-serif !important;
    font-size: 1.25rem !important;
    font-weight: 600 !important;
    color: #e2e8f0 !important;
    letter-spacing: -0.3px !important;
}
h3 {
    font-family: 'Inter', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 500 !important;
    color: #94a3b8 !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
}
h4 { font-family: 'Inter', sans-serif !important; color: #64748b !important; font-size: 11px !important; text-transform: uppercase; letter-spacing: 1.5px; }

/* ── Metric cards — full custom ── */
[data-testid="metric-container"] {
    background: #0d1525 !important;
    border: 1px solid #1a2540 !important;
    border-radius: 16px !important;
    padding: 1.25rem 1.5rem !important;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}
[data-testid="metric-container"]:hover {
    border-color: #2d4a7a !important;
}
[data-testid="metric-container"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #3b82f6, #6366f1);
    opacity: 0.6;
}
[data-testid="stMetricLabel"] {
    font-family: 'Inter', sans-serif !important;
    font-size: 11px !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 1.2px !important;
    color: #475569 !important;
}
[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 2.2rem !important;
    font-weight: 500 !important;
    color: #f1f5f9 !important;
    line-height: 1.1 !important;
}
[data-testid="stMetricDelta"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid #1a2540 !important;
    border-radius: 12px !important;
    overflow: hidden !important;
    background: #0d1525 !important;
}

/* ── All buttons ── */
.stButton > button,
[data-testid="stDownloadButton"] > button {
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    background: linear-gradient(135deg, #1d4ed8, #4f46e5) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 10px 20px !important;
    letter-spacing: 0.3px !important;
    box-shadow: 0 4px 15px rgba(59,130,246,0.25) !important;
    transition: all 0.2s !important;
}
.stButton > button:hover,
[data-testid="stDownloadButton"] > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(59,130,246,0.35) !important;
    opacity: 1 !important;
}
/* Logout button — secondary style */
.stButton:last-child > button {
    background: transparent !important;
    border: 1px solid #1a2540 !important;
    color: #64748b !important;
    box-shadow: none !important;
}

/* ── Inputs / selects / sliders ── */
.stSelectbox > label, .stSlider > label, .stTextInput > label {
    font-family: 'Inter', sans-serif !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
    color: #475569 !important;
}
.stSelectbox [data-baseweb="select"] > div {
    background: #0d1525 !important;
    border: 1px solid #1a2540 !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
}
.stSelectbox [data-baseweb="select"] > div:hover {
    border-color: #3b82f6 !important;
}
.stTextInput > div > div > input {
    background: #0d1525 !important;
    border: 1px solid #1a2540 !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
}
.stTextInput > div > div > input:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.15) !important;
}
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background: #3b82f6 !important;
    border-color: #3b82f6 !important;
}

/* ── Alerts ── */
.stAlert {
    border-radius: 12px !important;
    border: 1px solid #1a2540 !important;
    font-size: 13px !important;
}
.stSuccess { background: rgba(16,185,129,0.08) !important; border-color: rgba(16,185,129,0.3) !important; }
.stWarning { background: rgba(245,158,11,0.08) !important; border-color: rgba(245,158,11,0.3) !important; }
.stInfo    { background: rgba(59,130,246,0.08) !important; border-color: rgba(59,130,246,0.3) !important; }
.stError   { background: rgba(239,68,68,0.08)  !important; border-color: rgba(239,68,68,0.3)  !important; }

/* ── Caption ── */
.stCaption, [data-testid="stCaptionContainer"] {
    color: #334155 !important;
    font-size: 11px !important;
    letter-spacing: 0.3px !important;
}

/* ── Progress bar ── */
.stProgress > div > div {
    background: #1a2540 !important;
    border-radius: 8px !important;
}
.stProgress > div > div > div {
    border-radius: 8px !important;
    background: linear-gradient(90deg, #3b82f6, #6366f1) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: #0d1525 !important;
    border: 1px dashed #1a2540 !important;
    border-radius: 12px !important;
    padding: 12px !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: #3b82f6 !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: #0d1525 !important;
    border: 1px solid #1a2540 !important;
    border-radius: 10px !important;
    color: #64748b !important;
    font-size: 13px !important;
}

/* ── Custom components ── */
.kpi-card {
    background: #0d1525;
    border: 1px solid #1a2540;
    border-radius: 16px;
    padding: 20px 22px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s, transform 0.2s;
}
.kpi-card:hover { border-color: #2d4a7a; transform: translateY(-2px); }
.kpi-card::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 3px;
    background: var(--accent, linear-gradient(90deg,#3b82f6,#6366f1));
    opacity: 0.5;
}
.kpi-label {
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #475569;
    margin-bottom: 8px;
}
.kpi-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2.4rem;
    font-weight: 500;
    color: #f8fafc;
    line-height: 1;
}
.kpi-sub {
    font-size: 11px;
    color: #334155;
    margin-top: 6px;
}
.section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 1.5rem 0 1rem 0;
}
.section-header .dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #3b82f6;
    flex-shrink: 0;
}
.section-header span {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #475569;
}
.badge-high   { background:rgba(239,68,68,0.12);  color:#f87171; border:1px solid rgba(239,68,68,0.25);  padding:2px 10px; border-radius:20px; font-size:11px; font-weight:600; display:inline-block; }
.badge-medium { background:rgba(245,158,11,0.12); color:#fbbf24; border:1px solid rgba(245,158,11,0.25); padding:2px 10px; border-radius:20px; font-size:11px; font-weight:600; display:inline-block; }
.badge-low    { background:rgba(16,185,129,0.12); color:#34d399; border:1px solid rgba(16,185,129,0.25); padding:2px 10px; border-radius:20px; font-size:11px; font-weight:600; display:inline-block; }
.prob-bar-track { background:#1a2540; border-radius:6px; height:8px; overflow:hidden; margin:10px 0; }
.prob-bar-fill  { height:100%; border-radius:6px; transition: width 0.4s ease; }
.page-subtitle  { font-size:13px; color:#475569; margin-top:-8px; margin-bottom:20px; }

/* ── Login ── */
.login-outer {
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 80vh;
}
.login-box {
    background: #0d1525;
    border: 1px solid #1a2540;
    border-radius: 24px;
    padding: 56px 48px;
    width: 100%;
    max-width: 420px;
    text-align: center;
}
.login-icon {
    font-size: 52px;
    margin-bottom: 12px;
    display: block;
}
.login-title {
    font-family: 'Syne', sans-serif;
    font-size: 28px;
    font-weight: 700;
    color: #f8fafc;
    margin-bottom: 4px;
}
.login-sub {
    font-size: 13px;
    color: #475569;
    margin-bottom: 36px;
}
</style>
""", unsafe_allow_html=True)

# ── Matplotlib dark theme ──────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#0d1525",
    "axes.facecolor":    "#0d1525",
    "axes.edgecolor":    "#1a2540",
    "axes.labelcolor":   "#64748b",
    "xtick.color":       "#475569",
    "ytick.color":       "#475569",
    "text.color":        "#cbd5e1",
    "grid.color":        "#1a2540",
    "grid.linewidth":    0.6,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.family":       "sans-serif",
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "legend.fontsize":   10,
    "legend.framealpha": 0.15,
    "legend.edgecolor":  "#1a2540",
})

MODELS_DIR  = os.path.join(os.path.dirname(__file__), '..', 'models')
REPORTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'reports')

# ── Load model (cached) ────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model         = joblib.load(os.path.join(MODELS_DIR, 'best_model.joblib'))
    scaler        = joblib.load(os.path.join(MODELS_DIR, 'scaler.joblib'))
    feature_names = joblib.load(os.path.join(MODELS_DIR, 'feature_names.joblib'))
    threshold_path = os.path.join(MODELS_DIR, 'best_threshold.joblib')
    best_threshold = float(joblib.load(threshold_path)) if os.path.exists(threshold_path) else 0.5
    explainer = shap.TreeExplainer(model)
    return model, scaler, feature_names, explainer, best_threshold

# ── Cache SHAP values for the full uploaded dataset ────────────────────────────
@st.cache_data(show_spinner="Computing SHAP values…")
def compute_all_shap(_explainer, X_scaled_tuple):
    """
    Computes SHAP values once for the full dataset and caches the result.
    X_scaled is passed as a tuple (hashable) to make st.cache_data work.
    """
    X_scaled = np.array(X_scaled_tuple)
    return _explainer.shap_values(X_scaled)

# ── Login ──────────────────────────────────────────────────────────────────────
def login_page():
    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        st.markdown("""
        <div style="text-align:center; padding: 60px 0 32px 0;">
            <div style="font-size:52px; margin-bottom:16px;">⚡</div>
            <div style="font-family:'Syne',sans-serif; font-size:32px; font-weight:700;
                        color:#f8fafc; letter-spacing:-1px; margin-bottom:6px;">ChurnX</div>
            <div style="font-size:13px; color:#475569; margin-bottom:40px;">
                Customer Churn Prediction Platform
            </div>
        </div>
        """, unsafe_allow_html=True)

        username = st.text_input("Username", placeholder="admin")
        password = st.text_input("Password", type="password", placeholder="••••••••")
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        if st.button("Sign In →", use_container_width=True):
            try:
                valid_user = st.secrets["auth"]["username"]
                valid_pass = st.secrets["auth"]["password"]
            except Exception:
                valid_user, valid_pass = "admin", "churnx123"
            if username == valid_user and password == valid_pass:
                st.session_state['logged_in'] = True
                st.rerun()
            else:
                st.error("Incorrect credentials.")
        st.markdown("""
        <div style="text-align:center; margin-top:24px; font-size:11px; color:#334155;">
            Contact your administrator for access credentials
        </div>
        """, unsafe_allow_html=True)

# ── Preprocessing ──────────────────────────────────────────────────────────────
EXPECTED_RAW_COLUMNS = [
    'tenure', 'MonthlyCharges', 'TotalCharges', 'gender',
    'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaperlessBilling', 'PaymentMethod',
]

def preprocess_csv(df_raw, scaler, feature_names):
    missing_cols = [c for c in EXPECTED_RAW_COLUMNS if c not in df_raw.columns]
    if missing_cols:
        st.warning(f"⚠️ Missing columns: {missing_cols}. Will be treated as zero/absent.")

    df = df_raw.copy()
    for drop_col in ['customerID', 'Churn']:
        if drop_col in df.columns:
            df = df.drop(columns=[drop_col])

    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    else:
        df['TotalCharges'] = 0

    service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                    'TechSupport', 'StreamingTV', 'StreamingMovies']

    df['service_count']    = df[service_cols].apply(lambda r: sum(str(v).strip().lower() == 'yes' for v in r), axis=1)
    df['charge_per_tenure'] = df['MonthlyCharges'] / (df['tenure'] + 1)
    df['new_and_monthly']  = ((df['tenure'] <= 12) & (df['Contract'].str.strip() == 'Month-to-month')).astype(int)

    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling'] + service_cols
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: 1 if str(x).strip().lower() in ['yes', '1', 'true'] else 0)

    if 'gender' in df.columns:
        df['gender_Male'] = df['gender'].apply(lambda x: 1 if str(x).strip().lower() == 'male' else 0)
        df = df.drop(columns=['gender'])

    multi_cat_map = {
        'MultipleLines':  ['MultipleLines_No phone service', 'MultipleLines_Yes'],
        'InternetService': ['InternetService_Fiber optic', 'InternetService_No'],
        'Contract':       ['Contract_One year', 'Contract_Two year'],
        'PaymentMethod':  ['PaymentMethod_Credit card (automatic)',
                           'PaymentMethod_Electronic check',
                           'PaymentMethod_Mailed check'],
    }
    for col, dummies in multi_cat_map.items():
        if col in df.columns:
            for dummy in dummies:
                value = dummy[len(col) + 1:]
                df[dummy] = (df[col].str.strip() == value).astype(int)
            df = df.drop(columns=[col])

    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]

    df_scaled = scaler.transform(df)
    return df_scaled, df


def risk_label(prob):
    if prob >= 0.7:   return "🔴 High"
    elif prob >= 0.4: return "🟡 Medium"
    else:             return "🟢 Low"

def risk_badge_html(prob):
    if prob >= 0.7:   return '<span class="badge-high">High Risk</span>'
    elif prob >= 0.4: return '<span class="badge-medium">Medium Risk</span>'
    else:             return '<span class="badge-low">Low Risk</span>'


def build_display_row(df_raw_row, feature_names):
    raw = df_raw_row
    display = {}
    service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                    'TechSupport', 'StreamingTV', 'StreamingMovies']
    for feat in feature_names:
        if feat in raw.index:
            display[feat] = raw[feat]
        elif feat == 'charge_per_tenure':
            display[feat] = raw['MonthlyCharges'] / (raw['tenure'] + 1)
        elif feat == 'new_and_monthly':
            display[feat] = int(raw['tenure'] <= 12 and str(raw.get('Contract', '')).strip() == 'Month-to-month')
        elif feat == 'service_count':
            display[feat] = sum(str(raw.get(c, '')).strip().lower() == 'yes' for c in service_cols)
        elif feat == 'gender_Male':
            display[feat] = 1 if str(raw.get('gender', '')).strip().lower() == 'male' else 0
        else:
            for base_col in ['MultipleLines', 'InternetService', 'Contract', 'PaymentMethod']:
                prefix = base_col + '_'
                if feat.startswith(prefix):
                    value = feat[len(prefix):]
                    display[feat] = int(str(raw.get(base_col, '')).strip() == value)
                    break
            else:
                display[feat] = int(str(raw.get(feat, 0)).strip().lower() in ['yes', '1', 'true', '1.0'])
    return pd.Series(display, index=feature_names)


# ── Main App ───────────────────────────────────────────────────────────────────
def main_app():
    model, scaler, feature_names, explainer, best_threshold = load_model()

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style="padding: 8px 4px 20px 4px;">
            <div style="display:flex; align-items:center; gap:10px; margin-bottom:4px;">
                <span style="font-size:22px;">⚡</span>
                <span style="font-family:'Syne',sans-serif; font-size:20px; font-weight:700;
                             color:#f8fafc; letter-spacing:-0.5px;">ChurnX</span>
            </div>
            <div style="font-size:10px; color:#334155; text-transform:uppercase;
                        letter-spacing:1.5px; padding-left:32px;">v4 · Prediction Platform</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")
        page = st.radio(
            "Navigation",
            ["🏠 Dashboard", "🔍 Customer Lookup", "🎛 What-If Simulator",
             "📤 Export Report"],
            label_visibility="collapsed"
        )
        st.markdown("---")
        st.markdown(f"""
        <div style="font-size:11px; color:#334155; line-height:1.8; padding:0 4px;">
            <div><span style="color:#475569;">Threshold</span> &nbsp;
                 <span style="font-family:'JetBrains Mono',monospace; color:#60a5fa;">
                 {best_threshold:.2f}</span></div>
            <div><span style="color:#475569;">Strategy</span> &nbsp;
                 <span style="color:#64748b;">cost-optimised</span></div>
            <div><span style="color:#475569;">Recall</span> &nbsp;
                 <span style="font-family:'JetBrains Mono',monospace; color:#34d399;">98.4%</span></div>
            <div><span style="color:#475569;">FN cost</span> &nbsp;
                 <span style="color:#f87171;">$780</span> &nbsp;·&nbsp;
                 <span style="color:#475569;">FP cost</span> &nbsp;
                 <span style="color:#94a3b8;">$30</span></div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")
        uploaded_file = st.file_uploader("Upload customer CSV", type=["csv"])
        st.markdown("---")
        if st.button("Logout", use_container_width=True):
            st.session_state['logged_in'] = False
            st.rerun()

    df_raw = df_results = None

    if uploaded_file:
        try:
            df_raw = pd.read_csv(uploaded_file)
            X_scaled, X_aligned = preprocess_csv(df_raw, scaler, feature_names)
            probs = model.predict_proba(X_scaled)[:, 1]
            preds = (probs >= best_threshold).astype(int)

            df_results = df_raw.copy()
            if 'customerID' not in df_results.columns:
                df_results.insert(0, 'customerID', [f'CUST-{i:04d}' for i in range(len(df_results))])

            df_results['Churn Probability'] = (probs * 100).round(1)
            df_results['Prediction']        = preds
            df_results['Risk Level']        = pd.Series(probs).apply(risk_label).values

            # Pre-compute SHAP for whole dataset (cached)
            shap_vals_all = compute_all_shap(explainer, tuple(map(tuple, X_scaled)))

            st.session_state.update({
                'df_results': df_results, 'X_scaled': X_scaled,
                'X_aligned': X_aligned, 'probs': probs,
                'df_raw': df_raw, 'shap_vals_all': shap_vals_all
            })
        except Exception as e:
            st.sidebar.error(f"Error: {e}")

    if 'df_results' in st.session_state:
        df_results    = st.session_state['df_results']
        X_scaled      = st.session_state['X_scaled']
        X_aligned     = st.session_state['X_aligned']
        probs         = st.session_state['probs']
        df_raw        = st.session_state.get('df_raw')
        shap_vals_all = st.session_state.get('shap_vals_all')

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE: Dashboard
    # ══════════════════════════════════════════════════════════════════════════
    if page == "🏠 Dashboard":
        st.title("Customer Churn Dashboard")
        st.markdown('<div class="page-subtitle">Real-time churn prediction and risk analysis</div>', unsafe_allow_html=True)

        if df_results is None:
            st.markdown("""
<div class="churn-card">
<h4>Getting started</h4>
<p style="color:#64748b; font-size:14px; margin:0;">
Upload a customer CSV in the sidebar to begin. The model will predict churn probability
for every customer and explain the key risk drivers.
</p>
</div>
""", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
**What this platform does:**
- Predicts churn probability using tuned XGBoost (AUC 80.45%)
- Explains each prediction with SHAP waterfall plots
- Simulates "what if we change their contract?" scenarios
- Exports a prioritised retention call list
""")
            with col2:
                st.markdown("""
**How to use:**
1. Upload the Telco Churn CSV in the sidebar
2. Review predictions and risk distribution here
3. Use Customer Lookup for individual SHAP explanations
4. Use What-If Simulator to test retention scenarios
5. Export filtered predictions for the retention team
""")
            return

        total     = len(df_results)
        churned   = int((df_results['Prediction'] == 1).sum())
        high_risk = int((probs >= 0.7).sum())
        med_risk  = int(((probs >= 0.4) & (probs < 0.7)).sum())
        low_risk  = int((probs < 0.4).sum())
        avg_prob  = probs.mean() * 100

        # KPI row
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Customers",       f"{total:,}")
        c2.metric("Predicted Churners",    f"{churned:,}", delta=f"{churned/total*100:.1f}%", delta_color="inverse")
        c3.metric("🔴 High Risk",           f"{high_risk:,}")
        c4.metric("🟡 Medium Risk",         f"{med_risk:,}")
        c5.metric("Avg Churn Probability", f"{avg_prob:.1f}%")

        st.markdown("---")

        # Charts row
        col_left, col_right = st.columns([1, 2])

        with col_left:
            st.markdown('<div class="section-header"><div class="dot"></div><span>Risk Breakdown</span></div>', unsafe_allow_html=True)
            fig_pie, ax_pie = plt.subplots(figsize=(4, 4))
            sizes  = [high_risk, med_risk, low_risk]
            colors = ["#ef4444", "#f59e0b", "#10b981"]
            labels = ["High", "Medium", "Low"]
            wedges, texts, autotexts = ax_pie.pie(
                sizes, labels=labels, colors=colors,
                autopct="%1.0f%%", startangle=90,
                wedgeprops=dict(width=0.55, edgecolor="#0a0e1a", linewidth=2),
                pctdistance=0.75
            )
            for at in autotexts:
                at.set_color("#f1f5f9")
                at.set_fontsize(11)
                at.set_fontweight("bold")
            for t in texts:
                t.set_color("#94a3b8")
                t.set_fontsize(11)
            ax_pie.set_facecolor("#0d1525")
            fig_pie.patch.set_facecolor("#0d1525")
            st.pyplot(fig_pie, use_container_width=True)
            plt.close()

        with col_right:
            st.markdown('<div class="section-header"><div class="dot"></div><span>Probability Distribution</span></div>', unsafe_allow_html=True)
            fig_hist, ax_hist = plt.subplots(figsize=(8, 4))
            ax_hist.hist(probs, bins=40, color="#3b82f6", alpha=0.8, edgecolor="#0a0e1a", linewidth=0.5)
            ax_hist.axvline(best_threshold, color="#ef4444", linestyle="--", linewidth=1.5,
                            label=f"Decision threshold ({best_threshold:.2f})")
            ax_hist.axvline(0.7, color="#f59e0b", linestyle="--", linewidth=1.5, label="High risk (0.70)")
            ax_hist.set_xlabel("Churn Probability")
            ax_hist.set_ylabel("Customers")
            ax_hist.legend(fontsize=10)
            ax_hist.grid(True, axis="y", alpha=0.4)
            fig_hist.patch.set_facecolor("#0d1525")
            st.pyplot(fig_hist, use_container_width=True)
            plt.close()

        st.markdown("---")

        # Top 10 customers to call
        st.markdown('<div class="section-header"><div class="dot" style="background:#ef4444"></div><span>Top 10 Customers to Call First</span></div>', unsafe_allow_html=True)
        st.caption("Highest predicted churn probability — prioritise these for retention outreach")

        top10 = df_results[['customerID', 'tenure', 'MonthlyCharges', 'Contract',
                             'Churn Probability', 'Risk Level']].copy()
        if 'Contract' not in top10.columns and df_raw is not None and 'Contract' in df_raw.columns:
            top10['Contract'] = df_raw['Contract'].values
        top10 = top10.sort_values('Churn Probability', ascending=False).head(10).reset_index(drop=True)
        top10.index = top10.index + 1
        st.dataframe(top10, use_container_width=True, height=380)

        st.markdown("---")

        # Full table with filter
        st.markdown('<div class="section-header"><div class="dot" style="background:#475569"></div><span>All Customers</span></div>', unsafe_allow_html=True)
        risk_filter = st.selectbox("Filter by risk level", ["All", "🔴 High", "🟡 Medium", "🟢 Low"])
        probs_series = pd.Series(probs)
        if "High" in risk_filter:      mask = probs_series >= 0.7
        elif "Medium" in risk_filter:  mask = (probs_series >= 0.4) & (probs_series < 0.7)
        elif "Low" in risk_filter:     mask = probs_series < 0.4
        else:                           mask = pd.Series([True] * len(df_results))

        filtered = df_results[mask.values].copy()
        st.caption(f"Showing {len(filtered):,} customers")
        display_cols = ['customerID', 'tenure', 'MonthlyCharges', 'Contract',
                        'Churn Probability', 'Risk Level']
        display_cols = [c for c in display_cols if c in filtered.columns]
        st.dataframe(filtered[display_cols].reset_index(drop=True), use_container_width=True, height=400)

        # Retention action recommendations
        if df_results is not None:
            st.markdown("---")
            st.markdown('<div class="section-header"><div class="dot" style="background:#10b981"></div><span>Recommended Retention Actions</span></div>', unsafe_allow_html=True)
            st.caption("Top churn drivers and suggested interventions — based on SHAP feature importance")

            action_path = os.path.join(REPORTS_DIR, 'retention_action_table.csv')
            if os.path.exists(action_path):
                st.dataframe(pd.read_csv(action_path), use_container_width=True)
            else:
                actions = pd.DataFrame({
                    'SHAP Rank': [1, 2, 3, 4, 5, 6],
                    'Feature':   ['charge_per_tenure', 'Contract (month-to-month)', 'InternetService (fiber)',
                                  'MonthlyCharges', 'tenure', 'PaymentMethod (e-check)'],
                    'Signal':    ['High cost/tenure ratio → high churn', 'No commitment → easy to leave',
                                  'Service quality issues', 'High bill → perceived poor value',
                                  'New customers most vulnerable', 'Proxy for low commitment'],
                    'Recommended Action': ['Loyalty discount at month 3 & 6', 'Offer 3 months free on annual upgrade',
                                           'Proactive quality check in months 1–6', 'Flag >$80/mo for retention call',
                                           'Onboarding programme months 1–12', 'Incentivise auto-pay switch'],
                })
                st.dataframe(actions, use_container_width=True, hide_index=True)

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE: Customer Lookup
    # ══════════════════════════════════════════════════════════════════════════
    elif page == "🔍 Customer Lookup":
        st.title("Customer Risk Explanation")
        st.markdown('<div class="page-subtitle">SHAP waterfall analysis — understand why each customer is at risk</div>', unsafe_allow_html=True)

        if df_results is None:
            st.warning("Please upload a CSV file first.")
            return

        customer_ids = df_results['customerID'].tolist()
        selected_id  = st.selectbox("Select customer", customer_ids)
        position     = customer_ids.index(selected_id)
        prob         = probs[position]

        col1, col2, col3 = st.columns(3)
        col1.metric("Customer ID",       selected_id)
        col2.metric("Churn Probability", f"{prob*100:.1f}%")
        col3.metric("Risk Level",        risk_label(prob))

        # Probability bar
        bar_color = "#ef4444" if prob >= 0.7 else "#f59e0b" if prob >= 0.4 else "#10b981"
        st.markdown(f"""
<div style="background:#1e2d4a; border-radius:8px; height:10px; margin:12px 0 24px 0; overflow:hidden;">
  <div style="background:{bar_color}; width:{prob*100:.1f}%; height:100%; border-radius:8px; transition:width 0.5s;"></div>
</div>
""", unsafe_allow_html=True)

        st.markdown("#### Why is this customer at risk?")
        st.caption("Red bars push toward churn · Blue bars reduce risk · Values shown are original (unscaled)")

        # Use cached SHAP values — no recomputation
        if shap_vals_all is not None:
            shap_row = shap_vals_all[position]
        else:
            shap_row = explainer.shap_values(X_scaled[position:position+1])[0]

        if df_raw is not None:
            display_data = build_display_row(df_raw.iloc[position], feature_names).values
        else:
            display_data = X_aligned.iloc[position].values

        explanation = shap.Explanation(
            values        = shap_row,
            base_values   = float(explainer.expected_value),
            data          = display_data,
            feature_names = feature_names
        )

        fig_wf, _ = plt.subplots(figsize=(10, 6))
        fig_wf.patch.set_facecolor("#0d1525")
        shap.waterfall_plot(explanation, show=False, max_display=12)
        plt.tight_layout()
        st.pyplot(plt.gcf())
        plt.close()

        if df_raw is not None:
            with st.expander("📋 Full customer feature values"):
                st.dataframe(df_raw.iloc[position].to_frame(name='Value'), use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE: What-If Simulator  (LIVE — updates on every slider change)
    # ══════════════════════════════════════════════════════════════════════════
    elif page == "🎛 What-If Simulator":
        st.title("What-If Simulator")
        st.markdown('<div class="page-subtitle">Adjust features and see churn probability update live</div>', unsafe_allow_html=True)

        if df_results is None:
            st.warning("Please upload a CSV file first.")
            return

        customer_ids  = df_results['customerID'].tolist()
        selected_id   = st.selectbox("Select customer to simulate", customer_ids)
        position      = customer_ids.index(selected_id)
        original_prob = probs[position]

        st.markdown("---")

        col_sliders, col_result = st.columns([1.4, 1])

        with col_sliders:
            st.markdown("#### Adjust Features")
            col1, col2 = st.columns(2)

            raw_row = df_raw.iloc[position] if df_raw is not None else {}

            with col1:
                new_tenure = st.slider(
                    "Tenure (months)", 0, 72,
                    int(raw_row.get('tenure', 12))
                )
                new_monthly = st.slider(
                    "Monthly Charges ($)", 18.0, 120.0,
                    float(raw_row.get('MonthlyCharges', 65.0)), step=0.5
                )
                new_contract = st.selectbox(
                    "Contract",
                    ["Month-to-month", "One year", "Two year"],
                    index=["Month-to-month", "One year", "Two year"].index(
                        raw_row.get('Contract', 'Month-to-month')
                    )
                )

            with col2:
                new_internet = st.selectbox(
                    "Internet Service",
                    ["DSL", "Fiber optic", "No"],
                    index=["DSL", "Fiber optic", "No"].index(
                        raw_row.get('InternetService', 'Fiber optic')
                    )
                )
                new_techsupport = st.selectbox(
                    "Tech Support",
                    ["Yes", "No", "No internet service"],
                    index=["Yes", "No", "No internet service"].index(
                        raw_row.get('TechSupport', 'No')
                    )
                )
                new_paperless = st.selectbox(
                    "Paperless Billing",
                    ["Yes", "No"],
                    index=["Yes", "No"].index(raw_row.get('PaperlessBilling', 'Yes'))
                )

        # ── LIVE prediction (no button needed) ────────────────────────────────
        with col_result:
            st.markdown("#### Live Result")

            if df_raw is not None:
                modified_row = df_raw.iloc[position:position+1].copy()
                modified_row['tenure']           = new_tenure
                modified_row['MonthlyCharges']   = new_monthly
                modified_row['Contract']         = new_contract
                modified_row['InternetService']  = new_internet
                modified_row['TechSupport']      = new_techsupport
                modified_row['PaperlessBilling'] = new_paperless
                modified_row['TotalCharges']     = new_tenure * new_monthly

                X_mod_scaled, _ = preprocess_csv(modified_row, scaler, feature_names)
                new_prob = model.predict_proba(X_mod_scaled)[0, 1]
                delta_pp = (new_prob - original_prob) * 100

                # Before / After metrics
                m1, m2 = st.columns(2)
                m1.metric("Before", f"{original_prob*100:.1f}%")
                m2.metric("After",  f"{new_prob*100:.1f}%",
                          delta=f"{delta_pp:+.1f}%", delta_color="inverse")

                # Visual probability bar
                bar_color = "#ef4444" if new_prob >= 0.7 else "#f59e0b" if new_prob >= 0.4 else "#10b981"
                st.markdown(f"""
<div style="background:#1e2d4a; border-radius:8px; height:12px; margin:12px 0; overflow:hidden;">
  <div style="background:{bar_color}; width:{new_prob*100:.1f}%; height:100%; border-radius:8px;"></div>
</div>
""", unsafe_allow_html=True)

                # Result message
                if abs(delta_pp) < 0.1:
                    st.info("No meaningful change.")
                elif new_prob < original_prob:
                    st.success(f"✅ Risk dropped by **{abs(delta_pp):.1f}pp** — {risk_label(new_prob)}")
                else:
                    st.warning(f"⚠️ Risk increased by **{delta_pp:.1f}pp** — {risk_label(new_prob)}")

                # Engineered feature changes
                st.markdown("**Engineered features:**")
                orig_cpt = float(raw_row.get('MonthlyCharges', 65)) / (float(raw_row.get('tenure', 12)) + 1)
                new_cpt  = new_monthly / (new_tenure + 1)
                orig_nm  = int(float(raw_row.get('tenure', 12)) <= 12 and raw_row.get('Contract', '') == 'Month-to-month')
                new_nm   = int(new_tenure <= 12 and new_contract == 'Month-to-month')

                eng_df = pd.DataFrame({
                    'Feature': ['charge_per_tenure', 'new_and_monthly'],
                    'Before':  [f'{orig_cpt:.2f}', str(orig_nm)],
                    'After':   [f'{new_cpt:.2f}',  str(new_nm)],
                    '':        ['↕' if abs(new_cpt-orig_cpt)>0.01 else '—',
                                '↕' if new_nm!=orig_nm else '—']
                })
                st.dataframe(eng_df, use_container_width=True, hide_index=True)

        # SHAP waterfall for the modified customer
        st.markdown("---")
        st.markdown("#### Why did the probability change? (SHAP)")
        if df_raw is not None:
            shap_mod = explainer.shap_values(X_mod_scaled)
            mod_display = build_display_row(modified_row.iloc[0], feature_names).values
            exp_mod = shap.Explanation(
                values        = shap_mod[0],
                base_values   = float(explainer.expected_value),
                data          = mod_display,
                feature_names = feature_names
            )
            fig_mod, _ = plt.subplots(figsize=(10, 5))
            fig_mod.patch.set_facecolor("#0d1525")
            shap.waterfall_plot(exp_mod, show=False, max_display=10)
            plt.tight_layout()
            st.pyplot(plt.gcf())
            plt.close()

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE: Export Report
    # ══════════════════════════════════════════════════════════════════════════
    elif page == "📤 Export Report":
        st.title("Export Predictions")
        st.markdown('<div class="page-subtitle">Filter and download the retention call list</div>', unsafe_allow_html=True)

        if df_results is None:
            st.warning("Please upload a CSV file first.")
            return

        col1, col2, col3 = st.columns(3)
        col1.metric("🔴 High Risk",   int((probs >= 0.7).sum()))
        col2.metric("🟡 Medium Risk", int(((probs >= 0.4) & (probs < 0.7)).sum()))
        col3.metric("🟢 Low Risk",    int((probs < 0.4).sum()))

        st.markdown("---")

        # Filter before export
        st.markdown("### Filter before exporting")
        export_filter = st.selectbox(
            "Which customers to include?",
            ["All customers", "🔴 High risk only", "🟡 Medium risk only",
             "🟢 Low risk only", "🔴 High + 🟡 Medium (recommended for retention team)"]
        )

        probs_s = pd.Series(probs)
        if "High + 🟡" in export_filter:    export_mask = probs_s >= 0.4
        elif "High risk only" in export_filter:    export_mask = probs_s >= 0.7
        elif "Medium risk only" in export_filter:  export_mask = (probs_s >= 0.4) & (probs_s < 0.7)
        elif "Low risk only" in export_filter:     export_mask = probs_s < 0.4
        else:                                       export_mask = pd.Series([True]*len(df_results))

        export_df = df_results[export_mask.values][
            ['customerID', 'Churn Probability', 'Prediction', 'Risk Level']
        ].copy().sort_values('Churn Probability', ascending=False)

        st.markdown(f"**{len(export_df):,} customers** will be exported.")

        csv_buffer = io.StringIO()
        export_df.to_csv(csv_buffer, index=False)

        st.download_button(
            label=f"⬇️ Download {len(export_df):,} Predictions as CSV",
            data=csv_buffer.getvalue(),
            file_name="churnx_predictions.csv",
            mime="text/csv",
            use_container_width=True
        )

        st.markdown("*Preview — sorted by churn probability (highest first)*")
        st.dataframe(export_df.head(20), use_container_width=True)


# ── Entry point ────────────────────────────────────────────────────────────────
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if not st.session_state['logged_in']:
    login_page()
else:
    main_app()
