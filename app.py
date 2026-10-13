r"""Material Retention Predictor - 带欢迎页 + 进阶美化版本"""

from pathlib import Path
import base64

import joblib
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Material Retention Predictor",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"  # 欢迎页时收起侧边栏
)

# ===== 路径 =====
MODEL_PATH = Path("xgb_model_seed48.pkl")
FEATURE_PATH = Path("feature_names_seed48.pkl")
LOGO_PATH = Path("logo.png")
SHAP_PATH = Path("SHAP_Importance_seed48.png")
BACKGROUND_PATH = Path("background.png")

# ===== 加载背景图为 base64 =====
if BACKGROUND_PATH.is_file():
    with open(BACKGROUND_PATH, "rb") as f:
        bg_base64 = base64.b64encode(f.read()).decode()
else:
    bg_base64 = ""

EXPECTED_FEATURES = [
    "1.分子量", "2.溶解度", "3.添加剂含量", "4.Zn吸附能",
    "5.MnO2吸附能", "MolWt", "Chi1", "NumValenceElectrons",
    "Chi0", "NumHeavyAtoms",
]
DEFAULTS = {
    "mw": 200.0, "sol": 100.0, "content": 1.0,
    "zn": -1.0, "mno2": -1.0, "molwt": 200.0,
    "chi1": 5.0, "nve": 50, "chi0": 10.0, "nha": 20,
}

# ===== 全局 CSS =====
st.markdown(f"""
<style>
/* 全局背景图 */
.stApp {{
    background-image: url("data:image/png;base64,{bg_base64}");
    background-size: cover;
    background-attachment: fixed;
    background-position: center;
}}

/* 主内容区：磨砂玻璃 + 淡入动画 */
.block-container {{
    max-width: 1280px;
    padding: 2rem 3rem;
    background: rgba(255, 255, 255, 0.92);
    backdrop-filter: blur(12px);
    border-radius: 20px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25);
    margin-top: 2rem;
    margin-bottom: 2rem;
    animation: fadeIn 0.8s ease-in;
}}

@keyframes fadeIn {{
    from {{ opacity: 0; transform: translateY(20px); }}
    to {{ opacity: 1; transform: translateY(0); }}
}}

/* 标题：渐变 + 下划线 + 打字机光标 */
h1 {{
    background: linear-gradient(135deg, #006B3F 0%, #4A90E2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
    border-bottom: 3px solid #00E676;
    padding-bottom: 0.5rem;
    display: inline-block;
}}
h2, h3 {{color: #006B3F; font-weight: 600;}}

/* 按钮 - 绿色科技感 + 脉冲动画 */
div.stButton > button {{
    background: linear-gradient(135deg, #00C853 0%, #006B3F 100%);
    color: white;
    border: none;
    border-radius: 12px;
    font-weight: 600;
    min-height: 2.8rem;
    box-shadow: 0 4px 12px rgba(0, 200, 83, 0.4);
    transition: all 0.3s ease;
}}
div.stButton > button:hover {{
    background: linear-gradient(135deg, #00E676 0%, #00C853 100%);
    box-shadow: 0 6px 25px rgba(0, 230, 118, 0.8);
    transform: translateY(-3px) scale(1.02);
}}
div.stButton > button:disabled {{opacity: 0.45; box-shadow: none;}}

/* 下载按钮 */
div.stDownloadButton > button {{
    background: linear-gradient(135deg, #4A90E2 0%, #1976D2 100%);
    color: white;
    border: none;
    border-radius: 12px;
    font-weight: 600;
    box-shadow: 0 4px 12px rgba(74, 144, 226, 0.4);
}}
div.stDownloadButton > button:hover {{
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(74, 144, 226, 0.7);
}}

/* 指标卡片 */
[data-testid="stMetricValue"] {{
    color: #00C853;
    font-size: 2.2rem;
    font-weight: 800;
    text-shadow: 0 0 20px rgba(0, 200, 83, 0.3);
}}
[data-testid="stMetricLabel"] {{color: #006B3F; font-weight: 600;}}

/* 侧边栏：磨砂玻璃 */
[data-testid="stSidebar"] {{
    background: rgba(245, 250, 245, 0.9);
    backdrop-filter: blur(12px);
}}

/* Tab 标签 */
[data-baseweb="tab"][aria-selected="true"] {{
    color: #00C853;
    border-bottom: 3px solid #00C853;
    font-weight: 700;
}}

/* 信息框 */
[data-testid="stAlert"] {{
    border-radius: 10px;
    border-left: 5px solid #00C853;
    animation: fadeIn 0.5s ease-in;
}}

/* 欢迎页特有样式 */
.welcome-hero {{
    text-align: center;
    padding: 3rem 2rem;
    animation: fadeIn 1s ease-in;
}}
.welcome-title {{
    font-size: 3.5rem;
    font-weight: 900;
    background: linear-gradient(135deg, #00C853 0%, #4A90E2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 1rem;
    line-height: 1.2;
}}
.welcome-subtitle {{
    font-size: 1.3rem;
    color: #555;
    margin-bottom: 2rem;
}}
.welcome-features {{
    display: flex;
    justify-content: center;
    gap: 2rem;
    margin-top: 3rem;
    flex-wrap: wrap;
}}
.feature-card {{
    background: rgba(255, 255, 255, 0.85);
    backdrop-filter: blur(8px);
    padding: 1.5rem;
    border-radius: 15px;
    border-top: 4px solid #00C853;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    min-width: 200px;
    text-align: center;
    transition: all 0.3s ease;
}}
.feature-card:hover {{
    transform: translateY(-5px);
    box-shadow: 0 8px 25px rgba(0, 200, 83, 0.3);
}}
.feature-icon {{
    font-size: 2.5rem;
    margin-bottom: 0.5rem;
}}
.feature-title {{
    font-size: 1.1rem;
    font-weight: 700;
    color: #006B3F;
    margin-bottom: 0.3rem;
}}
.feature-desc {{
    font-size: 0.9rem;
    color: #666;
}}
</style>
""", unsafe_allow_html=True)


# ===== 模型加载 =====
@st.cache_resource(show_spinner=False)
def load_model(model_stamp, feature_stamp):
    model = joblib.load(MODEL_PATH)
    names = joblib.load(FEATURE_PATH)
    if not isinstance(names, (list, tuple, np.ndarray)):
        raise ValueError("Feature-name file must contain a list of names.")
    names = list(names)
    if names != EXPECTED_FEATURES:
        raise ValueError("Feature-name file does not match the page input order.")
    if not callable(getattr(model, "predict", None)):
        raise TypeError("The model file has no callable predict() method.")
    if int(getattr(model, "n_features_in_", -1)) != 10:
        raise ValueError("Model input dimension is not 10.")
    return model, names


def file_stamp(path):
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_size


def clear_prediction():
    st.session_state.pop("prediction", None)


def validate_input(data):
    if data.shape != (1, 10) or not np.isfinite(data).all():
        raise ValueError("Please enter 10 valid finite values.")
    if data[0, 0] <= 0 or data[0, 5] <= 0:
        raise ValueError("Molecular weight and MolWt must be greater than 0.")
    if data[0, 1] < 0:
        raise ValueError("Solubility cannot be negative.")
    if not 0 <= data[0, 2] <= 100:
        raise ValueError("Additive content must be between 0 and 100 wt.%.")
    if data[0, 6] < 0 or data[0, 8] < 0:
        raise ValueError("Chi1 and Chi0 cannot be negative.")
    for index, label in [(7, "valence electrons"), (9, "heavy atoms")]:
        value = data[0, index]
        if value < 1 or value != np.floor(value):
            raise ValueError(f"Number of {label} must be a positive integer.")


for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)

# ===== 欢迎页逻辑 =====
if "page" not in st.session_state:
    st.session_state.page = "welcome"

# ===== 欢迎页 =====
if st.session_state.page == "welcome":
    st.markdown("""
    <div class="welcome-hero">
        <div class="welcome-title">🔬 Material Retention Predictor</div>
        <div class="welcome-subtitle">
            AI-powered screening of electrolyte additives<br>
            for aqueous Zn-MnO₂ batteries
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="welcome-features">
        <div class="feature-card">
            <div class="feature-icon">⚡</div>
            <div class="feature-title">Fast Prediction</div>
            <div class="feature-desc">Predict retention in seconds</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🎯</div>
            <div class="feature-title">High Accuracy</div>
            <div class="feature-desc">Test R² = 0.853</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🔍</div>
            <div class="feature-title">SHAP Explainability</div>
            <div class="feature-desc">Understand the model</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🚀 Start Prediction", type="primary", use_container_width=True):
            st.session_state.page = "main"
            st.rerun()

    st.stop()

# ===== 主界面 =====
# 侧边栏
with st.sidebar:
    st.header("🧪 Model Information")
    st.markdown("""
    <div style="background-color: white; padding: 1rem; border-radius: 10px;
                border-left: 4px solid #00C853; box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                margin-bottom: 1rem;">
    <h4 style="color: #006B3F; margin-top: 0;">📖 About This Tool</h4>
    <p style="font-size: 0.9rem; color: #555; line-height: 1.5;">
    This ML tool predicts the <b>capacity retention</b> of electrolyte additives
    on MnO₂ cathodes in aqueous Zn-MnO₂ batteries.
    </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("**Algorithm:** XGBoost Regressor")
    st.markdown("**Random seed:** 48")
    st.markdown("**Input features:** 10")
    st.markdown("**Target:** Capacity retention (%)")
    st.markdown("**Test R²:** 0.853")
    st.divider()
    if st.button("🔄 Reset", key="reset", use_container_width=True):
        for key, value in DEFAULTS.items():
            st.session_state[key] = value
        clear_prediction()
        st.rerun()
    if st.button("🏠 Back to Home", use_container_width=True):
        st.session_state.page = "welcome"
        st.rerun()
    st.divider()
    st.caption("© 2026 Nanjing Forestry University")

# 顶部
logo_col, title_col = st.columns([1, 9], vertical_alignment="center")
with logo_col:
    if LOGO_PATH.is_file():
        st.image(str(LOGO_PATH), width=100)
with title_col:
    st.title("Material Retention Predictor")
    st.caption("**Application**: Electrolyte additive screening for aqueous Zn-MnO₂ batteries | "
               "XGBoost + SHAP | Test R² = 0.85")
st.divider()

# 加载模型
model = None
try:
    model, feature_names = load_model(file_stamp(MODEL_PATH), file_stamp(FEATURE_PATH))
except Exception as exc:
    st.error(f"❌ Model loading failed: {exc}")

# Tabs
tab_predict, tab_features, tab_model = st.tabs(["🎯 Prediction", "📖 Feature Guide", "📊 Model Info"])

with tab_predict:
    st.subheader("🧬 Input Material Features")
    st.info("💡 **Instructions**: Enter the 10 features of an additive to predict its capacity retention on the MnO₂ cathode.")
    left, right = st.columns(2, gap="large")
    with left:
        mw = st.number_input("1. Molecular weight (g/mol)", min_value=0.0, step=10.0, key="mw", on_change=clear_prediction)
        sol = st.number_input("2. Solubility (g/L)", min_value=0.0, step=10.0, key="sol", on_change=clear_prediction)
        content = st.number_input("3. Additive content (wt.%)", min_value=0.0, max_value=100.0, step=0.1, format="%.3f", key="content", on_change=clear_prediction)
        zn = st.number_input("4. Zn adsorption energy (eV)", step=0.1, format="%.4f", key="zn", on_change=clear_prediction)
        mno2 = st.number_input("5. MnO₂ adsorption energy (eV)", step=0.1, format="%.4f", key="mno2", on_change=clear_prediction)
    with right:
        molwt = st.number_input("6. MolWt · RDKit molecular weight (g/mol)", min_value=0.0, step=10.0, key="molwt", on_change=clear_prediction)
        chi1 = st.number_input("7. Chi1 · First-order connectivity index", min_value=0.0, step=0.1, format="%.4f", key="chi1", on_change=clear_prediction)
        nve = st.number_input("8. NumValenceElectrons · valence electrons", min_value=1, step=1, key="nve", on_change=clear_prediction)
        chi0 = st.number_input("9. Chi0 · Zeroth-order connectivity index", min_value=0.0, step=0.1, format="%.4f", key="chi0", on_change=clear_prediction)
        nha = st.number_input("10. NumHeavyAtoms · heavy atoms", min_value=1, step=1, key="nha", on_change=clear_prediction)

    clicked = st.button("🎯 Predict Retention", type="primary", use_container_width=True, disabled=model is None, key="predict")
    if clicked:
        clear_prediction()
        try:
            input_data = np.array([[mw, sol, content, zn, mno2, molwt, chi1, nve, chi0, nha]])
            validate_input(input_data)
            with st.spinner("🔄 Computing..."):
                output = np.asarray(model.predict(input_data), dtype=float).reshape(-1)
                st.session_state["prediction"] = float(output[0])
        except Exception as exc:
            st.error(f"❌ Prediction failed: {exc}")

    if "prediction" in st.session_state:
        prediction = st.session_state["prediction"]
        st.success(f"Predicted retention: **{prediction:.2f}%**")
        if prediction >= 80:
            st.info("📈 High-performance material.")
        elif prediction >= 60:
            st.warning("⚖️ Moderate-performance material.")
        else:
            st.error("📉 Low-performance material.")

        result_df = pd.DataFrame([{
            'Molecular weight': mw, 'Solubility': sol, 'Additive content': content,
            'Zn adsorption energy': zn, 'MnO2 adsorption energy': mno2,
            'MolWt': molwt, 'Chi1': chi1, 'NumValenceElectrons': nve,
            'Chi0': chi0, 'NumHeavyAtoms': nha,
            'Predicted retention (%)': round(prediction, 2),
        }])
        csv = result_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button("📥 Download Prediction (CSV)", data=csv,
                           file_name=f"prediction_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                           mime='text/csv', use_container_width=True)

with tab_features:
    st.subheader("📖 Feature Description")
    st.markdown("""
| Feature | Unit | Direction |
|---|---|---|
| Molecular weight | g/mol | No fixed direction |
| Solubility | g/L | No fixed direction |
| Additive content | wt.% | May have an optimum |
| Zn adsorption energy | eV | Depends on sign convention |
| MnO₂ adsorption energy | eV | Depends on sign convention |
| MolWt | g/mol | No fixed direction |
| Chi1 | dimensionless | No fixed direction |
| NumValenceElectrons | count | No fixed direction |
| Chi0 | dimensionless | No fixed direction |
| NumHeavyAtoms | count | No fixed direction |
""")

with tab_model:
    st.subheader("📊 Model Performance")
    c1, c2, c3 = st.columns(3)
    c1.metric("🧪 Training R²", "0.976")
    c2.metric("🎯 Test R²", "0.853")
    c3.metric("📏 Test RMSE", "7.55")
    st.subheader("🔎 SHAP Feature Importance")
    if SHAP_PATH.is_file():
        st.image(str(SHAP_PATH), caption="📊 SHAP · seed48", use_container_width=True)