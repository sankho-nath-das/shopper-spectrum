import streamlit as st
import pandas as pd
import pickle
import os

# ==============================================================
# PAGE CONFIGURATION
# ==============================================================
st.set_page_config(
    page_title="Shopper Spectrum",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================
# GLOBAL STYLING (PROFESSIONAL THEME)
# ==============================================================
st.markdown("""
    <style>
        /* ---------- Global font & background ---------- */
        html, body, [class*="css"]  {
            font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
        }
        .stApp {
            background: linear-gradient(180deg, #0f1220 0%, #161a2e 100%);
        }

        /* ---------- Hero header ---------- */
        .hero {
            padding: 2.2rem 2.5rem;
            border-radius: 16px;
            background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 55%, #db2777 100%);
            box-shadow: 0 10px 30px rgba(79, 70, 229, 0.35);
            margin-bottom: 1.8rem;
        }
        .hero h1 {
            color: #ffffff;
            font-size: 2.2rem;
            font-weight: 700;
            margin: 0;
            letter-spacing: -0.5px;
        }
        .hero p {
            color: rgba(255,255,255,0.88);
            font-size: 1.02rem;
            margin-top: 0.4rem;
            margin-bottom: 0;
        }

        /* ---------- Section card ---------- */
        .section-card {
            background: #1b1f36;
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 14px;
            padding: 1.8rem 2rem;
            margin-bottom: 1.2rem;
            box-shadow: 0 4px 18px rgba(0,0,0,0.25);
        }
        .section-card h2 {
            color: #ffffff;
            font-size: 1.35rem;
            margin-top: 0;
        }
        .section-card p.subtitle {
            color: #9ca3af;
            margin-top: -0.6rem;
            margin-bottom: 1.4rem;
            font-size: 0.95rem;
        }

        /* ---------- Sidebar ---------- */
        section[data-testid="stSidebar"] {
            background: #12142299;
            border-right: 1px solid rgba(255,255,255,0.06);
        }
        section[data-testid="stSidebar"] .stRadio label {
            font-size: 1rem;
        }

        /* ---------- Buttons ---------- */
        .stButton > button {
            background: linear-gradient(135deg, #4f46e5, #7c3aed);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.6rem 1.4rem;
            font-weight: 600;
            letter-spacing: 0.2px;
            transition: transform 0.15s ease, box-shadow 0.15s ease;
            width: 100%;
        }
        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 6px 16px rgba(124, 58, 237, 0.45);
            color: white;
        }

        /* ---------- Recommendation chip ---------- */
        .rec-chip {
            background: #1b1f36;
            border: 1px solid rgba(124, 58, 237, 0.35);
            border-left: 4px solid #7c3aed;
            border-radius: 10px;
            padding: 0.85rem 1.1rem;
            margin-bottom: 0.6rem;
            color: #f1f5f9;
            font-weight: 500;
        }

        /* ---------- Footer ---------- */
        .footer {
            text-align: center;
            color: #6b7280;
            font-size: 0.85rem;
            margin-top: 2.5rem;
            padding-top: 1rem;
            border-top: 1px solid rgba(255,255,255,0.07);
        }
    </style>
""", unsafe_allow_html=True)

# ==============================================================
# LOAD MODELS (with graceful error handling)
# ==============================================================
@st.cache_resource
def load_models():
    base_path = "models"
    with open(os.path.join(base_path, "kmeans_model.pkl"), "rb") as f:
        kmeans_model = pickle.load(f)
    with open(os.path.join(base_path, "scaler.pkl"), "rb") as f:
        scaler_model = pickle.load(f)
    with open(os.path.join(base_path, "product_similarity.pkl"), "rb") as f:
        similarity_matrix = pickle.load(f)
    return kmeans_model, scaler_model, similarity_matrix

try:
    kmeans, scaler, similarity_df = load_models()
    models_loaded = True
except FileNotFoundError as e:
    models_loaded = False
    load_error = str(e)

# ==============================================================
# HERO HEADER
# ==============================================================
st.markdown("""
    <div class="hero">
        <h1>🛍️ Shopper Spectrum</h1>
        <p>Customer Segmentation &amp; Product Recommendation System</p>
    </div>
""", unsafe_allow_html=True)

if not models_loaded:
    st.error(
        "⚠️ Model files could not be loaded. Please make sure the following files "
        "exist inside a `models/` folder next to this script:\n\n"
        "- `models/kmeans_model.pkl`\n"
        "- `models/scaler.pkl`\n"
        "- `models/product_similarity.pkl`"
    )
    st.stop()

# ==============================================================
# SIDEBAR NAVIGATION
# ==============================================================
with st.sidebar:
    st.markdown("### 📂 Navigation")
    menu = st.radio(
        "Select Module",
        ["Customer Segmentation", "Product Recommendation"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown(
        "<span style='color:#9ca3af; font-size:0.85rem;'>"
        "Built with Streamlit · Powered by K-Means &amp; Cosine Similarity"
        "</span>",
        unsafe_allow_html=True
    )

# ==============================================================
# MODULE 1 — CUSTOMER SEGMENTATION
# ==============================================================
if menu == "Customer Segmentation":

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("## 👤 Customer Segmentation")
    st.markdown(
        '<p class="subtitle">Enter RFM (Recency, Frequency, Monetary) values to '
        'predict which customer segment a shopper belongs to.</p>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        recency = st.number_input("Recency (Days)", min_value=0, help="Days since the customer's last purchase")
    with col2:
        frequency = st.number_input("Frequency", min_value=1, help="Total number of purchases made")
    with col3:
        monetary = st.number_input("Monetary", min_value=0.0, help="Total amount spent by the customer")

    predict_clicked = st.button("🔍 Predict Customer Segment")
    st.markdown('</div>', unsafe_allow_html=True)

    if predict_clicked:
        values = scaler.transform([[recency, frequency, monetary]])
        cluster = kmeans.predict(values)[0]

        label_map = {
            0: "Occasional",
            1: "At-Risk",
            2: "High-Value",
            3: "Regular"
        }

        descriptions = {
            "High-Value": "Very frequent customers with high spending.",
            "Regular": "Consistent customers with good purchasing behavior.",
            "Occasional": "Customers who purchase infrequently.",
            "At-Risk": "Customers who haven't purchased for a long time."
        }

        badge_colors = {
            "High-Value": "#16a34a",
            "Regular": "#2563eb",
            "Occasional": "#d97706",
            "At-Risk": "#dc2626"
        }

        segment = label_map[cluster]
        color = badge_colors[segment]

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### 🎯 Prediction Result")

        r1, r2 = st.columns([1, 2])
        with r1:
            st.markdown(
                f"""
                <div style="background:{color}22; border:1px solid {color};
                            border-radius:12px; padding:1.2rem; text-align:center;">
                    <div style="color:{color}; font-size:1.4rem; font-weight:700;">
                        {segment}
                    </div>
                    <div style="color:#9ca3af; font-size:0.85rem; margin-top:0.3rem;">
                        Predicted Segment
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with r2:
            st.info(descriptions[segment])

        st.markdown('</div>', unsafe_allow_html=True)

# ==============================================================
# MODULE 2 — PRODUCT RECOMMENDATION
# ==============================================================
elif menu == "Product Recommendation":

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("## 🛒 Product Recommendation")
    st.markdown(
        '<p class="subtitle">Enter a product name to discover similar items '
        'based on purchasing behavior.</p>',
        unsafe_allow_html=True
    )

    product = st.text_input("Product Name", placeholder="e.g. WHITE HANGING HEART T-LIGHT HOLDER")

    def recommend(product_name):
        if product_name not in similarity_df.columns:
            return None
        similar = similarity_df[product_name].sort_values(ascending=False)
        return similar.iloc[1:6].index.tolist()

    recommend_clicked = st.button("✨ Recommend")
    st.markdown('</div>', unsafe_allow_html=True)

    if recommend_clicked:
        products = recommend(product)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        if products is None:
            st.error("❌ Product not found. Please check the spelling and try again.")
        else:
            st.markdown("### 🌟 Recommended Products")
            for item in products:
                st.markdown(f'<div class="rec-chip">🛍️ {item}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ==============================================================
# FOOTER
# ==============================================================
st.markdown(
    '<div class="footer">Shopper Spectrum &copy; 2026 — Customer Analytics Suite</div>',
    unsafe_allow_html=True
)