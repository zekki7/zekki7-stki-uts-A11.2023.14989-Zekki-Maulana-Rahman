# app/main.py
import os
import sys
import streamlit as st

# Path setup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, "src"))

# Import modules
from src.boolean_ir import boolean_search
from src.vsm_ir import vsm_search
from src.eval import compare_vsm_schemes
from src.search_engine import corpus_statistics

# ============================================================
# 🌙 Custom Dark Theme + Poppins Font
# ============================================================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

    html, body, [class*="st-"], .stApp {
        font-family: 'Poppins', sans-serif;
        background-color: #0d1117;
        color: #f0f6fc;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #161b22;
        color: #f0f6fc;
        font-family: 'Poppins', sans-serif;
    }

    /* Headings */
    h1, h2, h3, h4 {
        color: #58a6ff;
        font-weight: 600;
    }

    /* Buttons */
    div.stButton > button {
        background-color: #238636;
        color: #fff;
        border-radius: 8px;
        border: none;
        padding: 0.5em 1em;
        font-weight: 600;
        transition: all 0.3s ease-in-out;
        font-family: 'Poppins', sans-serif;
    }
    div.stButton > button:hover {
        background-color: #2ea043;
        transform: scale(1.03);
        box-shadow: 0 0 12px rgba(46, 160, 67, 0.5);
    }

    /* Text input and sliders */
    input, textarea, .stSlider, .stSelectbox {
        background-color: #161b22 !important;
        color: #f0f6fc !important;
        border: 1px solid #30363d !important;
        border-radius: 6px !important;
        font-family: 'Poppins', sans-serif !important;
    }

    /* Result cards */
    .result-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 15px 20px;
        margin-bottom: 12px;
        box-shadow: 0 0 10px rgba(88, 166, 255, 0.15);
        transition: transform 0.2s ease-in-out;
    }
    .result-card:hover {
        transform: scale(1.02);
        box-shadow: 0 0 16px rgba(88, 166, 255, 0.3);
    }

    /* Highlighted query terms */
    .highlight {
        color: #ff79c6;
        font-weight: 500;
    }

    /* Sidebar radio buttons */
    .stRadio > label {
        color: #f0f6fc !important;
    }

    /* Info caption */
    .stCaption {
        color: #8b949e;
        font-style: italic;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# 🎛️ Sidebar Menu
# ============================================================
def display_menu():
    st.sidebar.title("⚙️ Navigation")
    return st.sidebar.radio(
        "Pilih Mode:",
        ("Boolean Search", "VSM Search", "Compare Schemes", "Corpus Statistics")
    )

# ============================================================
# 🚀 Main Interface
# ============================================================
st.title("🔍 MINI SEARCH ENGINE - STKI PROJECT")
st.caption("Information Retrieval System — Boolean & Vector Space Model")
menu = display_menu()

# ============================================================
# 🧩 BOOLEAN SEARCH MODE
# ============================================================
if menu == "Boolean Search":
    st.header("🔎 Boolean Search")
    query = st.text_input("Masukkan query (gunakan AND / OR / NOT):", "")
    if st.button("Cari"):
        if query.strip():
            hasil = boolean_search(query)
            st.subheader("📄 Hasil Pencarian:")
            if hasil:
                for doc in hasil:
                    st.markdown(f"<div class='result-card'>📘 <b>{doc}</b></div>", unsafe_allow_html=True)
            else:
                st.warning("Tidak ada dokumen yang cocok.")
        else:
            st.warning("Masukkan query terlebih dahulu.")

# ============================================================
# 🧮 VSM SEARCH MODE
# ============================================================
elif menu == "VSM Search":
    st.header("📊 Vector Space Model (TF-IDF)")
    query = st.text_input("Masukkan kata kunci bebas:")
    top_k = st.slider("Jumlah hasil yang ditampilkan:", 3, 20, 5)
    if st.button("Cari"):
        if query.strip():
            hasil = vsm_search(query)
            if hasil:
                for r in hasil[:top_k]:
                    score = r.get('score', 0)
                    snippet = r.get('snippet', '')
                    st.markdown(
                        f"<div class='result-card'>"
                        f"<b>{r['doc_id']}</b><br>"
                        f"Skor: <b style='color:#58a6ff'>{score:.4f}</b><br>"
                        f"<span class='highlight'>{snippet}</span>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
            else:
                st.warning("Tidak ada hasil ditemukan.")
        else:
            st.warning("Masukkan query terlebih dahulu.")

# ============================================================
# ⚖️ COMPARE MODE
# ============================================================
elif menu == "Compare Schemes":
    st.header("⚖️ Compare TF-IDF Schemes")
    query = st.text_input("Masukkan query untuk perbandingan:")
    if st.button("Bandingkan"):
        if query.strip():
            hasil = compare_vsm_schemes(query)
            st.write(hasil)
        else:
            st.warning("Masukkan query terlebih dahulu.")

# ============================================================
# 📊 CORPUS STATS
# ============================================================
elif menu == "Corpus Statistics":
    st.header("📚 Corpus Statistics")
    if st.button("Tampilkan Info"):
        corpus_statistics()
        st.success("Statistik corpus ditampilkan di console.")
