# ============================================================
# app/main.py - Mini Search Engine STKI Project
# ============================================================

import os
import sys
import streamlit as st

# Pastikan folder src bisa diimpor
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Import modul
from src.boolean_ir import boolean_search
from src.vsm_ir import vsm_search
from src.eval import compare_vsm_schemes as compare_schemes
from src.search_engine import corpus_statistics

# ============================================================
# 📋 Fungsi Display Menu (UNTUK STREAMLIT)
# ============================================================
def display_menu():
    """Tampilkan menu utama dengan radio button Streamlit"""
    menu = st.sidebar.radio(
        "📂 Menu Utama",
        [
            "🏠 Home",
            "Boolean Search",
            "VSM Search",
            "Compare Schemes",
            "Corpus Statistics",
        ]
    )
    return menu

# ============================================================
# 📊 Fungsi tambahan (CLI / info)
# ============================================================
def show_corpus_stats():
    """Menampilkan statistik corpus"""
    st.subheader("📚 Corpus Statistics")
    stats = corpus_statistics()
    st.json(stats)

# ============================================================
# 🧠 MAIN STREAMLIT APP
# ============================================================
st.title("🔍 MINI SEARCH ENGINE - STKI PROJECT")
st.caption("Information Retrieval System menggunakan Boolean & Vector Space Model")

menu = display_menu()

# ------------------------------------------------------------
# HOME
# ------------------------------------------------------------
if "Home" in menu or menu == "🏠 Home":
    st.markdown("""
    ### Selamat Datang di Mini Search Engine (STKI)
    Aplikasi ini mendemonstrasikan dua pendekatan utama *Information Retrieval*:
    - **Boolean Model** (menggunakan operator AND, OR, NOT)
    - **Vector Space Model (VSM)** dengan perbandingan skema pembobotan TF-IDF.

    Gunakan menu di sebelah kiri untuk memulai 🔎
    """)

# ------------------------------------------------------------
# BOOLEAN SEARCH
# ------------------------------------------------------------
elif "Boolean Search" in menu:
    st.header("🔎 Boolean Search")
    query = st.text_input("Masukkan query Boolean (gunakan AND, OR, NOT):")
    if st.button("Cari"):
        if query.strip():
            hasil = boolean_search(query)
            st.write("### Hasil Pencarian:")
            st.write(hasil)
        else:
            st.warning("Masukkan query terlebih dahulu!")

# ------------------------------------------------------------
# VSM SEARCH
# ------------------------------------------------------------
elif "VSM Search" in menu:
    st.header("📈 Vector Space Model Search")
    query = st.text_input("Masukkan query untuk VSM Search:")
    top_k = st.number_input("Jumlah top dokumen:", min_value=1, max_value=20, value=5)
    scheme = st.selectbox("Pilih weighting scheme:", ["standard", "sublinear"])
    if st.button("Cari"):
        if query.strip():
            hasil = vsm_search(query, top_k=top_k, scheme=scheme)
            st.write("### Hasil Pencarian TF-IDF:")
            st.write(hasil)
        else:
            st.warning("Masukkan query terlebih dahulu!")

# ------------------------------------------------------------
# COMPARE SCHEMES
# ------------------------------------------------------------
elif "Compare Schemes" in menu:
    st.header("⚖️ Perbandingan TF-IDF Schemes")
    query = st.text_input("Masukkan query untuk dibandingkan:")
    top_k = st.number_input("Top-k dokumen:", min_value=1, max_value=20, value=5)
    if st.button("Bandingkan"):
        if query.strip():
            hasil = compare_schemes(query, top_k=top_k)
            st.write("### Hasil Perbandingan Skema:")
            st.write(hasil)
        else:
            st.warning("Masukkan query terlebih dahulu!")

# ------------------------------------------------------------
# CORPUS STATISTICS
# ------------------------------------------------------------
elif "Corpus Statistics" in menu:
    st.header("📊 Informasi Corpus")
    if st.button("Tampilkan Statistik"):
        show_corpus_stats()

# ============================================================
# 🚀 Entry point untuk CLI (opsional)
# ============================================================
if __name__ == "__main__":
    print("Jalankan aplikasi dengan Streamlit:")
    print("   streamlit run app/main.py")
