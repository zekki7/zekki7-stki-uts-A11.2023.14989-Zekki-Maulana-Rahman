# app/main.py

import os
import sys
import streamlit as st

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from src.boolean_ir import boolean_search
from src.vsm_ir import vsm_search
from src.eval import compare_schemes
from src.search_engine import corpus_statistics


# ============================================================
# Tambahkan folder src ke sys.path agar modul bisa diimport
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, "src"))

# ============================================================
# Import modul dari src/
# ============================================================
try:
    from boolean_ir import boolean_retrieval, documents as bool_docs, vocabulary as bool_vocab
    from vsm_ir import search_vsm, set_weighting_scheme, documents, vocabulary
    from vsm_ir import compare_weighting_schemes as vsm_compare
    print("✅ Modul berhasil di-import!")
except ImportError as e:
    print(f"❌ ERROR: Gagal import modul!")
    print(f"   {e}")
    print(f"\n💡 Pastikan file berikut ada di folder src/:")
    print(f"   - boolean_ir.py")
    print(f"   - vsm_ir.py")
    print(f"   - preprocess.py")
    sys.exit(1)

# ============================================================
# 📊 Tampilkan Corpus Statistics
# ============================================================
def show_corpus_stats():
    """Tampilkan statistik corpus di awal"""
    print("\n" + "="*70)
    print("📚 CORPUS STATISTICS")
    print("="*70)
    print(f"Total Documents    : {len(documents)}")
    print(f"Vocabulary Size    : {len(vocabulary)} unique terms")
    print(f"Document List      : {', '.join(sorted(documents.keys())[:5])}...")
    print("="*70)

# ============================================================
# 🔍 VSM Search - Display Results
# ============================================================
def show_results_vsm(query, top_k=5, weighting="standard"):
    """
    Tampilkan hasil VSM search dengan format yang informatif
    
    Args:
        query (str): query string
        top_k (int): jumlah top documents
        weighting (str): "standard" atau "sublinear"
    """
    print(f"\n{'='*70}")
    print(f"🔎 VSM SEARCH MODE - {weighting.upper()}")
    print(f"{'='*70}")
    print(f"Query    : '{query}'")
    print(f"Top-K    : {top_k}")
    print(f"Weighting: {weighting}")
    print(f"{'─'*70}")
    
    # Set weighting scheme
    set_weighting_scheme(weighting)
    
    # Execute search
    try:
        results = search_vsm(query, top_k=top_k, verbose=False)
        
        if not results:
            print("\n⚠️  Tidak ada dokumen relevan ditemukan.")
            print(f"{'='*70}\n")
            return
        
        print(f"\n📊 TOP-{len(results)} RESULTS:")
        print(f"{'─'*70}")
        
        for r in results:
            print(f"\n{r['rank']}. {r['doc_id']:<15} | Score: {r['score']:.4f}")
            print(f"   📝 Snippet: {r['snippet']}")
            
            # Tampilkan matching terms
            doc_name = r['doc_id']
            if doc_name in documents:
                doc_tokens = documents[doc_name]
                query_tokens = set(query.lower().split())
                matching_terms = [t for t in query_tokens if t in doc_tokens]
                
                if matching_terms:
                    # Hitung frequency
                    term_freq = {t: doc_tokens.count(t) for t in matching_terms}
                    sorted_terms = sorted(term_freq.items(), key=lambda x: x[1], reverse=True)
                    top_matching = [f"{t}({c})" for t, c in sorted_terms[:3]]
                    print(f"   🔑 Matching: {', '.join(top_matching)}")
        
        print(f"\n{'='*70}\n")
    
    except Exception as e:
        print(f"\n❌ ERROR saat VSM search: {e}")
        print(f"{'='*70}\n")

# ============================================================
# 🔍 Boolean Search - Display Results
# ============================================================
def show_results_boolean(query):
    """
    Tampilkan hasil Boolean search dengan explain lengkap
    
    Args:
        query (str): Boolean query (support AND/OR/NOT)
    """
    print(f"\n{'='*70}")
    print(f"🔎 BOOLEAN SEARCH MODE")
    print(f"{'='*70}")
    print(f"Query: '{query}'")
    print(f"{'─'*70}")
    
    try:
        # Execute boolean retrieval dengan verbose untuk explain
        result_docs = boolean_retrieval(query, verbose=True)
        
        print(f"\n{'─'*70}")
        print(f"✅ FINAL RESULT: {len(result_docs)} dokumen ditemukan")
        
        if not result_docs:
            print(f"📄 Tidak ada dokumen yang memenuhi query")
        else:
            print(f"📄 Dokumen hasil:")
            for i, doc in enumerate(sorted(result_docs), 1):
                print(f"   {i}. {doc}")
        
        print(f"{'='*70}\n")
    
    except Exception as e:
        print(f"\n❌ ERROR saat Boolean search: {e}")
        print(f"{'='*70}\n")

# ============================================================
# ⚖️ Compare VSM Weighting Schemes
# ============================================================
def compare_schemes(query, top_k=5):
    """
    Bandingkan 2 skema weighting untuk query yang sama
    
    Args:
        query (str): query string
        top_k (int): jumlah top documents
    """
    print(f"\n{'='*70}")
    print(f"⚖️  COMPARISON MODE: VSM WEIGHTING SCHEMES")
    print(f"{'='*70}")
    print(f"Query: '{query}'")
    print(f"Top-K: {top_k}")
    print(f"Comparing: Standard TF-IDF vs Sublinear TF-IDF")
    print(f"{'='*70}")
    
    schemes = ["standard", "sublinear"]
    all_results = {}
    
    for scheme in schemes:
        print(f"\n{'─'*70}")
        print(f"📊 SCHEME: {scheme.upper()}")
        print(f"{'─'*70}")
        
        set_weighting_scheme(scheme)
        results = search_vsm(query, top_k=top_k, verbose=False)
        all_results[scheme] = results
        
        if results:
            for r in results:
                print(f"{r['rank']}. {r['doc_id']:<15} | Score: {r['score']:.4f}")
        else:
            print("⚠️  Tidak ada dokumen relevan")
    
    # Analysis
    print(f"\n{'='*70}")
    print(f"💡 ANALYSIS")
    print(f"{'─'*70}")
    
    if all_results["standard"] and all_results["sublinear"]:
        std_docs = [r['doc_id'] for r in all_results["standard"]]
        sub_docs = [r['doc_id'] for r in all_results["sublinear"]]
        
        same_docs = set(std_docs) & set(sub_docs)
        print(f"Common documents : {len(same_docs)}/{top_k}")
        
        if std_docs == sub_docs:
            print(f"Ranking order    : IDENTICAL")
        else:
            print(f"Ranking order    : DIFFERENT")
    
    print(f"\n📌 KEY DIFFERENCE:")
    print(f"   Standard  : TF = raw term count")
    print(f"   Sublinear : TF = 1 + log(count) → reduces high-frequency term dominance")
    print(f"{'='*70}\n")

# ============================================================
# 📋 Display Menu
# ============================================================
from src.boolean_ir import boolean_search
from src.vsm_ir import vsm_search
from src.eval import compare_schemes
from src.search_engine import corpus_statistics

st.title("🔍 MINI SEARCH ENGINE - STKI PROJECT")
st.write("Information Retrieval System dengan Boolean & Vector Space Model")

menu = display_menu()

if "Boolean Search" in menu:
    query = st.text_input("Masukkan query (gunakan AND, OR, NOT):")
    if st.button("Cari"):
        hasil = boolean_search(query)
        st.write("### Hasil Pencarian:")
        st.write(hasil)

elif "VSM Search" in menu:
    query = st.text_input("Masukkan query untuk VSM Search:")
    if st.button("Cari"):
        hasil = vsm_search(query)
        st.write("### Hasil Pencarian TF-IDF:")
        st.write(hasil)

elif "Compare Schemes" in menu:
    if st.button("Bandingkan"):
        hasil = compare_schemes()
        st.write(hasil)

elif "Corpus Statistics" in menu:
    if st.button("Tampilkan Info"):
        corpus_statistics()
# ============================================================
# 🎯 Main Interface (Interactive CLI)
# ============================================================
def main():
    """Main interactive interface"""
    
    # Header
    print("\n" + "="*70)
    print("🔍 MINI SEARCH ENGINE - STKI PROJECT")
    print("="*70)
    print("Information Retrieval System dengan Boolean & Vector Space Model")
    print("="*70)
    
    # Show corpus stats at start
    show_corpus_stats()
    
    # Main loop
    while True:
        try:
            display_menu()
            
            choice = input("\n➤ Pilih menu (1-5): ").strip()
            
            # ═══════════════════════════════════════════════════════
            # MENU 1: BOOLEAN SEARCH
            # ═══════════════════════════════════════════════════════
            if choice == "1":
                print("\n" + "─"*70)
                print("💡 Tips: Gunakan operator AND, OR, NOT")
                print("   Contoh: 'ayam AND enak', 'kopi OR nongkrong', 'enak AND NOT mahal'")
                print("─"*70)
                
                query = input("\n➤ Masukkan query Boolean: ").strip()
                
                if not query:
                    print("⚠️  Query tidak boleh kosong!")
                    continue
                
                show_results_boolean(query)
            
            # ═══════════════════════════════════════════════════════
            # MENU 2: VSM SEARCH
            # ═══════════════════════════════════════════════════════
            elif choice == "2":
                print("\n" + "─"*70)
                print("💡 Tips: Masukkan kata kunci bebas (tidak perlu operator)")
                print("   Contoh: 'ayam enak', 'kopi nongkrong', 'tempat bagus'")
                print("─"*70)
                
                query = input("\n➤ Masukkan query: ").strip()
                
                if not query:
                    print("⚠️  Query tidak boleh kosong!")
                    continue
                
                # Input top-k
                top_k_input = input(f"➤ Top-k dokumen [default=5]: ").strip()
                if top_k_input.isdigit() and int(top_k_input) > 0:
                    top_k = int(top_k_input)
                else:
                    top_k = 5
                
                # Input weighting scheme
                print("\n   Weighting schemes:")
                print("   1. standard   - TF = raw count")
                print("   2. sublinear  - TF = 1 + log(count)")
                
                weighting_input = input(f"➤ Pilih scheme [1/2, default=1]: ").strip()
                
                if weighting_input == "2":
                    weighting = "sublinear"
                else:
                    weighting = "standard"
                
                show_results_vsm(query, top_k=top_k, weighting=weighting)
            
            # ═══════════════════════════════════════════════════════
            # MENU 3: COMPARE SCHEMES
            # ═══════════════════════════════════════════════════════
            elif choice == "3":
                print("\n" + "─"*70)
                print("💡 Bandingkan hasil retrieval dengan 2 weighting schemes")
                print("─"*70)
                
                query = input("\n➤ Masukkan query: ").strip()
                
                if not query:
                    print("⚠️  Query tidak boleh kosong!")
                    continue
                
                top_k_input = input(f"➤ Top-k dokumen [default=5]: ").strip()
                if top_k_input.isdigit() and int(top_k_input) > 0:
                    top_k = int(top_k_input)
                else:
                    top_k = 5
                
                compare_schemes(query, top_k=top_k)
            
            # ═══════════════════════════════════════════════════════
            # MENU 4: CORPUS STATISTICS
            # ═══════════════════════════════════════════════════════
            elif choice == "4":
                show_corpus_stats()
                
                # Tampilkan sample vocabulary
                print("\n📖 Sample Vocabulary (first 20 terms):")
                print("─"*70)
                sample_vocab = list(vocabulary)[:20]
                for i in range(0, len(sample_vocab), 5):
                    print("   " + ", ".join(sample_vocab[i:i+5]))
                print("="*70)
            
            # ═══════════════════════════════════════════════════════
            # MENU 5: EXIT
            # ═══════════════════════════════════════════════════════
            elif choice == "5" or choice.lower() in ["exit", "quit", "q"]:
                print("\n" + "="*70)
                print("👋 Terima kasih telah menggunakan Mini Search Engine!")
                print("="*70)
                break
            
            else:
                print("\n❌ Pilihan tidak valid! Pilih menu 1-5.")
        
        except KeyboardInterrupt:
            print("\n\n" + "="*70)
            print("⚠️  Program dihentikan oleh user (Ctrl+C)")
            print("="*70)
            break
        
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            print("\n💡 Coba lagi atau pilih menu lain.\n")

# ============================================================
# 🚀 Entry Point
# ============================================================
if __name__ == "__main__":
    main()