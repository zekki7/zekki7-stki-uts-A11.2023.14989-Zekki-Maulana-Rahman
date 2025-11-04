import os
import pandas as pd
from scipy.sparse import csr_matrix
import numpy as np

# ==============================================================
# 1️⃣ Load dokumen hasil preprocessing
# ==============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
clean_path = os.path.join(BASE_DIR, "data", "processed")

# Cek apakah folder processed ada
if not os.path.exists(clean_path):
    print(f"❌ ERROR: Folder tidak ditemukan: {clean_path}")
    print("⚠️  Jalankan 'python src/preprocess.py' terlebih dahulu!")
    exit(1)

documents = {}
for file in sorted(os.listdir(clean_path)):
    if file.endswith(".txt"):
        with open(os.path.join(clean_path, file), "r", encoding="utf-8") as f:
            documents[file] = f.read().split()

# Cek apakah ada dokumen yang berhasil di-load
if not documents:
    print(f"❌ ERROR: Tidak ada file .txt di folder: {clean_path}")
    print("⚠️  Pastikan preprocessing sudah dijalankan dengan benar!")
    exit(1)

# ==============================================================
# 2️⃣ Fungsi Build Inverted Index
# ==============================================================

def build_inverted_index(docs):
    """
    Bangun inverted index dari dokumen
    
    Args:
        docs: dict {doc_name: [tokens]}
    
    Returns:
        tuple: (inverted_index, vocabulary)
        - inverted_index: dict {term: set(doc_names)}
        - vocabulary: sorted list of unique terms
    """
    # Bangun vocabulary
    vocab = sorted(set(term for tokens in docs.values() for term in tokens))
    
    # Bangun inverted index
    inv_idx = {}
    for term in vocab:
        inv_idx[term] = {doc for doc, tokens in docs.items() if term in tokens}
    
    return inv_idx, vocab

# ==============================================================
# 3️⃣ Fungsi Build Incidence Matrix (Sparse)
# ==============================================================

def build_incidence_matrix(docs, vocab, inv_idx):
    """
    Bangun sparse incidence matrix (term x document)
    
    Args:
        docs: dict {doc_name: [tokens]}
        vocab: sorted list of terms
        inv_idx: inverted index dict
    
    Returns:
        scipy.sparse.csr_matrix: sparse incidence matrix
    """
    incidence_data = []
    doc_names = list(docs.keys())
    
    for term in vocab:
        row = [1 if doc in inv_idx[term] else 0 for doc in doc_names]
        incidence_data.append(row)
    
    # Konversi ke sparse matrix untuk efisiensi memori
    sparse_matrix = csr_matrix(incidence_data)
    
    print(f"✅ Sparse Incidence Matrix: {len(vocab)} terms x {len(doc_names)} docs")
    
    total_elements = sparse_matrix.shape[0] * sparse_matrix.shape[1]
    if total_elements > 0:
        density = (sparse_matrix.nnz / total_elements) * 100
        print(f"   Memory efficiency: {sparse_matrix.nnz}/{total_elements} non-zero elements ({density:.1f}% density)")
    else:
        print(f"   ⚠️ Warning: Empty matrix (no documents loaded)")
    
    return sparse_matrix

# ==============================================================
# 4️⃣ Inisialisasi Global Variables
# ==============================================================

inverted_index, vocabulary = build_inverted_index(documents)
incidence_matrix_sparse = build_incidence_matrix(documents, vocabulary, inverted_index)

# Untuk display purpose, buat DataFrame biasa (opsional)
incidence_df = pd.DataFrame(
    incidence_matrix_sparse.toarray(), 
    index=vocabulary, 
    columns=documents.keys()
)

# ==============================================================
# 5️⃣ Fungsi Evaluasi (Precision & Recall)
# ==============================================================

def evaluate(query_result, relevant_docs):
    """
    Hitung Precision dan Recall
    
    Args:
        query_result: set of retrieved documents
        relevant_docs: set of gold standard relevant documents
    
    Returns:
        tuple: (precision, recall, f1)
    """
    if not query_result:
        return 0.0, 0.0, 0.0
    
    if not relevant_docs:
        return 0.0, 0.0, 0.0
    
    true_positive = len(query_result & relevant_docs)
    precision = true_positive / len(query_result)
    recall = true_positive / len(relevant_docs)
    
    # Hitung F1-Score
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0
    
    return precision, recall, f1

# ==============================================================
# 6️⃣ Fungsi Explain untuk Operasi Boolean
# ==============================================================

def explain_set(op, set_a, set_b, term_a=None, term_b=None):
    """
    Jelaskan operasi Boolean set
    
    Args:
        op: operator ("AND", "OR", "NOT")
        set_a, set_b: set dokumen
        term_a, term_b: nama term (untuk logging)
    
    Returns:
        set: hasil operasi
    """
    if op == "AND":
        result = set_a & set_b
        desc = f"'{term_a}' AND '{term_b}'" if term_a and term_b else "Intersection"
        print(f"   └─ {desc} → {len(result)} docs: {sorted(result)}")
    elif op == "OR":
        result = set_a | set_b
        desc = f"'{term_a}' OR '{term_b}'" if term_a and term_b else "Union"
        print(f"   └─ {desc} → {len(result)} docs: {sorted(result)}")
    elif op == "NOT":
        result = set_a - set_b
        desc = f"NOT '{term_b}'" if term_b else "Complement"
        print(f"   └─ {desc} → {len(result)} docs: {sorted(result)}")
    else:
        result = set_a
    
    return result

# ==============================================================
# 7️⃣ Parser Query Boolean (Diperbaiki)
# ==============================================================

def boolean_retrieval(query, verbose=True):
    """
    Parse dan eksekusi query Boolean sederhana
    Mendukung: AND, OR, NOT
    
    Args:
        query: string query (e.g., "term1 AND term2", "NOT term3")
        verbose: tampilkan explain step-by-step
    
    Returns:
        set: dokumen yang memenuhi query
    """
    tokens = query.lower().split()
    result = None
    current_op = None
    negate_next = False
    prev_term = None
    
    if verbose:
        print(f"\n🔍 Processing query: '{query}'")
    
    for i, token in enumerate(tokens):
        if token == "and":
            current_op = "AND"
            if verbose:
                print(f"   Operator: AND")
        elif token == "or":
            current_op = "OR"
            if verbose:
                print(f"   Operator: OR")
        elif token == "not":
            negate_next = True
            if verbose:
                print(f"   Operator: NOT (negasi untuk term berikutnya)")
        else:
            # Token adalah term
            docs_with_token = inverted_index.get(token, set())
            current_term = token
            
            if verbose:
                print(f"   Term: '{token}' → {len(docs_with_token)} docs: {sorted(docs_with_token)}")
            
            # Tangani negasi
            if negate_next:
                docs_with_token = set(documents.keys()) - docs_with_token
                current_term = f"NOT {token}"
                if verbose:
                    print(f"   Setelah negasi: {len(docs_with_token)} docs: {sorted(docs_with_token)}")
                negate_next = False
            
            # Inisialisasi result pertama kali
            if result is None:
                result = docs_with_token
                prev_term = current_term
            else:
                # Operasi Boolean
                if current_op == "AND":
                    result = explain_set("AND", result, docs_with_token, prev_term, current_term) if verbose else result & docs_with_token
                elif current_op == "OR":
                    result = explain_set("OR", result, docs_with_token, prev_term, current_term) if verbose else result | docs_with_token
                else:
                    # Default ke AND jika tidak ada operator
                    result = explain_set("AND", result, docs_with_token, prev_term, current_term) if verbose else result & docs_with_token
                
                prev_term = f"({prev_term} {current_op} {current_term})"
                current_op = None
    
    return result if result is not None else set()

# ==============================================================
# 8️⃣ Wrapper untuk Integrasi ke search_engine.py
# ==============================================================

def boolean_search(query):
    """Wrapper function untuk dipanggil dari search_engine.py"""
    print(f"\n{'='*60}")
    print(f"🔎 BOOLEAN SEARCH MODE")
    print(f"{'='*60}")
    result = boolean_retrieval(query, verbose=True)
    print(f"\n✅ Final Result: {len(result)} dokumen relevan")
    print(f"   Dokumen: {sorted(result)}")
    print(f"{'='*60}")
    return result

# ==============================================================
# 9️⃣ Mini Truth Set untuk Evaluasi
# ==============================================================

# ✅ Truth Set berdasarkan konten dokumen review makanan/kafe
# Disesuaikan dengan vocabulary dari processed documents

truth_set = {
    # Query 1: Makanan pedas dengan ayam
    "ayam AND pedes": {"doc1.txt", "doc3.txt", "doc7.txt"},
    
    # Query 2: Review tentang kopi atau tempat nongkrong
    "kopi OR nongkrong": {"doc2.txt", "doc10.txt"},
    
    # Query 3: Enak tapi bukan yang terlalu mahal
    "enak AND NOT mahal": {"doc1.txt", "doc2.txt", "doc7.txt", "doc10.txt"},
    
    # Query 4: Tempat dengan masalah service/kebersihan
    "tempat AND lama": {"doc4.txt", "doc6.txt", "doc8.txt"},
    
    # Query 5: Menu dengan keju atau kremes (crispy)
    "keju OR kremes": {"doc1.txt", "doc7.txt", "doc10.txt"},
    
    # Query 6: Review positif tanpa komplain service lambat
    "enak AND NOT lama": {"doc1.txt", "doc3.txt", "doc5.txt", "doc7.txt"},
    
    # Query 7: Dessert atau makanan manis
    "manis": {"doc2.txt", "doc5.txt"},
    
    # Query 8: Review dengan komplain staff/layanan
    "staff OR layan": {"doc6.txt", "doc8.txt", "doc9.txt", "doc10.txt"}
}

# ==============================================================
# 🔟 Testing & Evaluation
# ==============================================================

def run_evaluation():
    """Jalankan evaluasi dengan truth set"""
    print("\n" + "="*60)
    print("📊 EVALUASI BOOLEAN RETRIEVAL")
    print("="*60)
    
    results = []
    
    for query, gold_docs in truth_set.items():
        print(f"\n{'─'*60}")
        print(f"Query: {query}")
        print(f"Gold Standard: {sorted(gold_docs)}")
        
        # Retrieve
        retrieved = boolean_retrieval(query, verbose=True)
        
        # Evaluate
        precision, recall, f1 = evaluate(retrieved, gold_docs)
        
        print(f"\n📈 Metrics:")
        print(f"   Precision: {precision:.3f} ({len(retrieved & gold_docs)}/{len(retrieved)} relevan)")
        print(f"   Recall:    {recall:.3f} ({len(retrieved & gold_docs)}/{len(gold_docs)} ditemukan)")
        print(f"   F1-Score:  {f1:.3f}")
        
        results.append({
            'query': query,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'retrieved': len(retrieved),
            'relevant': len(gold_docs),
            'true_positive': len(retrieved & gold_docs)
        })
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 SUMMARY EVALUASI")
    print("="*60)
    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))
    print(f"\n📌 Average Metrics:")
    print(f"   Precision: {df_results['precision'].mean():.3f}")
    print(f"   Recall:    {df_results['recall'].mean():.3f}")
    print(f"   F1-Score:  {df_results['f1'].mean():.3f}")
    
    return df_results

# ==============================================================
# 🎯 Main Execution
# ==============================================================

if __name__ == "__main__":
    print("="*60)
    print("🔎 BOOLEAN INFORMATION RETRIEVAL MODEL")
    print("="*60)
    print(f"\n📚 Corpus Statistics:")
    print(f"   Vocabulary size: {len(vocabulary)} unique terms")
    print(f"   Documents: {len(documents)}")
    print(f"   Document list: {sorted(documents.keys())}")
    
    print(f"\n📊 Incidence Matrix (Sparse):")
    print(f"   Shape: {incidence_matrix_sparse.shape}")
    print(f"   Non-zero entries: {incidence_matrix_sparse.nnz}")
    if incidence_matrix_sparse.shape[0] * incidence_matrix_sparse.shape[1] > 0:
        print(f"   Sparsity: {(1 - incidence_matrix_sparse.nnz / (incidence_matrix_sparse.shape[0] * incidence_matrix_sparse.shape[1])) * 100:.2f}%")
    
    print(f"\n📋 Sample Incidence Matrix (first 10 terms):")
    print(incidence_df.head(10))
    
    print(f"\n🔍 Sample Inverted Index (first 5 terms):")
    for i, (term, docs) in enumerate(list(inverted_index.items())[:5]):
        print(f"   '{term}' → {sorted(docs)}")
    
    # Run evaluation
    results_df = run_evaluation()
    
    print(f"\n✅ Evaluation completed!")
    print(f"   Results saved to DataFrame: results_df")