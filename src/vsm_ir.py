import os
import numpy as np
import pandas as pd
from collections import Counter
from math import log10
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

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

if not documents:
    print(f"❌ ERROR: Tidak ada file .txt di folder: {clean_path}")
    exit(1)

filenames = list(documents.keys())
N = len(documents)

# ==============================================================  
# 2️⃣ Vocabulary, DF, dan IDF
# ==============================================================

vocabulary = sorted(set(term for tokens in documents.values() for term in tokens))
vocab_idx = {term: idx for idx, term in enumerate(vocabulary)}

# Document Frequency (DF)
df = {term: sum(1 for tokens in documents.values() if term in tokens) for term in vocabulary}

# Inverse Document Frequency (IDF)
idf = {term: log10(N / df[term]) if df[term] > 0 else 0.0 for term in vocabulary}

print(f"📊 Vocabulary: {len(vocabulary)} unique terms")
print(f"📚 Documents: {N}")

# ==============================================================  
# 3️⃣ SKEMA 1: TF-IDF Standard
# ==============================================================

def compute_tfidf_standard(docs, vocab, idf_dict):
    """
    Hitung TF-IDF matrix dengan skema STANDARD
    TF = raw count
    
    Returns:
        scipy.sparse.csr_matrix: sparse TF-IDF matrix
    """
    tfidf_data = []
    
    for tokens in docs.values():
        tf = Counter(tokens)
        doc_vec = [tf.get(term, 0) * idf_dict[term] for term in vocab]
        tfidf_data.append(doc_vec)
    
    return csr_matrix(tfidf_data)

# ==============================================================  
# 4️⃣ SKEMA 2: TF-IDF Sublinear (Log-scaled TF)
# ==============================================================

def compute_tfidf_sublinear(docs, vocab, idf_dict):
    """
    Hitung TF-IDF matrix dengan skema SUBLINEAR
    TF = 1 + log10(count) jika count > 0, else 0
    
    Keuntungan: mengurangi pengaruh term yang sangat sering muncul
    
    Returns:
        scipy.sparse.csr_matrix: sparse TF-IDF matrix
    """
    tfidf_data = []
    
    for tokens in docs.values():
        tf = Counter(tokens)
        doc_vec = []
        for term in vocab:
            count = tf.get(term, 0)
            # Sublinear scaling: 1 + log(tf) jika tf > 0
            tf_scaled = (1 + log10(count)) if count > 0 else 0
            doc_vec.append(tf_scaled * idf_dict[term])
        tfidf_data.append(doc_vec)
    
    return csr_matrix(tfidf_data)

# ==============================================================  
# 5️⃣ Global variable untuk TF-IDF matrix (akan di-switch)
# ==============================================================

# Default: gunakan standard
tfidf_matrix = compute_tfidf_standard(documents, vocabulary, idf)
current_scheme = "standard"

def set_weighting_scheme(scheme="standard"):
    """
    Switch antara skema weighting
    
    Args:
        scheme: "standard" atau "sublinear"
    """
    global tfidf_matrix, current_scheme
    
    if scheme == "standard":
        tfidf_matrix = compute_tfidf_standard(documents, vocabulary, idf)
        current_scheme = "standard"
        print(f"✅ Using TF-IDF Standard")
    elif scheme == "sublinear":
        tfidf_matrix = compute_tfidf_sublinear(documents, vocabulary, idf)
        current_scheme = "sublinear"
        print(f"✅ Using TF-IDF Sublinear (log-scaled)")
    else:
        print(f"❌ Unknown scheme: {scheme}")
    
    print(f"   Matrix shape: {tfidf_matrix.shape}")
    print(f"   Non-zero: {tfidf_matrix.nnz}")
    print(f"   Sparsity: {(1 - tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1])) * 100:.2f}%")

# ==============================================================  
# 6️⃣ Fungsi Query Processing
# ==============================================================

def process_query(query, scheme="standard"):
    """
    Convert query string menjadi TF-IDF vector
    
    Args:
        query (str): query string
        scheme (str): "standard" atau "sublinear"
    
    Returns:
        np.array: TF-IDF vector untuk query
    """
    tokens = query.lower().split()
    tf = Counter(tokens)
    
    if scheme == "sublinear":
        q_vec = np.array([
            ((1 + log10(tf.get(term, 0))) if tf.get(term, 0) > 0 else 0) * idf.get(term, 0.0) 
            for term in vocabulary
        ])
    else:  # standard
        q_vec = np.array([tf.get(term, 0) * idf.get(term, 0.0) for term in vocabulary])
    
    return q_vec

# ==============================================================  
# 7️⃣ VSM Search dengan Cosine Similarity
# ==============================================================

def search_vsm(query, top_k=5, verbose=True):
    """
    Search menggunakan Vector Space Model
    
    Args:
        query (str): query string
        top_k (int): jumlah top documents yang dikembalikan
        verbose (bool): tampilkan detail proses
    
    Returns:
        list: list of dict dengan doc_id, score, snippet
    """
    if verbose:
        print(f"\n🔍 Query: '{query}' (scheme: {current_scheme})")
    
    # Process query ke TF-IDF vector (sesuai current scheme)
    q_vec = process_query(query, scheme=current_scheme).reshape(1, -1)

    if np.all(q_vec == 0):
        print("⚠️ Query tidak ada di vocabulary. Tidak bisa dihitung similarity.")
        return []
    
    # Hitung cosine similarity
    cos_sim = cosine_similarity(tfidf_matrix, q_vec).flatten()
    
    # Ranking berdasarkan cosine similarity (descending)
    ranked_idx = np.argsort(-cos_sim)[:top_k]
    
    results = []
    for rank, idx in enumerate(ranked_idx, 1):
        doc_id = filenames[idx]
        score = cos_sim[idx]
        
        # Snippet maksimal 120 karakter
        full_text = " ".join(documents[doc_id])
        snippet = full_text[:120].replace("\n", " ")
        if len(full_text) > 120:
            snippet += "..."
        
        results.append({
            "rank": rank,
            "doc_id": doc_id,
            "score": score,
            "snippet": snippet
        })
        
        if verbose:
            print(f"   {rank}. {doc_id:<12} | Score: {score:.4f}")
    
    return results

# ==============================================================  
# 8️⃣ Evaluasi: Precision@k, Recall@k, MAP@k
# ==============================================================

def precision_at_k(retrieved, relevant, k):
    """Hitung Precision@k"""
    retrieved_k = set([r['doc_id'] for r in retrieved[:k]])
    if not retrieved_k:
        return 0.0
    return len(retrieved_k & relevant) / len(retrieved_k)

def recall_at_k(retrieved, relevant, k):
    """Hitung Recall@k"""
    retrieved_k = set([r['doc_id'] for r in retrieved[:k]])
    if not relevant:
        return 0.0
    return len(retrieved_k & relevant) / len(relevant)

def average_precision(retrieved, relevant):
    """
    Hitung Average Precision (AP) untuk satu query
    
    AP = (sum of P@k for each relevant doc) / total relevant docs
    """
    if not relevant:
        return 0.0
    
    ap = 0.0
    num_relevant_found = 0
    
    for k, result in enumerate(retrieved, 1):
        if result['doc_id'] in relevant:
            num_relevant_found += 1
            precision_k = num_relevant_found / k
            ap += precision_k
    
    return ap / len(relevant) if len(relevant) > 0 else 0.0

def mean_average_precision(results_dict, truth_set):
    """
    Hitung Mean Average Precision (MAP) untuk semua queries
    
    Args:
        results_dict: dict {query: [results]}
        truth_set: dict {query: set(relevant_docs)}
    
    Returns:
        float: MAP score
    """
    aps = []
    for query, relevant in truth_set.items():
        if query in results_dict:
            ap = average_precision(results_dict[query], relevant)
            aps.append(ap)
    
    return np.mean(aps) if aps else 0.0

# ==============================================================  
# 9️⃣ Truth Set (Gold Standard) - Disesuaikan dengan Dokumen
# ==============================================================

# ✅ Truth Set berdasarkan konten dokumen review makanan/kafe
truth_set = {
    # Query VSM lebih flexible, bisa multi-term
    "ayam enak sambal": {"doc1.txt", "doc3.txt", "doc7.txt"},
    
    "kopi enak nongkrong": {"doc2.txt", "doc10.txt"},
    
    "tempat bagus foto": {"doc4.txt"},
    
    "manis dessert": {"doc5.txt"},
    
    "lama antri staff": {"doc6.txt", "doc9.txt"},
    
    "kremes gurih ayam": {"doc1.txt", "doc7.txt", "doc10.txt"},
}

# ==============================================================  
# 🔟 Evaluasi dengan Truth Set
# ==============================================================

def run_evaluation(top_k=5, verbose=False):
    """
    Jalankan evaluasi VSM dengan truth set
    
    Args:
        top_k (int): berapa banyak top documents untuk evaluasi
        verbose (bool): tampilkan detail per query
    
    Returns:
        tuple: (DataFrame results, MAP score)
    """
    if verbose:
        print("\n" + "="*60)
        print(f"📊 EVALUASI VSM - Scheme: {current_scheme.upper()}")
        print("="*60)
    
    results_dict = {}
    eval_results = []
    
    for query, relevant_docs in truth_set.items():
        if verbose:
            print(f"\n{'─'*60}")
            print(f"Query: '{query}'")
            print(f"Gold Standard ({len(relevant_docs)} docs): {sorted(relevant_docs)}")
        
        # Retrieve top-k
        retrieved = search_vsm(query, top_k=top_k, verbose=verbose)
        results_dict[query] = retrieved
        
        # Hitung metrics
        prec_k = precision_at_k(retrieved, relevant_docs, top_k)
        recall_k = recall_at_k(retrieved, relevant_docs, top_k)
        ap = average_precision(retrieved, relevant_docs)
        
        if verbose:
            print(f"\n📈 Metrics (k={top_k}):")
            print(f"   Precision@{top_k}: {prec_k:.3f}")
            print(f"   Recall@{top_k}:    {recall_k:.3f}")
            print(f"   AP:           {ap:.3f}")
        
        eval_results.append({
            'query': query,
            f'P@{top_k}': prec_k,
            f'R@{top_k}': recall_k,
            'AP': ap,
            'retrieved': len(retrieved),
            'relevant': len(relevant_docs)
        })
    
    # Hitung MAP
    map_score = mean_average_precision(results_dict, truth_set)
    
    df_results = pd.DataFrame(eval_results)
    
    if verbose:
        # Summary
        print(f"\n{'='*60}")
        print("📊 SUMMARY EVALUASI")
        print("="*60)
        print(df_results.to_string(index=False))
        print(f"\n📌 Average Metrics:")
        print(f"   Mean Precision@{top_k}: {df_results[f'P@{top_k}'].mean():.3f}")
        print(f"   Mean Recall@{top_k}:    {df_results[f'R@{top_k}'].mean():.3f}")
        print(f"   MAP@{top_k}:            {map_score:.3f}")
    
    return df_results, map_score

# ==============================================================  
# ⭐ SOAL 05 POIN 1: Comparison Term Weighting Schemes
# ==============================================================

def compare_weighting_schemes(top_k=5):
    """
    Bandingkan 2 skema term weighting: Standard vs Sublinear
    Laporkan pengaruhnya terhadap metrik evaluasi
    """
    print("\n" + "="*70)
    print("⭐ SOAL 05 - COMPARISON: TERM WEIGHTING SCHEMES")
    print("="*70)
    print("\nMembandingkan 2 skema:")
    print("1️⃣  TF-IDF Standard     : TF = raw count")
    print("2️⃣  TF-IDF Sublinear    : TF = 1 + log10(count)")
    print("="*70)
    
    # Evaluasi Skema 1: Standard
    print("\n" + "─"*70)
    print("📊 EVALUASI SKEMA 1: TF-IDF STANDARD")
    print("─"*70)
    set_weighting_scheme("standard")
    results_std, map_std = run_evaluation(top_k=top_k, verbose=True)
    
    # Evaluasi Skema 2: Sublinear
    print("\n" + "─"*70)
    print("📊 EVALUASI SKEMA 2: TF-IDF SUBLINEAR")
    print("─"*70)
    set_weighting_scheme("sublinear")
    results_sub, map_sub = run_evaluation(top_k=top_k, verbose=True)
    
    # Comparison Summary
    print("\n" + "="*70)
    print("📊 COMPARISON SUMMARY")
    print("="*70)
    
    comparison_df = pd.DataFrame({
        'Metric': [f'Mean P@{top_k}', f'Mean R@{top_k}', f'MAP@{top_k}'],
        'Standard': [
            results_std[f'P@{top_k}'].mean(),
            results_std[f'R@{top_k}'].mean(),
            map_std
        ],
        'Sublinear': [
            results_sub[f'P@{top_k}'].mean(),
            results_sub[f'R@{top_k}'].mean(),
            map_sub
        ]
    })
    
    # Tambahkan kolom delta (improvement)
    comparison_df['Delta'] = comparison_df['Sublinear'] - comparison_df['Standard']
    comparison_df['Improvement (%)'] = (comparison_df['Delta'] / comparison_df['Standard'] * 100).round(2)
    
    print(comparison_df.to_string(index=False))
    
    # Kesimpulan
    print("\n" + "="*70)
    print("💡 ANALISIS & KESIMPULAN")
    print("="*70)
    
    if map_sub > map_std:
        winner = "TF-IDF Sublinear"
        improvement = ((map_sub - map_std) / map_std * 100)
        print(f"✅ {winner} menunjukkan performa LEBIH BAIK")
        print(f"   MAP@{top_k} meningkat {improvement:.2f}% dibanding Standard")
        print(f"\n📝 Interpretasi:")
        print(f"   Sublinear scaling (log) mengurangi dominasi term yang sangat frequent,")
        print(f"   sehingga meningkatkan akurasi retrieval untuk dokumen relevan.")
    elif map_sub < map_std:
        winner = "TF-IDF Standard"
        decline = ((map_std - map_sub) / map_std * 100)
        print(f"✅ {winner} menunjukkan performa LEBIH BAIK")
        print(f"   MAP@{top_k} dari Sublinear turun {decline:.2f}%")
        print(f"\n📝 Interpretasi:")
        print(f"   Untuk korpus ini, raw TF count lebih efektif karena distribusi")
        print(f"   term frequency sudah balanced setelah preprocessing.")
    else:
        print(f"⚖️  Kedua skema menunjukkan performa SETARA")
        print(f"   MAP@{top_k}: {map_std:.3f}")
    
    print("="*70)
    
    # Reset ke standard untuk konsistensi
    set_weighting_scheme("standard")
    
    return comparison_df

# ==============================================================  
# 1️⃣1️⃣ Wrapper untuk Integrasi ke search_engine.py
# ==============================================================

def vsm_search(query, k=5):
    """Wrapper function untuk dipanggil dari search_engine.py"""
    print(f"\n{'='*60}")
    print(f"🔎 VSM SEARCH MODE")
    print(f"{'='*60}")
    results = search_vsm(query, top_k=k, verbose=True)
    
    print(f"\n📋 Top-{k} Results:")
    print(f"{'─'*60}")
    for r in results:
        print(f"{r['rank']}. {r['doc_id']:<12} | Score: {r['score']:.4f}")
        print(f"   Snippet: {r['snippet']}")
        print()
    
    return results

# ==============================================================  
# 1️⃣2️⃣ Main Execution
# ==============================================================

if __name__ == "__main__":
    print("="*60)
    print("🔎 VECTOR SPACE MODEL (VSM) - INFORMATION RETRIEVAL")
    print("="*60)
    
    # Run comparison untuk Soal 05 Poin 1
    comparison_results = compare_weighting_schemes(top_k=5)
    
    print(f"\n✅ Evaluation & Comparison completed!")