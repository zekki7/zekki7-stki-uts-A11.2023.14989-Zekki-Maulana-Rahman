# src/eval.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

# Import dari modul yang sudah dibuat
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from boolean_ir import boolean_retrieval, truth_set as boolean_truth_set, evaluate as boolean_evaluate
from vsm_ir import search_vsm, set_weighting_scheme, truth_set as vsm_truth_set
from vsm_ir import precision_at_k, recall_at_k, average_precision, mean_average_precision

# =========================================================
# 📊 EVALUATION METRICS (Tambahan untuk Boolean)
# =========================================================

def f1_score(precision, recall):
    """Hitung F1-Score dari precision dan recall"""
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def ndcg_at_k(retrieved_docs, relevant_docs, k):
    """
    Hitung nDCG@k (Normalized Discounted Cumulative Gain)
    
    Args:
        retrieved_docs: list of doc_ids atau list of dicts dengan 'doc_id'
        relevant_docs: set of relevant doc_ids
        k: cutoff rank
    
    Returns:
        float: nDCG@k score
    """
    # Pastikan k adalah integer
    k = int(k) if k else 5
    
    # Handle jika retrieved_docs adalah list of dicts (dari VSM)
    if retrieved_docs and isinstance(retrieved_docs[0], dict):
        retrieved_docs = [r['doc_id'] for r in retrieved_docs]
    
    # Hitung DCG@k
    dcg = 0.0
    for i, doc in enumerate(retrieved_docs[:k]):
        if doc in relevant_docs:
            # Relevance = 1 jika relevan, 0 jika tidak
            dcg += 1.0 / np.log2(i + 2)  # i+2 karena rank dimulai dari 1
    
    # Hitung IDCG@k (ideal DCG)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(relevant_docs))))
    
    # nDCG = DCG / IDCG
    return dcg / idcg if idcg > 0 else 0.0


# =========================================================
# 🔍 EVALUASI BOOLEAN RETRIEVAL
# =========================================================

def evaluate_boolean_model(truth_set=None, verbose=True):
    """
    Evaluasi Boolean Retrieval Model dengan truth set
    
    Args:
        truth_set: dict {query: set(relevant_docs)}
        verbose: tampilkan detail per query
    
    Returns:
        pd.DataFrame: hasil evaluasi
    """
    if truth_set is None:
        truth_set = boolean_truth_set
    
    if verbose:
        print("\n" + "="*70)
        print("📊 EVALUASI MODEL: BOOLEAN RETRIEVAL")
        print("="*70)
    
    results = []
    
    for query, gold_docs in truth_set.items():
        if verbose:
            print(f"\n{'─'*70}")
            print(f"Query: '{query}'")
            print(f"Gold Standard ({len(gold_docs)} docs): {sorted(gold_docs)}")
        
        # Retrieve
        retrieved = boolean_retrieval(query, verbose=False)
        
        # Evaluate
        precision, recall, f1 = boolean_evaluate(retrieved, gold_docs)
        
        if verbose:
            print(f"Retrieved ({len(retrieved)} docs): {sorted(retrieved)}")
            print(f"\n📈 Metrics:")
            print(f"   Precision: {precision:.3f}")
            print(f"   Recall:    {recall:.3f}")
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
    
    df_results = pd.DataFrame(results)
    
    if verbose:
        print(f"\n{'='*70}")
        print("📊 SUMMARY - BOOLEAN RETRIEVAL")
        print("="*70)
        print(df_results.to_string(index=False))
        print(f"\n📌 Average Metrics:")
        print(f"   Precision: {df_results['precision'].mean():.3f}")
        print(f"   Recall:    {df_results['recall'].mean():.3f}")
        print(f"   F1-Score:  {df_results['f1'].mean():.3f}")
        print("="*70)
    
    return df_results


# =========================================================
# 🔍 EVALUASI VSM RETRIEVAL (Single Scheme)
# =========================================================

def evaluate_vsm_model(weighting="standard", top_k=5, truth_set=None, verbose=True):
    """
    Evaluasi VSM Retrieval Model dengan scheme tertentu
    
    Args:
        weighting: "standard" atau "sublinear"
        top_k: jumlah top documents
        truth_set: dict {query: set(relevant_docs)}
        verbose: tampilkan detail per query
    
    Returns:
        tuple: (DataFrame results, MAP score)
    """
    # ✅ Pastikan top_k adalah integer
    top_k = int(top_k) if top_k else 5
    
    if truth_set is None:
        truth_set = vsm_truth_set
    
    # Set weighting scheme
    set_weighting_scheme(weighting)
    
    if verbose:
        print("\n" + "="*70)
        print(f"📊 EVALUASI MODEL: VSM - {weighting.upper()}")
        print("="*70)
    
    results_dict = {}
    eval_results = []
    
    for query, relevant_docs in truth_set.items():
        if verbose:
            print(f"\n{'─'*70}")
            print(f"Query: '{query}'")
            print(f"Gold Standard ({len(relevant_docs)} docs): {sorted(relevant_docs)}")
        
        # Retrieve top-k
        retrieved = search_vsm(query, top_k=top_k, verbose=False)
        results_dict[query] = retrieved
        
        # Hitung metrics
        prec_k = precision_at_k(retrieved, relevant_docs, top_k)
        recall_k = recall_at_k(retrieved, relevant_docs, top_k)
        f1_k = f1_score(prec_k, recall_k)
        ap = average_precision(retrieved, relevant_docs)
        ndcg_k = ndcg_at_k(retrieved, relevant_docs, top_k)
        
        if verbose:
            retrieved_ids = [r['doc_id'] for r in retrieved]
            print(f"Retrieved top-{top_k}: {retrieved_ids}")
            print(f"\n📈 Metrics (k={top_k}):")
            print(f"   Precision@{top_k}: {prec_k:.3f}")
            print(f"   Recall@{top_k}:    {recall_k:.3f}")
            print(f"   F1@{top_k}:        {f1_k:.3f}")
            print(f"   AP:                {ap:.3f}")
            print(f"   nDCG@{top_k}:      {ndcg_k:.3f}")
        
        eval_results.append({
            'query': query,
            f'P@{top_k}': prec_k,
            f'R@{top_k}': recall_k,
            f'F1@{top_k}': f1_k,
            'AP': ap,
            f'nDCG@{top_k}': ndcg_k,
            'retrieved': len(retrieved),
            'relevant': len(relevant_docs)
        })
    
    # Hitung MAP
    map_score = mean_average_precision(results_dict, truth_set)
    
    df_results = pd.DataFrame(eval_results)
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"📊 SUMMARY - VSM {weighting.upper()}")
        print("="*70)
        print(df_results.to_string(index=False))
        print(f"\n📌 Average Metrics:")
        print(f"   Mean Precision@{top_k}: {df_results[f'P@{top_k}'].mean():.3f}")
        print(f"   Mean Recall@{top_k}:    {df_results[f'R@{top_k}'].mean():.3f}")
        print(f"   Mean F1@{top_k}:        {df_results[f'F1@{top_k}'].mean():.3f}")
        print(f"   MAP@{top_k}:            {map_score:.3f}")
        print(f"   Mean nDCG@{top_k}:      {df_results[f'nDCG@{top_k}'].mean():.3f}")
        print("="*70)
    
    return df_results, map_score


# =========================================================
# ⚖️ COMPARISON: VSM WEIGHTING SCHEMES
# =========================================================

def compare_vsm_schemes(top_k=5, truth_set=None, verbose=False):
    """
    Bandingkan performa 2 skema weighting VSM
    
    Args:
        top_k: jumlah top documents (integer)
        truth_set: dict {query: set(relevant_docs)}
        verbose: tampilkan detail
    
    Returns:
        pd.DataFrame: comparison results
    """
    # ✅ PERBAIKAN: Pastikan top_k adalah integer
    top_k = int(top_k) if top_k else 5
    
    if truth_set is None:
        truth_set = vsm_truth_set
    
    if verbose:
        print("\n" + "="*70)
        print("⚖️  COMPARISON: VSM WEIGHTING SCHEMES")
        print("="*70)
        print(f"Comparing: TF-IDF Standard vs TF-IDF Sublinear")
        print(f"Top-K: {top_k}")
        print(f"Queries: {len(truth_set)}")
        print("="*70)
    
    # Evaluasi kedua scheme
    results_std, map_std = evaluate_vsm_model("standard", top_k, truth_set, verbose=False)
    results_sub, map_sub = evaluate_vsm_model("sublinear", top_k, truth_set, verbose=False)
    
    # Comparison table
    comparison_df = pd.DataFrame({
        'Metric': [
            f'Mean P@{top_k}', 
            f'Mean R@{top_k}', 
            f'Mean F1@{top_k}',
            f'MAP@{top_k}',
            f'Mean nDCG@{top_k}'
        ],
        'Standard': [
            results_std[f'P@{top_k}'].mean(),
            results_std[f'R@{top_k}'].mean(),
            results_std[f'F1@{top_k}'].mean(),
            map_std,
            results_std[f'nDCG@{top_k}'].mean()
        ],
        'Sublinear': [
            results_sub[f'P@{top_k}'].mean(),
            results_sub[f'R@{top_k}'].mean(),
            results_sub[f'F1@{top_k}'].mean(),
            map_sub,
            results_sub[f'nDCG@{top_k}'].mean()
        ]
    })
    
    # Tambahkan kolom delta & improvement
    comparison_df['Delta'] = comparison_df['Sublinear'] - comparison_df['Standard']
    comparison_df['Improvement (%)'] = (
        (comparison_df['Delta'] / comparison_df['Standard']) * 100
    ).round(2)
    
    if verbose:
        print("\n" + "="*70)
        print("📊 COMPARISON SUMMARY")
        print("="*70)
        print(comparison_df.to_string(index=False))
        
        # Analisis
        print("\n" + "="*70)
        print("💡 ANALISIS & INTERPRETASI")
        print("="*70)
        
        if map_sub > map_std:
            winner = "TF-IDF Sublinear"
            improvement = ((map_sub - map_std) / map_std * 100)
            print(f"✅ {winner} menunjukkan performa LEBIH BAIK")
            print(f"   MAP@{top_k} meningkat {improvement:.2f}% dibanding Standard")
            print(f"\n🔍 Interpretasi:")
            print(f"   Sublinear scaling (1 + log TF) mengurangi dominasi term yang")
            print(f"   sangat frequent, sehingga meningkatkan akurasi retrieval.")
        elif map_sub < map_std:
            winner = "TF-IDF Standard"
            decline = ((map_std - map_sub) / map_std * 100)
            print(f"✅ {winner} menunjukkan performa LEBIH BAIK")
            print(f"   MAP@{top_k} dari Sublinear turun {decline:.2f}%")
            print(f"\n🔍 Interpretasi:")
            print(f"   Untuk korpus ini, raw TF count lebih efektif karena distribusi")
            print(f"   term frequency sudah balanced setelah preprocessing.")
        else:
            print(f"⚖️  Kedua skema menunjukkan performa SETARA")
            print(f"   MAP@{top_k}: {map_std:.3f}")
        
        print("="*70)
    
    return comparison_df


# =========================================================
# 📊 EVALUASI KOMPREHENSIF (Boolean + VSM)
# =========================================================

def evaluate_all_models(top_k=5, verbose=True):
    """
    Evaluasi semua model: Boolean, VSM Standard, VSM Sublinear
    
    Args:
        top_k: jumlah top documents untuk VSM
        verbose: tampilkan detail
    
    Returns:
        dict: hasil evaluasi semua model
    """
    top_k = int(top_k) if top_k else 5
    
    print("\n" + "="*70)
    print("🎯 EVALUASI KOMPREHENSIF - SEMUA MODEL")
    print("="*70)
    print(f"Models: Boolean, VSM Standard, VSM Sublinear")
    print(f"VSM Top-K: {top_k}")
    print("="*70)
    
    results = {}
    
    # 1. Boolean Retrieval
    print("\n" + "─"*70)
    print("1️⃣  BOOLEAN RETRIEVAL")
    print("─"*70)
    df_bool = evaluate_boolean_model(verbose=verbose)
    results['boolean'] = df_bool
    
    # 2. VSM Standard
    print("\n" + "─"*70)
    print("2️⃣  VSM - STANDARD TF-IDF")
    print("─"*70)
    df_vsm_std, map_std = evaluate_vsm_model("standard", top_k, verbose=verbose)
    results['vsm_standard'] = {'df': df_vsm_std, 'map': map_std}
    
    # 3. VSM Sublinear
    print("\n" + "─"*70)
    print("3️⃣  VSM - SUBLINEAR TF-IDF")
    print("─"*70)
    df_vsm_sub, map_sub = evaluate_vsm_model("sublinear", top_k, verbose=verbose)
    results['vsm_sublinear'] = {'df': df_vsm_sub, 'map': map_sub}
    
    # Summary comparison
    print("\n" + "="*70)
    print("📊 OVERALL SUMMARY")
    print("="*70)
    
    summary = pd.DataFrame({
        'Model': ['Boolean', 'VSM Standard', 'VSM Sublinear'],
        'Precision': [
            df_bool['precision'].mean(),
            df_vsm_std[f'P@{top_k}'].mean(),
            df_vsm_sub[f'P@{top_k}'].mean()
        ],
        'Recall': [
            df_bool['recall'].mean(),
            df_vsm_std[f'R@{top_k}'].mean(),
            df_vsm_sub[f'R@{top_k}'].mean()
        ],
        'F1': [
            df_bool['f1'].mean(),
            df_vsm_std[f'F1@{top_k}'].mean(),
            df_vsm_sub[f'F1@{top_k}'].mean()
        ],
        'MAP': [
            '-',  # Boolean tidak pakai MAP
            f"{map_std:.3f}",
            f"{map_sub:.3f}"
        ]
    })
    
    print(summary.to_string(index=False))
    print("="*70)
    
    return results


# =========================================================
# 🔗 ALIAS untuk backward compatibility
# =========================================================
compare_schemes = compare_vsm_schemes


# =========================================================
# 🚀 MAIN EXECUTION
# =========================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("📊 SISTEM EVALUASI - MINI SEARCH ENGINE STKI")
    print("="*70)
    
    # Menu pilihan
    print("\nPilih mode evaluasi:")
    print("1. Evaluasi Boolean Retrieval")
    print("2. Evaluasi VSM Standard")
    print("3. Evaluasi VSM Sublinear")
    print("4. Comparison VSM Schemes")
    print("5. Evaluasi Semua Model (Komprehensif)")
    print("="*70)
    
    try:
        choice = input("\nMasukkan pilihan (1-5) [default: 5]: ").strip()
        if not choice:
            choice = "5"
        
        top_k = 5
        
        if choice == "1":
            evaluate_boolean_model(verbose=True)
        
        elif choice == "2":
            evaluate_vsm_model("standard", top_k=top_k, verbose=True)
        
        elif choice == "3":
            evaluate_vsm_model("sublinear", top_k=top_k, verbose=True)
        
        elif choice == "4":
            compare_vsm_schemes(top_k=top_k, verbose=True)
        
        elif choice == "5":
            evaluate_all_models(top_k=top_k, verbose=True)
        
        else:
            print("❌ Pilihan tidak valid!")
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Evaluasi dibatalkan oleh user")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()