# src/search_engine.py

import argparse
import sys
import os

# Import dari modul yang sudah dibuat
from boolean_ir import boolean_retrieval, documents as bool_docs
from vsm_ir import search_vsm, set_weighting_scheme, documents, vocabulary

def corpus_statistics():
    """
    Tampilkan statistik corpus yang sudah di-load
    
    Returns:
        dict: informasi corpus
    """
    stats = {
        'total_documents': len(documents),
        'vocabulary_size': len(vocabulary),
        'document_list': sorted(documents.keys()),
        'avg_doc_length': sum(len(doc) for doc in documents.values()) / len(documents) if documents else 0
    }
    
    print(f"\n{'='*70}")
    print(f"📚 CORPUS STATISTICS")
    print(f"{'='*70}")
    print(f"Total Documents    : {stats['total_documents']}")
    print(f"Vocabulary Size    : {stats['vocabulary_size']} unique terms")
    print(f"Avg Document Length: {stats['avg_doc_length']:.1f} terms")
    print(f"Document List      : {', '.join(stats['document_list'][:5])}...")
    print(f"{'='*70}")
    
    return stats


# =========================================================
# 🔍 SEARCH ENGINE ORCHESTRATOR
# =========================================================

def run_boolean_search(query, verbose=True):
    """
    Jalankan Boolean Search dengan explain lengkap
    
    Args:
        query (str): Boolean query (support AND/OR/NOT)
        verbose (bool): tampilkan step-by-step explanation
    
    Returns:
        set: dokumen hasil retrieval
    """
    print(f"\n{'='*70}")
    print(f"🔎 BOOLEAN RETRIEVAL MODE")
    print(f"{'='*70}")
    print(f"Query: '{query}'")
    print(f"{'─'*70}")
    
    # Panggil fungsi dari boolean_ir.py dengan verbose=True untuk explain
    result = boolean_retrieval(query, verbose=verbose)
    
    print(f"\n{'─'*70}")
    print(f"✅ FINAL RESULT: {len(result)} dokumen ditemukan")
    if result:
        print(f"📄 Dokumen: {sorted(result)}")
    else:
        print(f"📄 Tidak ada dokumen yang memenuhi query")
    print(f"{'='*70}")
    
    return result


def run_vsm_search(query, top_k=5, weighting="standard", verbose=True):
    """
    Jalankan VSM Search dengan TF-IDF
    
    Args:
        query (str): free text query
        top_k (int): jumlah top documents
        weighting (str): "standard" atau "sublinear"
        verbose (bool): tampilkan detail hasil
    
    Returns:
        list: hasil retrieval dengan score dan snippet
    """
    print(f"\n{'='*70}")
    print(f"🔎 VECTOR SPACE MODEL (VSM) - TF-IDF")
    print(f"{'='*70}")
    print(f"Query: '{query}'")
    print(f"Top-K: {top_k}")
    print(f"Weighting Scheme: {weighting.upper()}")
    print(f"{'─'*70}")
    
    # Set weighting scheme sesuai parameter
    set_weighting_scheme(weighting)
    
    # Panggil fungsi dari vsm_ir.py
    results = search_vsm(query, top_k=top_k, verbose=verbose)
    
    if results:
        print(f"\n{'─'*70}")
        print(f"📊 TOP-{top_k} RESULTS")
        print(f"{'─'*70}")
        
        for r in results:
            print(f"\n{r['rank']}. {r['doc_id']:<15} | Score: {r['score']:.4f}")
            print(f"   📝 Snippet: {r['snippet']}")
            
            # Tambahan: Top terms yang berkontribusi
            doc_name = r['doc_id']
            if doc_name in documents:
                doc_tokens = documents[doc_name]
                query_tokens = set(query.lower().split())
                
                # Ambil term yang ada di query DAN di dokumen
                matching_terms = [t for t in query_tokens if t in doc_tokens]
                if matching_terms:
                    print(f"   🔑 Matching terms: {', '.join(matching_terms[:5])}")
    else:
        print(f"\n⚠️  Tidak ada dokumen relevan ditemukan")
    
    print(f"{'='*70}")
    
    return results


def compare_weighting_schemes(query, top_k=5):
    """
    Bandingkan 2 skema weighting untuk query yang sama
    
    Args:
        query (str): query string
        top_k (int): jumlah top documents
    """
    print(f"\n{'='*70}")
    print(f"⚖️  COMPARISON: WEIGHTING SCHEMES")
    print(f"{'='*70}")
    print(f"Query: '{query}'")
    print(f"Comparing: TF-IDF Standard vs TF-IDF Sublinear")
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
    
    print(f"\n{'='*70}")
    print(f"💡 INSIGHT:")
    print(f"{'─'*70}")
    print(f"Standard TF-IDF  : Raw term frequency × IDF")
    print(f"Sublinear TF-IDF : (1 + log(TF)) × IDF → mengurangi dominasi high-frequency terms")
    print(f"{'='*70}")
    
    return all_results


# =========================================================
# 🎯 COMMAND LINE INTERFACE
# =========================================================

def main():
    parser = argparse.ArgumentParser(
        description="🔍 Mini Search Engine - STKI Project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Boolean Search
  python search_engine.py --model boolean --query "ayam AND enak"
  
  # VSM Search (default: standard TF-IDF)
  python search_engine.py --model vsm --query "kopi enak" --k 5
  
  # VSM with sublinear weighting
  python search_engine.py --model vsm --query "kopi enak" --k 5 --weighting sublinear
  
  # Compare weighting schemes
  python search_engine.py --model vsm --query "ayam enak" --k 3 --compare
  
  # Demo mode (no arguments)
  python search_engine.py
        """
    )
    
    parser.add_argument(
        "--model", 
        choices=["boolean", "vsm"], 
        help="Model retrieval: 'boolean' atau 'vsm'"
    )
    
    parser.add_argument(
        "--query", 
        type=str,
        help="Query string"
    )
    
    parser.add_argument(
        "--k", 
        type=int, 
        default=5,
        help="Top-K documents untuk VSM (default: 5)"
    )
    
    parser.add_argument(
        "--weighting", 
        choices=["standard", "sublinear"], 
        default="standard",
        help="Weighting scheme untuk VSM: 'standard' atau 'sublinear' (default: standard)"
    )
    
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare 2 weighting schemes (hanya untuk VSM)"
    )
    
    args = parser.parse_args()
    
    # =========================================================
    # DEMO MODE - Jika tidak ada argumen
    # =========================================================
    if not args.model and not args.query:
        print("\n" + "="*70)
        print("🎬 DEMO MODE - MINI SEARCH ENGINE STKI")
        print("="*70)
        print(f"📚 Corpus: {len(documents)} dokumen")
        print(f"📖 Vocabulary: {len(vocabulary)} unique terms")
        print("="*70)
        
        demo_queries = [
            ("ayam enak sambal", "vsm"),
            ("ayam AND enak", "boolean"),
            ("kopi nongkrong", "vsm")
        ]
        
        for query, model in demo_queries:
            if model == "boolean":
                run_boolean_search(query)
            else:
                run_vsm_search(query, top_k=3, weighting="standard")
            
            print("\n")
        
        print("="*70)
        print("💡 TIP: Jalankan dengan --help untuk melihat usage lengkap")
        print("="*70)
        return
    
    # =========================================================
    # Validasi input
    # =========================================================
    if not args.query:
        print("❌ ERROR: --query harus diisi!")
        print("💡 Contoh: python search_engine.py --model vsm --query 'ayam enak'")
        sys.exit(1)
    
    if not args.model:
        print("❌ ERROR: --model harus diisi (boolean atau vsm)!")
        print("💡 Contoh: python search_engine.py --model vsm --query 'ayam enak'")
        sys.exit(1)
    
    # =========================================================
    # Eksekusi sesuai model
    # =========================================================
    
    if args.model == "boolean":
        if args.compare:
            print("⚠️  WARNING: --compare hanya berlaku untuk VSM model")
        
        run_boolean_search(args.query, verbose=True)
    
    elif args.model == "vsm":
        if args.compare:
            # Mode comparison
            compare_weighting_schemes(args.query, top_k=args.k)
        else:
            # Mode normal
            run_vsm_search(args.query, top_k=args.k, weighting=args.weighting, verbose=True)


# =========================================================
# 🚀 ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()