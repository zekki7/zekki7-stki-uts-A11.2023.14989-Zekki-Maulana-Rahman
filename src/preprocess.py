import os
import string
import re
import json
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# ==============================================================
# Inisialisasi Stemmer & Stopword
# ==============================================================

stemmer = StemmerFactory().create_stemmer()
stop_factory = StopWordRemoverFactory()
stopwords = set(stop_factory.get_stop_words())

# Ambil folder utama proyek
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Folder input/output sesuai soal
input_path = os.path.join(BASE_DIR, "data", "raw")
output_path = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(output_path, exist_ok=True)

# ==============================================================
# Fungsi-fungsi preprocessing
# ==============================================================

def clean(text):
    """Case folding, hapus angka dan tanda baca."""
    text = text.lower()
    text = re.sub(r'\d+', ' ', text)  # hapus angka
    text = text.translate(str.maketrans("", "", string.punctuation))  # hapus tanda baca
    return text

def tokenize(text):
    """Pisahkan teks jadi token (list kata)."""
    return text.split()

def remove_stopwords(tokens):
    """Hilangkan stopwords Bahasa Indonesia."""
    return [t for t in tokens if t not in stopwords]

def stem(tokens):
    """Lakukan stemming ke bentuk dasar."""
    return [stemmer.stem(t) for t in tokens]

# ==============================================================
# Main process
# ==============================================================

if not os.path.exists(input_path):
    print(f"❌ Folder tidak ditemukan: {input_path}")
    exit()

print("="*60)
print("🔄 MEMULAI PREPROCESSING DOKUMEN")
print("="*60)

# List untuk menyimpan statistik
processing_stats = []
files_processed = []

for filename in os.listdir(input_path):
    if filename.endswith(".txt"):
        with open(os.path.join(input_path, filename), "r", encoding="utf-8") as f:
            text = f.read()

        # Tahapan sesuai rubrik
        original_length = len(text)
        
        cleaned = clean(text)
        tokens = tokenize(cleaned)
        tokens_before_filter = len(tokens)
        
        filtered = remove_stopwords(tokens)
        tokens_after_filter = len(filtered)
        
        stemmed = stem(filtered)
        final_tokens = len(stemmed)
        unique_tokens = len(set(stemmed))

        final_text = " ".join(stemmed)

        # Simpan hasil
        with open(os.path.join(output_path, filename), "w", encoding="utf-8") as f:
            f.write(final_text)

        # Simpan statistik
        stats = {
            'filename': filename,
            'original_chars': original_length,
            'tokens_after_tokenize': tokens_before_filter,
            'tokens_after_stopword': tokens_after_filter,
            'final_tokens': final_tokens,
            'unique_tokens': unique_tokens,
            'stopwords_removed': tokens_before_filter - tokens_after_filter,
            'reduction_rate': round((1 - final_tokens/tokens_before_filter) * 100, 2) if tokens_before_filter > 0 else 0
        }
        processing_stats.append(stats)
        files_processed.append(filename)

        print(f"✅ {filename:<20} | Original: {original_length:>6} chars | Final: {final_tokens:>4} tokens | Unique: {unique_tokens:>4}")

# ==============================================================
# Simpan Log Ringkas ke JSON
# ==============================================================

log_file = os.path.join(output_path, "preprocessing_log.json")
with open(log_file, "w", encoding="utf-8") as f:
    json.dump({
        'summary': {
            'total_files': len(files_processed),
            'total_stopwords': len(stopwords),
            'input_path': input_path,
            'output_path': output_path
        },
        'files': processing_stats
    }, f, indent=2, ensure_ascii=False)

print("\n" + "="*60)
print("📊 PREPROCESSING SUMMARY")
print("="*60)
print(f"✅ Total dokumen diproses  : {len(files_processed)}")
print(f"📁 Input folder            : {input_path}")
print(f"📁 Output folder           : {output_path}")
print(f"📄 Log file                : {log_file}")

if processing_stats:
    total_original = sum(s['original_chars'] for s in processing_stats)
    total_final = sum(s['final_tokens'] for s in processing_stats)
    total_unique = sum(s['unique_tokens'] for s in processing_stats)
    avg_reduction = sum(s['reduction_rate'] for s in processing_stats) / len(processing_stats)
    
    print(f"\n📈 Statistik Agregat:")
    print(f"   Total karakter original : {total_original:,}")
    print(f"   Total tokens final      : {total_final:,}")
    print(f"   Total unique tokens     : {total_unique:,}")
    print(f"   Rata-rata reduksi       : {avg_reduction:.2f}%")

print("="*60)
print("✅ Preprocessing selesai! Hasil disimpan di folder data/processed/")
print("="*60)