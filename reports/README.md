# STKI UTS – A11.2023.14989 Zekki Maulana Rahman

Aplikasi ini merupakan implementasi sistem **Information Retrieval (IR)** menggunakan **Vector Space Model (VSM)** dan **Boolean Retrieval** untuk mata kuliah *Sistem Temu Kembali Informasi (STKI)*.

---

## Deploy Link
Aplikasi dapat dijalankan langsung melalui Streamlit Cloud pada link berikut:

**[https://zekki7-stki-uts-a11202314989-zekki-maulana-rahman-2mcvbtsbktao.streamlit.app/](https://zekki7-stki-uts-a11202314989-zekki-maulana-rahman-2mcvbtsbktao.streamlit.app/)**

---

## Asumsi Proyek
- Dataset berupa dokumen teks (`.txt`) yang disimpan di folder `data/`.
- Setiap file dalam folder `data/` dianggap sebagai satu dokumen.
- Aplikasi menampilkan hasil pencarian berdasarkan *similarity score* (VSM) atau hasil *Boolean matching*.
stki-uts-A11.2023.14989-Zekki Maulana Rahman>/
├─ data/ raw dan prosesed
├─ src/
│ ├─ preprocess.py
│ ├─ boolean_ir.py
│ ├─ vsm_ir.py
│ ├─ search.py 
│ └─ eval.py
├─ app/
│ └─ main.py 
├─ notebooks/
│ └─ UTS_STKI_A11.2023.14989.ipynb
├─ reports/
│ ├─ laporan.pdf 
│ └─ readme.md dan Esai.pdf
└─ requirements.txt
