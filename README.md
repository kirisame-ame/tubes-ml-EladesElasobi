# Tubes ML - FFNN From Scratch - EladesElasobi

Repository ini berisi implementasi Feedforward Neural Network (FFNN) menggunakan NumPy, beserta notebook eksperimen untuk analisis:

- inisialisasi bobot
- regularisasi
- hyperparameter (depth, width, aktivasi, learning rate)
- MLP sklearn
- bonus RMSNorm
- bonus optimizer Adam

Dataset utama berada di folder `data/` dan seluruh eksperimen ada di folder `src/`.

## Struktur Repository

```text
tubes-ml-EladesElasobi/
|-- data/
|   |-- dataset.csv
|-- doc/
|   |-- references.md
|   |-- EladesElasobi-Tubes1ML.pdf
|-- src/
|   |-- ffnn.py
|   |-- test_ffnn.py
|   |-- testing.ipynb
|   |-- test_inisialisasi_bobot.ipynb
|   |-- test_regularization.ipynb
|   |-- test_hyperparameter.ipynb
|   |-- test_bonus_rmsnorm.ipynb
|   |-- test_bonus_adam.ipynb
`-- README.md
```

## Setup Environment

Disarankan menggunakan Python 3.10+.

1. Clone repository

```bash
git clone <url-repo>
cd tubes-ml-EladesElasobi
```

2. Buat virtual environment

```bash
python -m venv .venv
```

3. Aktivasi virtual environment

Windows (PowerShell):

```bash
.venv\Scripts\Activate.ps1
```

Windows (CMD):

```bash
.venv\Scripts\activate.bat
```

4. Install dependency

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter notebook
```

## Cara Menjalankan Program

### 1) Menjalankan test script

```bash
cd src
python test_ffnn.py
```

### 2) Menjalankan notebook eksperimen

```bash
cd src
jupyter notebook
```

Lalu jalankan notebook sesuai kebutuhan:

- `testing.ipynb` untuk baseline/testing awal
- `test_inisialisasi_bobot.ipynb` untuk eksperimen inisialisasi bobot
- `test_regularization.ipynb` untuk eksperimen regularisasi
- `test_hyperparameter.ipynb` untuk eksperimen depth/width/aktivasi/lr
- `test_bonus_rmsnorm.ipynb` untuk eksperimen RMSNorm
- `test_bonus_adam.ipynb` untuk eksperimen Adam optimizer

## Pembagian Tugas Anggota Kelompok

Isi tabel berikut sesuai nama dan NIM anggota:

| No | NIM | Tugas Utama |
|---|---|---|
| 1 | 13523006 | Semua |
| 2 | 13523008 | Semua |
| 3 | 13523032 | Semua |
