# Kaggle Learn — Intermediate Machine Learning

Catatan belajar lengkap, disusun per lesson sesuai urutan resmi course.

| Item | Keterangan |
|---|---|
| Course | Intermediate Machine Learning (`intermediate-machine-learning`) |
| Instructor | Alexis Cook |
| Estimasi waktu | 4 jam |
| Biaya | Gratis |
| Prasyarat | Intro to Machine Learning, Pandas |
| Lanjutan | Feature Engineering, Time Series |
| Deskripsi resmi | "Handle missing values, non-numeric values, data leakage, and more." |

**7 lesson:** 1) Introduction, 2) Missing Values, 3) Categorical Variables, 4) Pipelines, 5) Cross-Validation, 6) XGBoost, 7) Data Leakage.

> Catatan penyusunan: seluruh isi di bawah diambil dari halaman course dan notebook tutorial resmi Kaggle (7 tutorial + 7 exercise). Exercise **belum** dikerjakan — dokumen ini hanya materi belajar dan penjelasan alur kerjanya.

---

## Cara Kerja Course dan Exercise

Setiap lesson punya dua bagian: **Tutorial** (notebook bacaan, dijalankan sel demi sel) dan **Exercise** (notebook latihan yang harus diisi). Exercise memakai satu dataset yang sama sepanjang course: **Housing Prices Competition for Kaggle Learn Users** — 79 explanatory variable (tipe atap, jumlah kamar, jumlah kamar mandi, dst.) untuk memprediksi `SalePrice` rumah di Ames, Iowa.

**Struktur notebook exercise**

1. **Setup cell** — membuat symlink `../input/train.csv` dan `../input/test.csv` ke dataset kompetisi, lalu mengaktifkan sistem pengecekan jawaban:
   ```python
   from learntools.core import binder
   binder.bind(globals())
   from learntools.ml_intermediate.ex2 import *   # ex1..ex7 sesuai lesson
   ```
2. **Cell data loading** — dijalankan apa adanya (jangan diubah), mengisi `X_train`, `X_valid`, `y_train`, `y_valid`, `X_test`.
3. **Step 1..N** — ada placeholder `____` yang harus diisi. Setiap step punya `step_x.check()` untuk menilai jawaban, plus `step_x.hint()` dan `step_x.solution()` yang sengaja dikomentari (`#`) supaya tidak terbaca tanpa sengaja.
4. **Cell submission** — menyimpan `submission.csv` dengan kolom `Id` dan `SalePrice`.

**Alur submit prediksi ke leaderboard** (dari instruksi resmi exercise)

1. Buka [kompetisi Housing Prices](https://www.kaggle.com/c/home-data-for-ml-course), klik **Join Competition** (kalau tombolnya sudah bertuliskan "Submit Predictions", berarti sudah join).
2. Klik **Save Version** di kanan atas notebook.
3. Pilih **Save and Run All**, lalu **Save**. Proses berjalan di panel kiri bawah.
4. Setelah selesai, klik angka di sebelah kanan tombol **Save Version** → daftar versi muncul di kanan → klik elipsis **(...)** pada versi terbaru → **Open in Viewer**.
5. Di tampilan viewer, buka tab **Data**, pilih `submission.csv`, klik **Submit**.

**Sejauh mana submit wajib:** Exercise 1 dan 2 memakai submit sebagai bagian alur latihan; Exercise 3 (categorical) dan 4 (pipelines) menyebutnya eksplisit **opsional**; Exercise 5–7 tidak memerlukan submit sama sekali (hanya `check()` jawaban). Kelulusan course dihitung dari semua lesson + exercise selesai (100%), bukan dari peringkat leaderboard.

**Pola kerja yang disarankan:** baca tutorial → kerjakan exercise tanpa membuka solution → kalau `check()` gagal, baca `hint()` → setelah semua hijau, submit sekali untuk melihat posisi → baru boleh bandingkan dengan `solution()`.

---

## Introduction

Lesson pembuka tanpa konten teknis baru: meninjau ulang prasyarat dan memperkenalkan alur exercise + kompetisi.

### Poin inti

- Tujuan course: menangani tipe data dunia nyata (missing values, categorical variables), mendesain pipeline, memakai cross-validation untuk validasi model, membangun model state-of-the-art (XGBoost), dan menghindari kesalahan fatal (data leakage).
- Prasyarat: pernah membangun model ML, paham model validation, underfitting/overfitting, dan random forest. Kalau belum, ikuti Intro to Machine Learning dulu.
- Semua exercise memakai satu kompetisi: Housing Prices for Kaggle Learn Users, dengan 79 explanatory variable dan target `SalePrice`.
- Exercise 1 menguji dasar: memilih model terbaik dari 5 kandidat berdasarkan MAE pada validation set, lalu membuat prediksi test dan submit.
- Lima model yang dibandingkan: `n_estimators=50`, `n_estimators=100`, `n_estimators=100 + criterion='absolute_error'`, `n_estimators=200 + min_samples_split=20`, dan `n_estimators=100 + max_depth=7`.
- Metrik yang dipakai sepanjang course adalah **MAE** (mean absolute error) — semakin kecil semakin baik, satuannya sama dengan target (dolar).
- Alur submission: `submission.csv` berisi `Id` (dari index `X_test`) dan `SalePrice` (hasil `predict`), dibuat dengan `to_csv(..., index=False)`.
- Progres diukur dari peringkat leaderboard; improving model adalah iterasi, bukan sekali jadi.

### Contoh kode

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# 1. Muat data (di Kaggle: ../input/train.csv hasil symlink dataset kompetisi)
X_full = pd.read_csv('../input/train.csv', index_col='Id')
X_test_full = pd.read_csv('../input/test.csv', index_col='Id')

# 2. Pisahkan target dan pilih fitur
y = X_full.SalePrice
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF',
            'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = X_full[features].copy()
X_test = X_test_full[features].copy()

# 3. Bagi validation set
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, train_size=0.8, test_size=0.2, random_state=0)

# 4. Bandingkan beberapa model dengan MAE validation
def score_model(model, X_t=X_train, X_v=X_valid, y_t=y_train, y_v=y_valid):
    model.fit(X_t, y_t)
    return mean_absolute_error(y_v, model.predict(X_v))

models = [
    RandomForestRegressor(n_estimators=50, random_state=0),
    RandomForestRegressor(n_estimators=100, random_state=0),
    RandomForestRegressor(n_estimators=100, criterion='absolute_error', random_state=0),
    RandomForestRegressor(n_estimators=200, min_samples_split=20, random_state=0),
    RandomForestRegressor(n_estimators=100, max_depth=7, random_state=0),
]
for i, m in enumerate(models, start=1):
    print("Model %d MAE: %d" % (i, score_model(m)))

# 5. Latih model terpilih pada SELURUH data latih, lalu prediksi test
my_model = RandomForestRegressor(n_estimators=100, random_state=0)
my_model.fit(X, y)
preds_test = my_model.predict(X_test)

# 6. Simpan dalam format kompetisi
pd.DataFrame({'Id': X_test.index, 'SalePrice': preds_test}) \
  .to_csv('submission.csv', index=False)
```

### Kesalahan umum

- Menganggap MAE harus dimaksimalkan. MAE adalah error: model terbaik = MAE **terkecil**.
- Memilih model berdasarkan skor di data latih, bukan validation set — model dengan `max_depth` besar akan terlihat "sempurna" di train tapi buruk di test.
- Lupa bahwa setelah memilih hyperparameter, model final dilatih ulang pada `X` dan `y` penuh (train + validation), bukan hanya `X_train`.

### Drill

1. Berapa jumlah explanatory variable pada kompetisi yang dipakai seluruh exercise course ini?
2. Metrik apa yang dipakai untuk memilih model di Exercise 1, dan apakah nilainya harus besar atau kecil?
3. Sebutkan tiga konsep prasyarat yang harus sudah dikuasai sebelum mengambil course ini.
4. Di Exercise 1, `best_model` harus diisi salah satu dari `model_1`..`model_5`. Apa dasar pemilihannya, dan apa yang terjadi kalau diisi model dengan MAE bukan terendah?
5. File apa yang disubmit ke kompetisi, dan apa dua kolomnya?

---

## Missing Values

### Poin inti

- Hampir semua library ML (termasuk scikit-learn) melempar error kalau diberi data ber-`NaN`, jadi missing values harus ditangani sebelum `fit`.
- **Approach 1 — drop kolom**: paling sederhana, tapi membuang informasi. Kalau satu kolom penting hanya kosong 1 dari 10.000 baris, seluruh kolom tetap hilang.
- **Approach 2 — imputation**: isi nilai hilang dengan statistik kolom (default `SimpleImputer` = mean). Nilai imputasi tidak pernah persis benar, tapi biasanya mengalahkan drop kolom.
- **Approach 3 — imputation + indikator**: imputasi seperti biasa, lalu tambah kolom boolean `*_was_missing` yang menandai posisi nilai yang diimputasi. Berguna kalau nilai hilang itu sendiri informatif (baris dengan missing mungkin punya karakter unik).
- Hasil pada dataset Melbourne (random forest 10 tree, MAE validation): drop = **183.550**, imputasi = **178.166**, imputasi+indikator = **178.928** → imputasi menang, indikator sedikit merugikan di dataset ini.
- Alasan drop kalah: data latih 10.864 baris × 12 kolom, tiga kolom ber-missing (`Car` 49, `BuildingArea` 5.156, `YearBuilt` 4.307), dan **tidak ada kolom yang missing > 50%** — jadi drop membuang banyak sinyal.
- `SimpleImputer` menghapus nama kolom (mengembalikan numpy array), jadi nama kolom harus dipasang ulang secara manual.
- Saat bekerja dengan train + validation, daftar kolom yang di-drop harus **sama** di kedua DataFrame — tentukan dari `X_train`, terapkan ke keduanya.

### Contoh kode

```python
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=10, random_state=0)
    model.fit(X_train, y_train)
    return mean_absolute_error(y_valid, model.predict(X_valid))

# --- Investigasi: sebaran missing value
print(X_train.shape)
miss = X_train.isnull().sum()
print(miss[miss > 0])

# --- Approach 1: drop kolom yang punya missing
cols_with_missing = [c for c in X_train.columns if X_train[c].isnull().any()]
reduced_X_train = X_train.drop(cols_with_missing, axis=1)
reduced_X_valid = X_valid.drop(cols_with_missing, axis=1)

# --- Approach 2: imputasi mean (default)
imputer = SimpleImputer()                       # strategy='mean'
imputed_X_train = pd.DataFrame(imputer.fit_transform(X_train))   # fit HANYA di train
imputed_X_valid = pd.DataFrame(imputer.transform(X_valid))       # valid: transform saja
imputed_X_train.columns = X_train.columns       # nama kolom hilang, pasang ulang
imputed_X_valid.columns = X_valid.columns

# --- Approach 3: imputasi + kolom indikator
X_train_plus, X_valid_plus = X_train.copy(), X_valid.copy()
for col in cols_with_missing:
    X_train_plus[col + '_was_missing'] = X_train_plus[col].isnull()
    X_valid_plus[col + '_was_missing'] = X_valid_plus[col].isnull()

imputer = SimpleImputer()
imputed_X_train_plus = pd.DataFrame(imputer.fit_transform(X_train_plus))
imputed_X_valid_plus = pd.DataFrame(imputer.transform(X_valid_plus))
imputed_X_train_plus.columns = X_train_plus.columns
imputed_X_valid_plus.columns = X_valid_plus.columns
```

### Kesalahan umum

- Memanggil `fit_transform` pada validation set — statistik imputasi jadi "melihat" data validasi, itu **train-test contamination** (bentuk data leakage).
- Menentukan `cols_with_missing` dari gabungan train+valid, atau dari `X_valid`, sehingga kolom yang di-drop di train dan valid tidak konsisten → jumlah kolom tidak cocok saat `predict`.
- Lupa memasang kembali `columns` setelah `SimpleImputer`, lalu bingung karena `KeyError` saat memilih kolom atau saat masuk ke `ColumnTransformer` berbasis nama.

### Drill

1. Sebutkan tiga approach penanganan missing value di lesson ini, dan approach mana yang menghasilkan MAE terendah pada dataset Melbourne?
2. Mengapa drop kolom kalah jauh pada dataset Melbourne? Kaitkan dengan angka missing per kolom.
3. Tulis satu baris list comprehension untuk mendapatkan nama kolom yang punya minimal satu nilai hilang di `X_train`.
4. Mengapa `imputer.fit_transform(X_train)` boleh tapi `imputer.fit_transform(X_valid)` tidak?
5. Approach 3 (imputasi + indikator) menghasilkan MAE 178.928 vs 178.166 pada imputasi biasa. Apa artinya, dan kapan approach 3 tetap layak dicoba?

---

## Categorical Variables

### Poin inti

- Categorical variable = variabel yang hanya mengambil sejumlah nilai terbatas (contoh: frekuensi sarapan "Never"/"Rarely"/"Most days"/"Every day", atau merek mobil "Honda"/"Toyota"/"Ford").
- Tanpa preprocessing, variabel teks membuat model scikit-learn error. Di pandas, kolom teks terdeteksi lewat `dtype == 'object'`.
- **Approach 1 — drop**: hapus kolom kategorikal. Hanya masuk akal kalau kolomnya tidak informatif.
- **Approach 2 — ordinal encoding**: setiap nilai unik → integer. Mengasumsikan ada urutan; cocok untuk **ordinal variable** dan bekerja baik dengan model berbasis tree (decision tree, random forest).
- **Approach 3 — one-hot encoding**: satu kolom biner per kategori. Tidak mengasumsikan urutan; cocok untuk **nominal variable** ("Red" tidak lebih besar dari "Yellow").
- One-hot umumnya buruk untuk cardinality tinggi — rule of thumb di tutorial: jangan dipakai untuk variabel dengan lebih dari ~15 nilai unik.
- Hasil pada dataset Melbourne (random forest 100 tree): drop = **175.703**, ordinal = **165.936**, one-hot = **166.089** → drop paling buruk, ordinal dan one-hot praktis seri.
- Secara umum one-hot cenderung paling baik dan drop paling buruk, tapi hasilnya case-by-case — selalu ukur dengan validation set.
- Dua parameter penting `OneHotEncoder`: `handle_unknown='ignore'` (kategori baru di validation tidak bikin error) dan output dense (`sparse=False` di versi lama scikit-learn; `sparse_output=False` di versi baru).
- One-hot menghapus index dan nama kolom → index harus dipasang ulang dan kolom kategorikal asli harus di-drop sebelum `pd.concat`.

### Contoh kode

```python
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    return mean_absolute_error(y_valid, model.predict(X_valid))

# Daftar kolom kategorikal
object_cols = [c for c in X_train.columns if X_train[c].dtype == 'object']

# --- Approach 2: ordinal encoding
label_X_train, label_X_valid = X_train.copy(), X_valid.copy()
ordinal_encoder = OrdinalEncoder()
label_X_train[object_cols] = ordinal_encoder.fit_transform(X_train[object_cols])
label_X_valid[object_cols] = ordinal_encoder.transform(X_valid[object_cols])

# --- Approach 3: one-hot encoding (hanya kolom cardinality rendah)
low_cardinality_cols = [c for c in object_cols if X_train[c].nunique() < 10]

OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)  # sklearn >= 1.2
OH_cols_train = pd.DataFrame(
    OH_encoder.fit_transform(X_train[low_cardinality_cols]),
    index=X_train.index)                       # index dipasang ulang
OH_cols_valid = pd.DataFrame(
    OH_encoder.transform(X_valid[low_cardinality_cols]),
    index=X_valid.index)

# Buang kolom kategorikal asli, lalu gabungkan dengan kolom numerik
num_X_train = X_train.drop(object_cols, axis=1)
num_X_valid = X_valid.drop(object_cols, axis=1)
OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)
```

### Kesalahan umum

- `fit_transform` pada `X_valid` atau pada gabungan train+valid → encoder "belajar" kategori dari validation, hasil CV terlihat bagus tapi tidak generalize (train-test contamination).
- Tidak memakai `handle_unknown='ignore'`: begitu validation punya kategori yang tidak ada di train (mis. `Condition2`), `transform` langsung error. Solusi lain: drop kolom bermasalah, atau pakai encoder kustom.
- One-hot pada kolom cardinality tinggi (mis. `Neighborhood`) → ratusan kolom baru, data meledak, model lambat dan mudah overfit.

### Drill

1. Apa beda **ordinal** dan **nominal** variable, dan encoding mana yang diasumsikan cocok untuk masing-masing?
2. Kapan one-hot encoding sebaiknya **tidak** dipakai, dan apa alternatifnya?
3. Parameter `OneHotEncoder` apa yang mencegah error ketika validation set berisi kategori yang tidak ada di training set?
4. Pada dataset Melbourne, urutkan tiga approach dari MAE terendah ke tertinggi beserta angkanya.
5. Setelah `OneHotEncoder.fit_transform`, ada dua hal yang harus diperbaiki sebelum kolom hasil encoding digabung ke DataFrame numerik. Sebutkan keduanya.

---

## Pipelines

### Poin inti

- Pipeline membungkus langkah preprocessing dan modeling jadi satu objek yang bisa dipakai seperti satu estimator.
- Empat manfaat yang disebut tutorial: **cleaner code** (tidak perlu melacak train/valid di setiap langkah), **fewer bugs** (tidak ada langkah preprocessing yang terlewat atau salah urutan), **easier to productionize**, dan **more options for model validation** (dipakai di cross-validation lesson berikutnya).
- `ColumnTransformer` menggabungkan preprocessing yang berbeda per jenis kolom: imputer untuk numerik, imputer + one-hot untuk kategorikal.
- `Pipeline` menggabungkan `preprocessor` dan `model` dengan `steps=[(nama, objek), ...]`.
- Dengan pipeline, `fit(X_train, y_train)` menjalankan seluruh preprocessing **dan** training; `predict(X_valid)` menerima data **mentah** karena preprocessing tersimpan di dalam pipeline.
- Preprocessing yang di-`fit` di dalam pipeline hanya melihat data training → otomatis mencegah leakage.
- Contoh di tutorial: imputer `strategy='constant'` untuk numerik, `strategy='most_frequent'` + one-hot untuk kategorikal, model `RandomForestRegressor(n_estimators=100)`, MAE = **160.679**.
- Di exercise, baseline pipeline yang sama (dijalankan pada dataset Ames dengan seluruh kolom kategorikal + numerik) menghasilkan MAE sekitar **17.862**, dan tugasnya adalah menurunkan angka itu dengan preprocessing dan model versi sendiri.

### Contoh kode

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# 1. Preprocessing per jenis kolom
numerical_transformer = SimpleImputer(strategy='constant')      # isi 0

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore')),
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer, numerical_cols),
    ('cat', categorical_transformer, categorical_cols),
])

# 2. Model
model = RandomForestRegressor(n_estimators=100, random_state=0)

# 3. Gabungkan
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)])

my_pipeline.fit(X_train, y_train)                 # preprocessing + fit sekaligus
preds = my_pipeline.predict(X_valid)              # input mentah, aman
print('MAE:', mean_absolute_error(y_valid, preds))
```

### Kesalahan umum

- Melakukan imputasi/encoding di luar pipeline sebelum `train_test_split` atau sebelum CV — inilah penyebab paling umum skor validasi yang terlalu optimistis.
- Mengisi `transformers` di `ColumnTransformer` dengan daftar kolom yang salah atau tidak lengkap: kolom yang tidak tercakup akan dibuang (default `remainder='drop'`), sehingga fitur penting hilang tanpa error.
- Memberi `steps` berupa objek langsung, bukan tuple `(nama, objek)` — `Pipeline` mensyaratkan setiap langkah punya nama unik.

### Drill

1. Apa peran `ColumnTransformer` dan apa peran `Pipeline`? Jelaskan pembagian tugasnya.
2. Mengapa `my_pipeline.predict(X_valid)` boleh diberi `X_valid` mentah (masih ada `NaN` dan kolom teks)?
3. Sebutkan empat manfaat pipeline yang disebutkan tutorial.
4. Berapa MAE yang dihasilkan contoh pipeline di tutorial (random forest 100 tree)?
5. Kalau ada kolom numerik yang tidak dimasukkan ke `numerical_cols` maupun `categorical_cols`, apa yang terjadi dan mengapa berbahaya?

---

## Cross-Validation

### Poin inti

- Single validation set (holdout) punya kelemahan: skor bergantung keberuntungan pembagian baris. Validation set kecil → noisy; validation set besar → data latih berkurang.
- Cross-validation: data dibagi menjadi `k` fold (contoh 5 fold @ 20%). Setiap fold bergiliran jadi holdout, sisanya jadi training. Hasilnya `k` skor.
- Karena setiap fold pernah jadi holdout, 100% baris dipakai sebagai data evaluasi pada suatu titik — estimasi kualitas jauh lebih stabil.
- Biaya: model dilatih `k` kali, jadi waktu komputasi naik ~`k` kali.
- Aturan praktis dari tutorial: dataset kecil → pakai CV; dataset besar → satu validation set cukup. Kalau `fit` selesai dalam beberapa menit atau kurang, CV biasanya layak.
- Cara lain memutuskan: jalankan CV dan lihat apakah skor antar fold mirip. Kalau iya, single validation set sudah memadai.
- scikit-learn memakai konvensi "semakin besar semakin baik", jadi MAE dilaporkan negatif: `scoring='neg_mean_absolute_error'` dan kalikan `-1`.
- Contoh di tutorial: skor per fold `[301.629, 303.164, 287.298, 236.062, 260.383]`, rata-rata **277.707**.
- CV **wajib** dibungkus pipeline: tanpa pipeline, preprocessing yang di-fit di luar akan membocorkan informasi validation ke training di setiap fold.

### Contoh kode

```python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

my_pipeline = Pipeline(steps=[
    ('preprocessor', SimpleImputer()),                       # di dalam pipeline = aman
    ('model', RandomForestRegressor(n_estimators=50, random_state=0)),
])

# sklearn melaporkan MAE sebagai nilai negatif -> kalikan -1
scores = -1 * cross_val_score(my_pipeline, X, y,
                              cv=5,
                              scoring='neg_mean_absolute_error')
print("MAE per fold:", scores)
print("Rata-rata MAE:", scores.mean())

# Pola umum untuk tuning hyperparameter: bungkus dalam fungsi
def get_score(n_estimators):
    pipe = Pipeline(steps=[
        ('preprocessor', SimpleImputer()),
        ('model', RandomForestRegressor(n_estimators=n_estimators, random_state=0)),
    ])
    s = -1 * cross_val_score(pipe, X, y, cv=3,
                             scoring='neg_mean_absolute_error')
    return s.mean()

results = {n: get_score(n) for n in range(50, 401, 50)}
```

### Kesalahan umum

- Lupa mengalikan `-1` sehingga tanda skor terbalik dan model "terbaik" justru yang MAE-nya paling besar secara absolut.
- Melakukan imputasi/encoding/feature selection pada seluruh dataset sebelum `cross_val_score` — preprocessing harus ada **di dalam** pipeline agar tiap fold hanya belajar dari fold training-nya.
- Memakai CV mahal (mis. `cv=10`) untuk model yang lambat, atau sebaliknya memakai satu split kecil untuk dataset kecil sehingga kesimpulan tuning sekadar noise.

### Drill

1. Apa keunggulan utama CV dibanding satu validation set, dan apa biayanya?
2. Untuk kondisi apa tutorial menyarankan tetap memakai satu validation set saja?
3. Mengapa `scoring='neg_mean_absolute_error'` menghasilkan angka negatif, dan bagaimana mengubahnya jadi MAE biasa?
4. Berapa rata-rata MAE dari lima skor CV pada contoh tutorial?
5. Mengapa cross-validation tanpa pipeline berisiko menghasilkan skor yang terlalu bagus?

---

## XGBoost

### Poin inti

- Random forest = ensemble (merata-ratakan banyak tree). Gradient boosting = ensemble lain yang menambah model **secara berurutan**.
- Siklus gradient boosting: prediksi dengan ensemble saat ini → hitung loss → fit model baru yang mengurangi loss (gradient descent pada fungsi loss) → tambahkan ke ensemble → ulangi.
- Model pertama boleh sangat kasar; model berikutnya memperbaiki error sisa.
- `XGBRegressor` adalah implementasi gradient boosting dengan API gaya scikit-learn (`fit`, `predict`), plus banyak parameter dan optimasi kecepatan.
- `n_estimators` = jumlah model/tahap boosting. Terlalu kecil → underfitting; terlalu besar → overfitting. Nilai tipikal 100–1000, tergantung `learning_rate`.
- `early_stopping_rounds` menghentikan iterasi otomatis saat skor validasi berhenti membaik; butuh `eval_set` sebagai sumber skor validasi. Set `n_estimators` tinggi lalu biarkan early stopping yang menentukan.
- `learning_rate` mengalikan kontribusi tiap tree. Default XGBoost 0.1. Learning rate kecil + `n_estimators` besar → biasanya lebih akurat, tapi latih lebih lama.
- `n_jobs` = jumlah core paralel; hanya terasa pada dataset besar, tidak memperbaiki kualitas model.
- Setelah menemukan jumlah iterasi optimal lewat early stopping, latih ulang model final pada seluruh data dengan `n_estimators` = jumlah iterasi optimal itu (tanpa early stopping).
- Baseline default `XGBRegressor()` di tutorial menghasilkan MAE **239.435**.

### Contoh kode

```python
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

# Baseline: semua parameter default
my_model = XGBRegressor()
my_model.fit(X_train, y_train)
preds = my_model.predict(X_valid)
print("MAE baseline:", mean_absolute_error(preds, y_valid))

# Versi tuned: banyak tree + learning rate kecil + early stopping
my_model = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    n_jobs=4,
    random_state=0,
    early_stopping_rounds=5,          # xgboost >= 2.0: parameter di constructor
)
my_model.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],     # wajib untuk early stopping
    verbose=False,
)
preds = my_model.predict(X_valid)
print("MAE tuned:", mean_absolute_error(preds, y_valid))
```

### Kesalahan umum

- Memakai `early_stopping_rounds` tanpa `eval_set` → error, karena tidak ada data untuk menghitung skor validasi.
- Memakai validation set yang sama untuk early stopping **dan** untuk melaporkan skor akhir → skor jadi optimistis (model tidak pernah "buta" terhadap data itu).
- Menaikkan `n_estimators` tanpa menurunkan `learning_rate`: kombinasi learning rate besar + banyak tree adalah resep overfitting cepat.

### Drill

1. Apa perbedaan mendasar cara random forest dan gradient boosting membangun ensemble?
2. Apa efek `n_estimators` yang terlalu besar, dan apa efek yang terlalu kecil?
3. Apa fungsi `early_stopping_rounds=5`, dan parameter apa yang wajib disertakan agar bisa dipakai?
4. Berapa default `learning_rate` XGBoost, dan apa trade-off mengecilkannya?
5. Setelah early stopping menemukan jumlah iterasi optimal, bagaimana cara melatih model final pada seluruh data?

---

## Data Leakage

### Poin inti

- **Data leakage**: data training mengandung informasi tentang target yang **tidak akan tersedia** saat model dipakai untuk prediksi. Akibatnya skor train/validasi bagus, tapi performa produksi jeblok.
- Leakage membuat model tampak akurat sampai keputusan mulai diambil dengan model itu.
- **Target leakage**: predictor berisi data yang baru ada/berubah setelah target diketahui. Contoh tutorial: memprediksi `got_pneumonia` dengan fitur `took_antibiotic_medicine` — antibiotik diminum **setelah** pneumonia, jadi fitur itu "membocorkan" jawaban.
- Cara berpikir yang benar: urutan waktu ketersediaan data, bukan sekadar "apakah fitur ini prediktif". Aturan pencegahan: keluarkan variabel yang diperbarui atau dibuat setelah target terealisasi.
- **Train-test contamination**: validasi tercemar karena preprocessing di-`fit` pada data yang mencakup validation/test. Contoh: menjalankan imputer sebelum `train_test_split`.
- Pencegahan: split dulu, lalu semua fitting (imputer, encoder, feature selection) hanya pada data training — paling mudah dengan pipeline. Saat CV, aturan ini lebih kritis lagi.
- Contoh credit card (1.319 baris): akurasi CV **0.979537** dengan fitur bocor, **0.832440** setelah `expenditure`, `share`, `active`, `majorcards` dibuang.
- `expenditure` bocor karena berarti pengeluaran **pada kartu yang dilamar**: 100% yang tidak dapat kartu punya expenditure 0, sedangkan yang dapat kartu hanya 2% yang 0.
- `share` harus ikut dibuang karena sebagian ditentukan oleh `expenditure`; `active` dan `majorcards` ambigu — kalau tidak bisa memastikan asal datanya, lebih aman dibuang.
- Akurasi yang lebih rendah (0.83) justru lebih dipercaya karena mencerminkan performa nyata pada aplikasi baru.

### Contoh kode

```python
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

data = pd.read_csv('../input/AER_credit_card_data.csv',
                   true_values=['yes'], false_values=['no'])
y = data.card
X = data.drop(['card'], axis=1)

# Pipeline dipakai walau tanpa preprocessing — best practice
my_pipeline = make_pipeline(RandomForestClassifier(n_estimators=100))

print("CV accuracy (semua fitur):",
      cross_val_score(my_pipeline, X, y, cv=5, scoring='accuracy').mean())

# Cek manual: apakah fitur berkorelasi "terlalu sempurna" dengan target?
print('Yang tidak dapat kartu & expenditure 0: %.2f'
      % (X.expenditure[~y] == 0).mean())
print('Yang dapat kartu & expenditure 0: %.2f'
      % (X.expenditure[y] == 0).mean())

# Buang fitur yang berpotensi bocor, evaluasi ulang
potential_leaks = ['expenditure', 'share', 'active', 'majorcards']
X2 = X.drop(potential_leaks, axis=1)
print("CV accuracy (fitur bocor dibuang):",
      cross_val_score(my_pipeline, X2, y, cv=5, scoring='accuracy').mean())
```

### Kesalahan umum

- Menjalankan `SimpleImputer`, `OrdinalEncoder`, atau seleksi fitur pada seluruh dataset sebelum `train_test_split` / `cross_val_score` — bentuk train-test contamination yang paling sering terjadi.
- Menyimpan fitur turunan dari fitur bocor. `share` dihitung dari `expenditure`, jadi membuang `expenditure` saja tidak cukup.
- Menganggap skor validasi/CV yang sangat tinggi (mis. akurasi 98%) sebagai keberhasilan tanpa memeriksa apakah ada fitur yang baru tersedia setelah target diketahui.

### Drill

1. Definisikan data leakage dalam satu kalimat, dan sebutkan dua tipe utamanya.
2. Mengapa `took_antibiotic_medicine` merupakan target leakage saat memprediksi `got_pneumonia`?
3. Pada contoh credit card, akurasi turun dari 0.979537 ke 0.832440 setelah fitur bocor dibuang. Mengapa model dengan akurasi lebih rendah justru lebih dapat dipercaya?
4. Apa itu train-test contamination, dan berikan satu contoh konkret yang melibatkan imputer?
5. Kamu membangun model harga rumah yang dipakai untuk menilai rumah baru saat iklan dipasang. Dari empat fitur ini — (1) luas rumah, (2) rata-rata harga jual rumah di neighborhood yang sama, (3) latitude & longitude, (4) ada/tidaknya basement — mana yang paling mungkin bocor, dan mengapa?

---

## Kunci Jawaban

### Introduction

1. 79 explanatory variable, target `SalePrice` (Housing Prices Competition for Kaggle Learn Users).
2. MAE (mean absolute error); semakin **kecil** semakin baik.
3. Model validation, underfitting & overfitting, random forest (plus pengalaman membangun model ML).
4. Dasar pemilihan: MAE validation terendah di antara lima model. Kalau diisi model lain, `step_1.check()` menandai jawaban salah karena `best_model` harus menunjuk model dengan MAE minimum.
5. `submission.csv`, kolom `Id` dan `SalePrice`.

### Missing Values

1. (a) drop kolom ber-missing, (b) imputasi, (c) imputasi + kolom indikator. MAE terendah: imputasi (178.166).
2. Karena tidak ada kolom yang missing lebih dari setengah: dari 10.864 baris, `Car` 49, `BuildingArea` 5.156, `YearBuilt` 4.307. Drop membuang tiga kolom yang sebagian besar masih terisi.
3. `cols_with_missing = [c for c in X_train.columns if X_train[c].isnull().any()]`
4. Karena `fit_transform` pada validation akan menghitung statistik (mean) dari data validation itu sendiri — informasi validation masuk ke preprocessing (train-test contamination). Validation hanya boleh `transform` memakai statistik dari train.
5. Artinya indikator `_was_missing` tidak menambah informasi berguna di dataset ini dan malah menambah noise/dimensi. Approach 3 tetap layak dicoba kalau ada alasan kuat bahwa "nilai ini hilang" berkorelasi dengan target (missing not at random).

### Categorical Variables

1. Ordinal = kategorinya punya urutan intrinsik (Never < Rarely < Most days < Every day) → ordinal encoding. Nominal = tanpa urutan (Red/Yellow/Green) → one-hot encoding.
2. Tidak dipakai kalau cardinality tinggi (rule of thumb: > ~15 nilai unik) karena jumlah kolom meledak. Alternatif: ordinal encoding atau drop kolom tersebut.
3. `handle_unknown='ignore'` (plus output dense lewat `sparse_output=False` bila hasilnya mau dimasukkan ke DataFrame).
4. Ordinal 165.936 < one-hot 166.089 < drop 175.703.
5. (a) memasang kembali index asli (`OH_cols.index = X_train.index`), (b) membuang kolom kategorikal asli dari DataFrame numerik sebelum `pd.concat`.

### Pipelines

1. `ColumnTransformer` menangani preprocessing berbeda per kelompok kolom (numerik vs kategorikal) dan menggabungkan hasilnya; `Pipeline` merangkai preprocessor + model menjadi satu estimator dengan urutan langkah yang tetap.
2. Karena pipeline menyimpan objek preprocessing yang sudah di-`fit`; saat `predict`, ia otomatis menjalankan imputasi dan encoding yang sama seperti saat training.
3. Cleaner code, fewer bugs, easier to productionize, more options for model validation.
4. 160.679 (MAE validation).
5. Kolom itu dibuang oleh `ColumnTransformer` (`remainder='drop'`), jadi fitur hilang diam-diam tanpa error. Berbahaya karena model tetap jalan dan skornya terlihat wajar, padahal kehilangan sinyal — dan tidak ada pesan kesalahan yang menandainya.

### Cross-Validation

1. Keunggulan: estimasi kualitas lebih stabil/akurat karena setiap baris pernah jadi holdout (100% data terpakai sebagai evaluasi). Biaya: model dilatih `k` kali sehingga waktu komputasi naik.
2. Dataset besar dan/atau model yang lambat dilatih (mis. butuh lebih dari beberapa menit per fit).
3. Karena scikit-learn memakai konvensi "higher is better" untuk semua scorer, MAE dilaporkan sebagai negatif. Kalikan hasilnya dengan `-1`.
4. 277.707 (rata-rata dari 301.629, 303.164, 287.298, 236.062, 260.383).
5. Karena preprocessing (imputer/encoder) akan di-`fit` pada data yang mencakup fold validasi, sehingga statistik validation bocor ke training di setiap fold — skor jadi terlalu optimistis (train-test contamination).

### XGBoost

1. Random forest membangun banyak tree **independen** lalu merata-ratakan prediksinya; gradient boosting menambah model **berurutan**, setiap model baru dilatih untuk mengurangi loss sisa ensemble sebelumnya.
2. Terlalu besar → overfitting (akurat di train, buruk di data baru); terlalu kecil → underfitting (buruk di keduanya).
3. Menghentikan iterasi setelah 5 round berturut-turut skor validasi tidak membaik; wajib menyertakan `eval_set=[(X_valid, y_valid)]`.
4. Default 0.1. Learning rate lebih kecil → tiap tree berkontribusi lebih sedikit, perlu lebih banyak tree, biasanya lebih akurat tapi latih lebih lama.
5. Latih ulang pada seluruh data (`X`, `y`) dengan `n_estimators` = jumlah iterasi optimal hasil early stopping, tanpa `early_stopping_rounds` dan tanpa `eval_set`.

### Data Leakage

1. Kondisi ketika data training mengandung informasi tentang target yang tidak tersedia saat prediksi sebenarnya; tipe: target leakage dan train-test contamination.
2. Karena antibiotik baru dikonsumsi **setelah** diagnosis pneumonia ditegakkan — saat prediksi dilakukan, informasi itu belum ada. Model memakai konsekuensi target sebagai predictor.
3. Karena 0.83 diukur tanpa fitur yang tidak akan tersedia saat aplikasi nyata diproses, sehingga lebih mencerminkan performa produksi; 0.98 berasal dari fitur `expenditure`/`share` yang baru terisi setelah kartu disetujui.
4. Validasi tercemar karena data validation ikut memengaruhi proses fitting preprocessing. Contoh: `SimpleImputer().fit_transform(X)` pada seluruh dataset sebelum `train_test_split`, sehingga mean imputasi dihitung termasuk baris validation.
5. Fitur (2) — rata-rata harga jual rumah di neighborhood yang sama. Harga jual baru terealisasi setelah rumah terjual, sedangkan prediksi dibuat saat iklan dipasang; fitur ini membawa informasi masa depan (dan turunannya dari target). Fitur (1), (3), (4) tersedia saat iklan dibuat.

---

## Ringkasan 1 paragraf

Intermediate Machine Learning adalah course 4 jam yang mengubah model "bisa jalan" menjadi model yang layak dipakai: ia mulai dari dua masalah data paling umum (missing values dan categorical variables) dengan tiga pendekatan pembanding untuk masing-masing — drop, imputasi/ordinal encoding, dan imputasi+indikator/one-hot encoding — lalu menunjukkan bahwa pilihan terbaik harus diukur, bukan ditebak, karena hasilnya bergantung dataset (imputasi menang di Melbourne, ordinal dan one-hot hampir seri). Semua kerumitan itu kemudian dirapikan dengan `Pipeline` + `ColumnTransformer` yang menyatukan preprocessing dan model, sekaligus menjadi prasyarat teknis untuk `cross_val_score` yang benar: preprocessing harus selalu di-`fit` di dalam fold training, jika tidak skor validasi akan menipu. XGBoost masuk sebagai model tabular terkuat dengan tiga tuas utama (`n_estimators`, `early_stopping_rounds` + `eval_set`, `learning_rate`) dan satu disiplin tambahan: setelah early stopping menemukan jumlah iterasi optimal, model final dilatih ulang pada seluruh data. Lesson terakhir — data leakage — merangkum logika yang menyatukan semuanya: target leakage (fitur yang baru ada setelah target terealisasi, seperti `took_antibiotic_medicine` atau `expenditure` pada kartu kredit) dan train-test contamination (preprocessing yang melihat data validasi) sama-sama menghasilkan skor indah yang runtuh di produksi, sehingga skor CV 0.98 yang turun menjadi 0.83 setelah fitur bocor dibuang justru adalah perbaikan, bukan kemunduran.

## Sumber

**Halaman course**
- Kaggle Learn — Intermediate Machine Learning: https://www.kaggle.com/learn/intermediate-machine-learning

**Tutorial resmi (7 lesson)**
1. Introduction — https://www.kaggle.com/code/alexisbcook/introduction
2. Missing Values — https://www.kaggle.com/code/alexisbcook/missing-values
3. Categorical Variables — https://www.kaggle.com/code/alexisbcook/categorical-variables
4. Pipelines — https://www.kaggle.com/code/alexisbcook/pipelines
5. Cross-Validation — https://www.kaggle.com/code/alexisbcook/cross-validation
6. XGBoost — https://www.kaggle.com/code/alexisbcook/xgboost
7. Data Leakage — https://www.kaggle.com/code/alexisbcook/data-leakage

**Exercise resmi (7 notebook)**
- Exercise: Introduction — https://www.kaggle.com/code/alexisbcook/exercise-introduction
- Exercise: Missing Values — https://www.kaggle.com/code/alexisbcook/exercise-missing-values
- Exercise: Categorical Variables — https://www.kaggle.com/code/alexisbcook/exercise-categorical-variables
- Exercise: Pipelines — https://www.kaggle.com/code/alexisbcook/exercise-pipelines
- Exercise: Cross-Validation — https://www.kaggle.com/code/alexisbcook/exercise-cross-validation
- Exercise: XGBoost — https://www.kaggle.com/code/alexisbcook/exercise-xgboost
- Exercise: Data Leakage — https://www.kaggle.com/code/alexisbcook/exercise-data-leakage

**Dataset & kompetisi**
- Housing Prices Competition for Kaggle Learn Users — https://www.kaggle.com/c/home-data-for-ml-course
- Melbourne Housing Snapshot (dipakai di tutorial) — `../input/melbourne-housing-snapshot/melb_data.csv`
- AER Credit Card Data (dipakai di lesson Data Leakage) — `../input/AER_credit_card_data.csv`

**Dokumentasi teknis pendamping**
- scikit-learn `SimpleImputer` — https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html
- scikit-learn `OneHotEncoder` — https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
- scikit-learn `ColumnTransformer` — https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html
- scikit-learn `cross_val_score` — https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html
- XGBoost `XGBRegressor` (Python API) — https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBRegressor

**Catatan akses & verifikasi**
- Isi tutorial dan exercise diambil dari notebook publik Kaggle melalui endpoint resmi `https://www.kaggle.com/api/v1/kernels/pull?userName=alexisbcook&kernelSlug=<slug>` (respons 200, teks lengkap), bukan dari ringkasan pihak ketiga. Halaman `kaggle.com/code/...` sendiri dirender dengan JavaScript sehingga ekstraksi HTML biasa hanya mengembalikan judul.
- Dua penyesuaian versi library yang perlu diketahui saat menjalankan kode tutorial di lingkungan terbaru:
  - `OneHotEncoder(sparse=False)` pada tutorial sudah diganti `sparse_output=False` (scikit-learn ≥ 1.2; `sparse` dihapus pada rilis berikutnya).
  - `early_stopping_rounds` pada tutorial diletakkan di `fit()`; sejak XGBoost 2.0 parameter itu dipindah ke constructor `XGBRegressor(...)`, dan dihapus dari `fit()` pada 2.1.
- Seluruh angka MAE/akurasi yang dikutip adalah output yang tercetak di notebook tutorial resmi, bukan hasil pengukuran ulang di mesin sendiri.
- Status pengerjaan: **exercise belum dikerjakan** dan tidak ada submission yang dibuat. Dokumen ini murni catatan belajar.
