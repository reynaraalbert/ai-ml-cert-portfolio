# Kaggle Learn — Feature Engineering

Catatan belajar untuk course **Feature Engineering** (`kaggle.com/learn/feature-engineering`), instructor Ryan Holbrook. Estimasi 5 jam, level Intermediate ML, 6 lesson + 1 bonus project.

Isi course (sesuai halaman resmi):

| # | Lesson | Fokus |
|---|--------|-------|
| 1 | What Is Feature Engineering | Prinsip & tujuan |
| 2 | Mutual Information | Feature utility metric |
| 3 | Creating Features | Transformasi dengan Pandas |
| 4 | Clustering With K-Means | Unsupervised: label cluster |
| 5 | Principal Component Analysis | Unsupervised: komponen variasi |
| 6 | Target Encoding | Encoding supervised untuk kategorikal |
| Bonus | Feature Engineering for House Prices | Project gabungan (House Prices) |

Dataset yang dipakai course: `concrete.csv`, `autos.csv` (1985 Automobiles), `accidents.csv` (US Traffic Accidents), `customer.csv` (Customer Lifetime Value), `ames.csv`, `housing.csv` (California Housing), `movielens1m.csv`.

---

## What Is Feature Engineering

### Poin inti

- Tujuan feature engineering: membuat data **lebih cocok untuk masalah yang dihadapi** — bukan sekadar menambah kolom.
- Tiga alasan praktis melakukannya: memperbaiki performa prediksi, mengurangi kebutuhan komputasi/data, dan meningkatkan interpretabilitas.
- Prinsip pemandu: sebuah fitur berguna hanya jika **hubungannya dengan target bisa dipelajari model**. Model linear hanya bisa belajar hubungan linear; tugasmu adalah mentransformasi fitur sehingga hubungannya menjadi linear.
- Konsekuensi prinsip itu: **transformasi yang kamu terapkan pada dasarnya menjadi bagian dari model**. Contoh kanonik: prediksi `Price` tanah dari `Length`; model linear gagal karena hubungannya kuadratik, tapi jika `Length**2` dibuat sebagai fitur `Area`, model linear bisa mencocokkan parabola.
- Baseline dulu, baru eksperimen. Skor baseline (cross-validated) adalah pembanding untuk memutuskan fitur baru layak disimpan atau dibuang.
- Contoh *Concrete*: menambahkan tiga fitur rasio (`FCRatio`, `AggCmtRatio`, `WtrCmtRatio`) menurunkan MAE random forest dibanding baseline pada fitur mentah — bukti rasio mengekspos informasi yang tidak ditangkap model.
- Feature engineering bersifat iteratif: ukur → bandingkan dengan baseline → simpan/buang → ulangi.
- Model berbeda butuh transformasi berbeda (linear vs tree-based); tidak ada resep universal.

### Contoh kode

```python
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

df = pd.read_csv("concrete.csv")

# 1) Baseline: ukur dulu sebelum menambah fitur apa pun
X, y = df.copy(), df["CompressiveStrength"]
X = X.drop(columns="CompressiveStrength")

baseline = RandomForestRegressor(criterion="mae", random_state=0)
base_score = -cross_val_score(
    baseline, X, y, cv=5, scoring="neg_mean_absolute_error"
).mean()

# 2) Tambahkan fitur rasio (bukan jumlah absolut) — hipotesis dari domain
X2 = X.copy()
X2["FCRatio"] = X2["FineAggregate"] / X2["CoarseAggregate"]
X2["AggCmtRatio"] = (X2["CoarseAggregate"] + X2["FineAggregate"]) / X2["Cement"]
X2["WtrCmtRatio"] = X2["Water"] / X2["Cement"]

model = RandomForestRegressor(criterion="mae", random_state=0)
new_score = -cross_val_score(
    model, X2, y, cv=5, scoring="neg_mean_absolute_error"
).mean()

print(f"MAE baseline: {base_score:.4}")   # makin kecil makin baik
print(f"MAE + rasio : {new_score:.4}")
```

```python
# Ilustrasi prinsip: fitur kuadrat membuat hubungan linear bagi model linear
df["Area"] = df["Length"] ** 2   # model linear kini bisa fit parabola via Area
```

### Kesalahan umum

- Langsung menambah puluhan fitur tanpa baseline dan tanpa cross-validation, sehingga tidak bisa membuktikan apakah ada perbaikan.
- Menguji skor pada data training yang sama (in-sample) sehingga kenaikan skor hanya overfitting, bukan perbaikan nyata.
- Menerapkan transformasi yang sama ke semua model — misalnya menormalisasi untuk random forest (dampak kecil) sambil tetap memberi model linear fitur yang hubungannya non-linear.

### Drill

1. Sebutkan tiga alasan melakukan feature engineering menurut lesson ini.
2. Mengapa `Length**2` (menjadi `Area`) membuat model linear lebih baik dalam memprediksi harga tanah?
3. Apa fungsi skor baseline dalam alur kerja feature engineering?
4. Pada contoh *Concrete*, jenis fitur apa yang ditambahkan dan mengapa rasio lebih masuk akal daripada nilai absolut?
5. Model seperti apa yang paling diuntungkan oleh fitur yang sudah dinormalisasi, dan mana yang biasanya tidak terlalu butuh?

---

## Mutual Information

### Poin inti

- Langkah pertama menghadapi dataset baru: bangun **ranking fitur** dengan *feature utility metric*, lalu kembangkan subset kecil dengan potensi terbesar.
- Mutual information (MI) mengukur hubungan lewat **uncertainty**: seberapa besar pengetahuan tentang satu variabel mengurangi ketidakpastian variabel lain.
- Keunggulan MI vs korelasi: MI mendeteksi **segala jenis hubungan**; korelasi hanya hubungan linear.
- Skala MI: minimum 0.0 (independen). Secara teori tanpa batas atas, tetapi dalam praktik nilai di atas ~2.0 jarang (MI bersifat logaritmik, naik sangat lambat).
- MI adalah metrik **univariate**: ia tidak mendeteksi interaksi antar fitur. Fitur bisa sangat informatif saat berinteraksi tapi skor MI-nya rendah.
- MI menunjukkan *relative potential*, bukan jaminan model bisa memanfaatkannya — kegunaan nyata bergantung pada model dan transformasi yang dipakai.
- scikit-learn menyediakan `mutual_info_regression` (target kontinu) dan `mutual_info_classif` (target kategorikal) di `sklearn.feature_selection`, dengan estimasi non-parametrik berbasis entropi dari jarak k-NN; karena fitur diskret diperlakukan berbeda, tandai lewat `discrete_features` (kategorikal → `factorize` dulu agar menjadi integer).
- Visualisasi adalah pelengkap ranking: contoh `fuel_type` skor MI rendah tetapi memisahkan dua populasi `price` pada `horsepower` — indikasi interaction effect.

### Contoh kode

```python
import pandas as pd
from sklearn.feature_selection import mutual_info_regression

df = pd.read_csv("autos.csv")
X, y = df.copy(), df.pop("price")

# Kategorikal -> integer (label encoding) agar bisa ditandai diskret
for col in X.select_dtypes("object"):
    X[col], _ = X[col].factorize()

# Wajib: beri tahu sklearn mana fitur diskret (integer) vs kontinu
discrete_features = X.dtypes == int

def make_mi_scores(X, y, discrete_features):
    mi = mutual_info_regression(X, y, discrete_features=discrete_features,
                                random_state=0)
    return (pd.Series(mi, name="MI Scores", index=X.columns)
              .sort_values(ascending=False))

mi_scores = make_mi_scores(X, y, discrete_features)
print(mi_scores.head(10))
```

```python
import seaborn as sns
# Cek interaksi: fitur ber-MI rendah bisa penting dalam kombinasi
sns.lmplot(x="horsepower", y="price", hue="fuel_type", data=df)
```

### Kesalahan umum

- Lupa mengatur `discrete_features` sehingga semua fitur dianggap kontinu — skor menjadi tidak akurat (sklearn sendiri memperingatkan bahwa salah menandai diskret/kontinu biasanya memberi hasil keliru).
- Membuang fitur ber-MI rendah tanpa memeriksa interaksi atau domain knowledge; fitur itu bisa jadi penting hanya dalam kombinasi.
- Membandingkan nilai MI antar dataset/target berbeda seolah skala absolut, padahal MI hanya bermakna relatif di dalam satu ranking.

### Drill

1. Apa perbedaan utama MI dan korelasi dalam mendeteksi bentuk hubungan?
2. Berapa nilai MI minimum, dan apa artinya? Mengapa nilai > 2.0 disebut jarang?
3. Mengapa MI disebut metrik *univariate*, dan apa konsekuensi praktisnya?
4. Dua fungsi scikit-learn apa yang dipakai untuk MI, dan kapan memakai masing-masing?
5. Pada dataset *Automobiles*, fitur apa yang skor MI-nya rendah tetapi tetap layak dipertahankan, dan apa alasannya?

---

## Creating Features

### Poin inti

- Lesson ini fokus pada transformasi yang bisa dilakukan **sepenuhnya di Pandas**: mathematical transforms, counts, build-up/break-down, dan group transforms.
- **Mathematical transforms**: operasi aritmetika antar kolom. Contoh domain: `stroke_ratio = stroke / bore`; displacement = `π * (0.5*bore)**2 * stroke * num_of_cylinders`. Semakin rumit kombinasinya, semakin sulit model mempelajarinya — jadi buat eksplisit.
- Distribusi miring sering dinormalkan dengan logaritma; gunakan `np.log1p` bila ada nilai 0.
- **Counts**: fitur biner/boolean tentang ada-tidaknya sesuatu (mis. fasilitas jalan di sekitar kecelakaan) dijumlahkan per baris dengan `.sum(axis=1)`; boolean bisa dijumlahkan seperti integer.
- **Build-up / break-down**: memecah string terstruktur (ID, telepon, alamat, kode produk, tanggal) dengan `.str.split(..., expand=True)`, atau menggabungkan fitur sederhana menjadi fitur komposit (`make + "_" + body_style`).
- **Group transforms**: agregasi lintas baris per kategori (`groupby(...)[col].transform("mean")`) — mis. rata-rata income per state, atau frequency encoding lewat `transform("count")`.
- Saat ada split train/validasi: buat group transform **hanya dari training set**, lalu `merge` (setelah `drop_duplicates`) ke validasi agar independensi terjaga.
- Panduan per model: model linear belajar jumlah/selisih tapi tidak lebih; rasio sulit dipelajari hampir semua model sehingga cepat memberi gain; model linear & neural net butuh normalisasi (NN sangat butuh skala dekat 0); tree model bisa mengaproksimasi kombinasi apa pun tetapi tetap terbantu bila kombinasi penting dibuat eksplisit, terutama saat data terbatas; counts sangat membantu tree model karena tree tidak punya cara alami mengagregasi banyak fitur sekaligus.

### Contoh kode

```python
import numpy as np
import pandas as pd

autos = pd.read_csv("autos.csv")
accidents = pd.read_csv("accidents.csv")
concrete = pd.read_csv("concrete.csv")
customer = pd.read_csv("customer.csv")

# 1) Mathematical transform sederhana & kompleks
autos["stroke_ratio"] = autos.stroke / autos.bore
autos["displacement"] = (
    np.pi * ((0.5 * autos.bore) ** 2) * autos.stroke * autos.num_of_cylinders
)

# 2) Normalisasi distribusi miring (log1p aman untuk nilai 0)
accidents["LogWindSpeed"] = accidents.WindSpeed.apply(np.log1p)

# 3) Counts: jumlah fasilitas jalan di sekitar kecelakaan
roadway_features = ["Amenity", "Bump", "Crossing", "GiveWay", "Junction",
                    "NoExit", "Railway", "Roundabout", "Station", "Stop",
                    "TrafficCalming", "TrafficSignal"]
accidents["RoadwayFeatures"] = accidents[roadway_features].sum(axis=1)

# Counts dari nilai > 0 (bukan boolean)
components = ["Cement", "BlastFurnaceSlag", "FlyAsh", "Water",
              "Superplasticizer", "CoarseAggregate", "FineAggregate"]
concrete["Components"] = concrete[components].gt(0).sum(axis=1)

# 4) Break-down string terstruktur
customer[["Type", "Level"]] = customer["Policy"].str.split(" ", expand=True)

# 5) Build-up fitur komposit
autos["make_and_style"] = autos["make"] + "_" + autos["body_style"]
```

```python
# 6) Group transform tanpa leakage: hitung di train, merge ke valid
df_train = customer.sample(frac=0.5, random_state=0)
df_valid = customer.drop(df_train.index)

df_train["AverageClaim"] = df_train.groupby("Coverage")["ClaimAmount"].transform("mean")

df_valid = df_valid.merge(
    df_train[["Coverage", "AverageClaim"]].drop_duplicates(),
    on="Coverage", how="left",
)
```

### Kesalahan umum

- Memakai `np.log` pada fitur yang mengandung 0 sehingga muncul `-inf`; gunakan `np.log1p` (log(1+x)).
- Menghitung group transform pada gabungan train+validasi lalu memakainya di keduanya — informasi validasi bocor ke fitur training.
- Mengasumsikan semua model butuh normalisasi; untuk tree-based biasanya dampaknya kecil, sementara untuk neural net hampir wajib.

### Drill

1. Sebutkan empat keluarga transformasi yang dibahas di lesson ini.
2. Kapan `np.log1p` dipilih daripada `np.log`, dan mengapa?
3. Bagaimana cara membuat fitur count dari sekumpulan kolom yang nilainya 0/positif?
4. Tuliskan langkah aman membuat group transform `mean` saat data dibagi train/validasi.
5. Menurut panduan lesson, mengapa fitur count sangat berguna untuk tree model?

---

## Clustering With K-Means

### Poin inti

- Clustering termasuk **unsupervised learning**: tidak memakai target; tujuannya mempelajari struktur fitur — dalam konteks prediksi berfungsi sebagai teknik "feature discovery".
- Cluster label sebagai fitur bekerja seperti *binning*/*discretization* pada satu fitur, dan seperti "multi-dimensional binning" (*vector quantization*) pada banyak fitur.
- Fitur `Cluster` bersifat **kategorikal** (hasil algoritma berupa label encoding integer); untuk model tertentu one-hot encoding lebih tepat.
- Motivasi: cluster memecah hubungan rumit antar fitur menjadi potongan sederhana — strategi "divide and conquer". Contoh: hubungan melengkung `YearBuilt`–`SalePrice` membuat model linear *underfit*, tetapi di dalam cluster hubungannya nyaris linear.
- K-means mengukur kemiripan dengan **jarak Euclidean**, menempatkan **centroid**, dan menetapkan tiap titik ke centroid terdekat; `k` = jumlah centroid/cluster yang kamu tentukan sendiri.
- Hasil k-means setara **Voronoi tessellation** ruang fitur; tessellation inilah yang "dipelajari" untuk menentukan cluster data baru.
- Tiga parameter sklearn: `n_clusters` (k), `max_iter` (batas iterasi), `n_init` (jumlah pengulangan inisialisasi; dipilih yang total jaraknya terkecil). Biasanya hanya `n_clusters` yang perlu kamu set sendiri, dan sebaiknya di-tuning seperti hyperparameter (mis. lewat cross-validation).
- k-means sensitif terhadap skala, jadi fitur dengan rentang ekstrem sebaiknya di-standardize/dinormalisasi lebih dulu.

### Contoh kode

```python
import pandas as pd
from sklearn.cluster import KMeans

df = pd.read_csv("housing.csv")
X = df.loc[:, ["MedInc", "Latitude", "Longitude"]]

# Standardisasi bila skala antar fitur jauh berbeda (di sini sudah mirip)
# X = (X - X.mean()) / X.std()

kmeans = KMeans(n_clusters=6, n_init=10, random_state=0)
X["Cluster"] = kmeans.fit_predict(X)      # label cluster per baris
X["Cluster"] = X["Cluster"].astype("category")  # ingat: ini kategorikal
```

```python
# Alternatif fitur: jarak ke tiap centroid (bukan label)
dist = kmeans.transform(X[["MedInc", "Latitude", "Longitude"]])
dist_df = pd.DataFrame(dist, columns=[f"Centroid_{i}" for i in range(dist.shape[1])])
```

### Kesalahan umum

- Tidak melakukan scaling sehingga fitur berskala besar (mis. luas tanah) mendominasi jarak dan cluster jadi tidak bermakna.
- Memperlakukan label cluster sebagai angka berurutan untuk model linear/pohon tanpa mempertimbangkan one-hot encoding — padahal cluster tidak punya urutan.
- Memakai `n_init` kecil atau `k` asal-asalan tanpa tuning, lalu menyimpulkan clustering "tidak membantu" padahal solusinya local optimum.

### Drill

1. Mengapa clustering disebut teknik unsupervised, dan apa perannya dalam feature engineering?
2. Apa yang dimaksud cluster label berperilaku seperti binning pada satu fitur?
3. Jelaskan dua langkah iteratif inti algoritma k-means.
4. Apa fungsi `n_init` dan mengapa nilainya penting?
5. Dua cara memakai hasil k-means sebagai fitur, dan kapan masing-masing lebih cocok?

---

## Principal Component Analysis

### Poin inti

- PCA adalah metode berbasis model kedua setelah clustering: jika clustering mempartisi data berdasarkan kedekatan, PCA mempartisi **variasi** data.
- Gagasan inti: alih-alih mendeskripsikan data dengan fitur asli, deskripsikan dengan **axes of variation**; sumbu-sumbu itu menjadi fitur baru.
- Komponen baru adalah **kombinasi linear** (weighted sum) fitur asli; bobotnya disebut **loadings**, dan jumlah komponen = jumlah fitur asli. Tanda/magnitudo loadings menunjukkan jenis variasi yang ditangkap (contoh Abalone: `Size` = Height + Diameter, `Shape` = Height − Diameter).
- Setiap komponen punya **percent of explained variance**; variansi besar **tidak otomatis** berarti prediktif — tergantung target.
- Dua cara pemakaian: (a) **deskriptif** — hitung MI score komponen untuk melihat variasi mana yang paling prediktif, lalu ciptakan fitur turunan (produk, rasio) atau clustering pada komponen; (b) **komponen sebagai fitur**.
- Use case komponen sebagai fitur: dimensionality reduction untuk fitur multicollinear, anomaly detection pada komponen variansi rendah, noise reduction (meningkatkan signal-to-noise), dan decorrelation untuk algoritma yang kesulitan dengan fitur berkorelasi.
- Best practice: PCA hanya untuk fitur numerik; sensitif skala sehingga standardisasi dulu; pertimbangkan membuang/membatasi outlier karena pengaruhnya besar.
- Catatan penting: PCA hanya rotasi — jarak antar titik tidak berubah, jadi clustering dengan seluruh komponen identik dengan clustering fitur asli. Pilih subset komponen (variansi/MI tertinggi) bila ingin manfaat.

### Contoh kode

```python
import pandas as pd
from sklearn.decomposition import PCA

df = pd.read_csv("autos.csv")
y = df.pop("price")
features = ["highway_mpg", "engine_size", "horsepower", "curb_weight"]
X = df.loc[:, features]

# 1) Standardisasi (wajib bila skala berbeda)
X_scaled = (X - X.mean(axis=0)) / X.std(axis=0)

# 2) Fit PCA, ambil komponen
pca = PCA()
X_pca = pca.fit_transform(X_scaled)
component_names = [f"PC{i+1}" for i in range(X_pca.shape[1])]
X_pca = pd.DataFrame(X_pca, columns=component_names)

# 3) Loadings: baris = fitur asli, kolom = komponen
loadings = pd.DataFrame(
    pca.components_.T,
    columns=component_names,
    index=X.columns,
)
print(loadings)

# 4) Explained variance per komponen
print(pca.explained_variance_ratio_)
```

```python
# 5) Pakai PCA sebagai alat deskriptif: komponen mana yang informatif?
from sklearn.feature_selection import mutual_info_regression

mi = mutual_info_regression(X_pca, y, discrete_features=False, random_state=0)
print(pd.Series(mi, index=X_pca.columns).sort_values(ascending=False))

# 6) Terjemahkan komponen kembali menjadi fitur yang bisa dipakai model
df["sports_or_wagon"] = X.curb_weight / X.horsepower   # dari kontras PC3
```

### Kesalahan umum

- Menjalankan PCA pada data mentah tanpa standardisasi sehingga komponen didominasi fitur dengan satuan/skala besar.
- Menganggap PC1 (variance terbesar) pasti fitur terbaik untuk target; variansi ≠ daya prediksi.
- Memberi PCA data yang masih mengandung missing value atau kolom kategorikal tanpa encoding — PCA hanya menerima fitur numerik lengkap.

### Drill

1. Apa yang dimaksud "axes of variation", dan bagaimana hubungannya dengan fitur asli?
2. Apa itu loadings, dan bagaimana tanda loadings membantu interpretasi komponen?
3. Mengapa komponen dengan explained variance kecil masih bisa berguna?
4. Sebutkan empat use case komponen PCA sebagai fitur.
5. Mengapa clustering memakai seluruh komponen PCA sama saja dengan clustering fitur asli?

---

## Target Encoding

### Poin inti

- Target encoding ditujukan untuk fitur **kategorikal**: mengganti kategori dengan angka yang diturunkan dari **target** — karena itu disebut teknik **supervised**.
- Versi paling sederhana: group aggregation pada target, mis. `groupby("make")["price"].transform("mean")`. Nama lain: *mean encoding*; untuk target biner disebut *bin counting* (juga: likelihood/impact/leave-one-out encoding).
- Dua masalah utama: (1) **unknown categories** — kategori yang tidak ada di data encoding akan menjadi NaN saat di-join ke data masa depan; (2) **rare categories** — statistik dari kategori dengan sedikit observasi tidak akurat dan memicu overfitting.
- Solusi: **smoothing** — campur rata-rata dalam kategori dengan rata-rata keseluruhan:
  `encoding = weight * in_category + (1 - weight) * overall`, dengan `weight = n / (n + m)`.
- `m` adalah smoothing factor: makin besar `m`, makin besar bobot rata-rata keseluruhan (kategori jarang "ditarik" ke rata-rata global). Pilih `m` berdasarkan seberapa noisy kategori diperkirakan.
- Konsekuensi penting: target encoding berisiko overfitting sehingga encoder harus dilatih pada **encoding split terpisah**, lalu di-join ke split lain.
- Use case: fitur **high-cardinality** (one-hot akan meledak jumlah kolom) dan fitur yang secara domain penting meskipun skor metrik fiturnya rendah.
- Implementasi: `category_encoders.MEstimateEncoder(cols=[...], m=...)` dari scikit-learn-contrib.

### Contoh kode

```python
import pandas as pd

autos = pd.read_csv("autos.csv")

# Bentuk paling dasar: mean encoding (perhatikan: dipakai hanya untuk eksplorasi,
# bukan untuk train langsung tanpa split — lihat catatan overfitting)
autos["make_encoded"] = autos.groupby("make")["price"].transform("mean")
```

```python
from category_encoders import MEstimateEncoder

X = df.copy()
y = X.pop("Rating")

# 1) Pisahkan data khusus untuk MELATIH encoder
X_encode = X.sample(frac=0.25, random_state=0)
y_encode = y[X_encode.index]
X_pretrain = X.drop(X_encode.index)      # sisa data untuk training model
y_train = y[X_pretrain.index]

# 2) Fit encoder hanya pada encoding split, transform ke sisa data
encoder = MEstimateEncoder(cols=["Zipcode"], m=5.0)
encoder.fit(X_encode, y_encode)
X_train = encoder.transform(X_pretrain)

# 3) Cek: kategori baru (tidak terlihat saat fit) -> diisi prior/overall mean
```

### Kesalahan umum

- Melatih encoder dan mentransformasi baris yang sama (in-fold encoding), sehingga target tiap baris "bocor" ke fiturnya sendiri dan skor validasi jadi terlalu optimistis.
- Tidak memakai smoothing untuk kategori jarang, sehingga kategori dengan 1–2 observasi mendapat nilai ekstrem yang tidak representatif.
- Lupa menangani kategori yang tidak muncul di encoding split; nilainya menjadi NaN dan harus diimputasi (default `handle_unknown="value"` mengisi prior).

### Drill

1. Mengapa target encoding disebut teknik supervised, berbeda dari one-hot/label encoding?
2. Sebutkan dua nama lain untuk target encoding pada target biner/umum.
3. Tuliskan rumus smoothing m-estimate dan arti setiap simbol.
4. Dua masalah apa yang diselesaikan smoothing?
5. Kapan target encoding lebih dipilih daripada one-hot encoding?

---

## Feature Engineering for House Prices (bonus project)

### Poin inti

- Ini project gabungan yang menerapkan lesson 1–6 pada kompetisi *House Prices — Advanced Regression Techniques*; alurnya 5 step: preliminaries → feature utility scores → create features → hyperparameter tuning → train & submit.
- **Preprocessing**: load CSV, `clean` (perbaiki typo kategori seperti `Brk Cmn` → `BrkComm`, perbaiki `GarageYrBlt` yang korup, rename kolom berawalan angka), `encode` (set tipe statistik: nominal `category`, ordinal `CategoricalDtype` dengan levels terurut, plus level `"None"` untuk missing), `impute` (0 untuk numerik, `"None"` untuk kategorikal).
- **Baseline + utility**: `score_dataset` melakukan label encoding, `np.log(y)`, lalu cross-validated RMSE pada target log (setara RMSLE); sesudahnya `make_mi_scores` dipakai untuk membuang fitur berskor 0.0 (`drop_uninformative`) karena fitur tak informatif bisa memicu overfitting.
- **Create features**: pipeline modular berisi fungsi terpisah — `mathematical_transforms` (LivLotRatio, Spaciousness), `interactions` (dummy BldgType × GrLivArea), `counts` (PorchTypes), `break_down` (MSSubClass), `group_transforms` (median GrLivArea per Neighborhood).
- **k-Means**: dua varian fitur — `cluster_labels` (label cluster) dan `cluster_distance` (jarak ke tiap centroid, `Centroid_i`), dengan standardisasi dan `n_init=50`.
- **PCA**: `apply_pca` mengembalikan `(pca, X_pca, loadings)`; dipakai untuk fitur "PCA-inspired" (`GrLivArea + TotalBsmtSF`, `YearRemodAdd * TotalBsmtSF`), komponen langsung, dan indikator outlier (`Neighborhood == "Edwards" & SaleCondition == "Partial"`).
- **Target encoding hemat data**: `CrossFoldEncoder` — latih encoder pada satu fold, transform fold lain, ulangi; prediksi test memakai rata-rata encoding dari semua fold. Bisa dipakai dengan encoder apa pun dari `category_encoders` (mis. `CatBoostEncoder`).
- **Tuning & submit**: XGBoost dengan parameter manual atau Optuna; target di-log saat fit dan di-`exp` saat prediksi karena XGB meminimalkan MSE sementara loss kompetisi RMSLE.

### Contoh kode

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from xgboost import XGBRegressor

# 1) Skor RMSLE via CV pada target log
def score_dataset(X, y, model=XGBRegressor()):
    for col in X.select_dtypes(["category"]):
        X[col] = X[col].cat.codes       # label encoding untuk XGB
    log_y = np.log(y)
    score = cross_val_score(model, X, log_y, cv=5,
                            scoring="neg_mean_squared_error")
    return np.sqrt(-score.mean())

# 2) Buang fitur tanpa informasi
def drop_uninformative(df, mi_scores):
    return df.loc[:, mi_scores > 0.0]

# 3) Fitur turunan ala lesson 3
def mathematical_transforms(df):
    X = pd.DataFrame()
    X["LivLotRatio"] = df.GrLivArea / df.LotArea
    X["Spaciousness"] = (df.FirstFlrSF + df.SecondFlrSF) / df.TotRmsAbvGrd
    return X
```

```python
# 4) Target encoding lintas fold (tanpa membuang data)
class CrossFoldEncoder:
    def __init__(self, encoder, **kwargs):
        self.encoder_, self.kwargs_ = encoder, kwargs
        self.cv_ = KFold(n_splits=5)

    def fit_transform(self, X, y, cols):
        self.fitted_encoders_, self.cols_ = [], cols
        parts = []
        for idx_encode, idx_train in self.cv_.split(X):
            enc = self.encoder_(cols=cols, **self.kwargs_)
            enc.fit(X.iloc[idx_encode, :], y.iloc[idx_encode])
            parts.append(enc.transform(X.iloc[idx_train, :])[cols])
            self.fitted_encoders_.append(enc)
        out = pd.concat(parts)
        out.columns = [c + "_encoded" for c in out.columns]
        return out

encoder = CrossFoldEncoder(MEstimateEncoder, m=1)
X = X.join(encoder.fit_transform(X, y, cols=["MSSubClass"]))
```

```python
# 5) Train + submit (log target karena loss kompetisi RMSLE)
X_train, X_test = create_features(df_train, df_test)
y_train = df_train.loc[:, "SalePrice"]

xgb = XGBRegressor(**xgb_params)
xgb.fit(X_train, np.log(y_train))
predictions = np.exp(xgb.predict(X_test))

pd.DataFrame({"Id": X_test.index, "SalePrice": predictions}) \
  .to_csv("my_submission.csv", index=False)
```

### Kesalahan umum

- Menghitung MI score atau target encoding pada gabungan train+test sebelum split, sehingga informasi test/target bocor ke fitur training.
- Lupa bahwa XGBoost meminimalkan MSE sementara metrik kompetisi RMSLE — tanpa `log`/`exp` pada target, skor CV tidak sebanding dengan leaderboard.
- Menyalakan semua transformasi sekaligus tanpa mengukur kontribusi tiap blok; di notebook aslinya beberapa transformasi dikomentari justru karena tidak menolong (mis. `TotalOutsideSF`).

### Drill

1. Sebutkan lima step alur project House Prices.
2. Mengapa `score_dataset` melakukan `np.log(y)` sebelum cross-validation?
3. Apa perbedaan fitur dari `cluster_labels` dan `cluster_distance`?
4. Bagaimana `CrossFoldEncoder` menghindari pemborosan data dibanding encoding split 25%?
5. Mengapa fitur dengan MI score 0.0 dibuang dalam pipeline?

---

## Kunci Jawaban

**What Is Feature Engineering**

1. Memperbaiki performa prediksi, mengurangi kebutuhan komputasi/data, meningkatkan interpretabilitas hasil.
2. Karena model linear hanya bisa belajar hubungan linear; `Area = Length**2` membuat hubungan harga–luas menjadi linear, sehingga model dapat menyesuaikan parabola lewat fitur `Area`. Transformasi menjadi bagian dari model.
3. Sebagai pembanding terukur (cross-validated) untuk menilai apakah fitur baru benar-benar memperbaiki performa atau sebaiknya dibuang.
4. Fitur rasio: `FCRatio` (fine/coarse aggregate), `AggCmtRatio` ((coarse+fine)/cement), `WtrCmtRatio` (water/cement). Rasio komposisi lebih menentukan kekuatan beton daripada jumlah absolut bahan.
5. Model linear dan neural network paling diuntungkan normalisasi (NN butuh skala dekat 0); tree-based seperti random forest/XGBoost biasanya jauh lebih sedikit butuh.

**Mutual Information**

1. Korelasi hanya menangkap hubungan linear; MI menangkap segala jenis hubungan (non-linear, non-monotonik).
2. Minimum 0.0 = kedua variabel independen. Nilai di atas ~2.0 jarang karena MI bersifat logaritmik sehingga naik sangat lambat.
3. MI mengukur asosiasi satu fitur dengan target secara terpisah, tidak mendeteksi interaksi; akibatnya fitur yang hanya berguna dalam kombinasi bisa mendapat skor rendah dan salah dibuang.
4. `mutual_info_regression` untuk target kontinu (real-valued), `mutual_info_classif` untuk target kategorikal.
5. `fuel_type`: MI-nya rendah tetapi memisahkan dua populasi `price` dengan tren berbeda di dalam `horsepower` — indikasi interaction effect.

**Creating Features**

1. Mathematical transforms, counts, building-up/breaking-down features, group transforms.
2. Saat fitur mengandung nilai 0 (atau sangat kecil); `log1p` menghitung log(1+x) sehingga 0 tetap valid dan tidak menghasilkan `-inf`.
3. `df[cols].gt(0).sum(axis=1)` — ubah tiap kolom menjadi boolean > 0 lalu jumlahkan per baris. Untuk kolom yang sudah biner/boolean cukup `.sum(axis=1)`.
4. Buat split, hitung transform hanya pada train (`groupby(kat)[col].transform("mean")`), `drop_duplicates` pada pasangan kategori–nilai, lalu `merge(..., on=kat, how="left")` ke validasi.
5. Tree model tidak punya cara alami mengagregasi informasi dari banyak fitur sekaligus, sehingga count (ringkasan "berapa banyak") memberi informasi yang sulit mereka bangun sendiri.

**Clustering With K-Means**

1. Karena tidak memakai target; ia mempelajari struktur/properti data. Perannya sebagai "feature discovery" yang menghasilkan fitur baru (label cluster atau jarak centroid).
2. Pada satu fitur, cluster membagi nilai menjadi beberapa kelompok — sama seperti binning/discretization; pada banyak fitur ia menjadi binning multi-dimensi (vector quantization).
3. (a) Menetapkan tiap titik ke centroid terdekat; (b) memindahkan tiap centroid untuk meminimalkan jarak ke titik-titiknya. Diulang sampai centroid stabil atau `max_iter` tercapai.
4. `n_init` = berapa kali algoritma dijalankan dengan inisialisasi berbeda; dipilih hasil dengan total jarak terkecil. Ini menghindari hasil buruk akibat inisialisasi acak yang jelek (local optimum).
5. (a) Label cluster sebagai kolom kategorikal — mudah diinterpretasi, cocok untuk model tree atau dengan one-hot; (b) jarak ke tiap centroid (`transform`) — memberi informasi gradual tentang kedekatan ke segmen, berguna bila batas cluster terlalu kasar.

**Principal Component Analysis**

1. Sumbu yang menggambarkan cara data bervariasi (arah di mana titik-titik paling menyebar); sumbu ini merupakan kombinasi linear fitur asli dan menjadi fitur baru pengganti fitur asli.
2. Loadings adalah bobot (koefisien) kombinasi linear tiap fitur asli pada sebuah komponen. Tanda menunjukkan arah: tanda sama = fitur bergerak searah, tanda berlawanan = berkontras (mis. `Shape` = Height − Diameter).
3. Karena variansi tidak sama dengan daya prediksi; komponen variansi rendah bisa menangkap pola halus yang justru berhubungan dengan target, atau berguna untuk anomaly/noise detection.
4. Dimensionality reduction pada fitur multicollinear, anomaly detection, noise reduction, decorrelation.
5. Karena PCA hanyalah rotasi ruang fitur — jarak Euclidean antar titik tidak berubah, sehingga partisi cluster yang dihasilkan identik.

**Target Encoding**

1. Karena nilai encoding diturunkan dari target (statistik target per kategori), bukan semata dari fitur itu sendiri; one-hot/label encoding tidak menyentuh target.
2. Bin counting (untuk target biner) dan mean encoding; nama lain: likelihood encoding, impact encoding, leave-one-out encoding.
3. `encoding = weight * in_category + (1 - weight) * overall`, dengan `weight = n / (n + m)`; `n` = jumlah kemunculan kategori, `m` = smoothing factor (makin besar, makin berat ke rata-rata keseluruhan).
4. (a) Kategori jarang yang statistiknya tidak akurat, (b) kategori tak dikenal (tidak ada di encoding split) yang bisa diisi rata-rata keseluruhan alih-alih NaN.
5. Untuk fitur high-cardinality (one-hot akan menghasilkan terlalu banyak kolom) dan untuk fitur yang secara domain penting tetapi skornya rendah pada metrik fitur.

**Feature Engineering for House Prices**

1. Preliminaries (load/clean/encode/impute + baseline), feature utility scores (MI), create features (Pandas/k-means/PCA/target encoding), hyperparameter tuning, train model & create submission.
2. Karena metrik kompetisi adalah RMSLE; XGBoost meminimalkan MSE, sehingga target di-log agar MSE pada log-target setara RMSE pada skala log = RMSLE. Prediksi kemudian di-`exp` kembali.
3. `cluster_labels` menghasilkan satu kolom label cluster (kategorikal); `cluster_distance` menghasilkan k kolom jarak ke tiap centroid (numerik kontinu).
4. Dengan K-fold: encoder dilatih pada satu fold dan digunakan untuk mentransformasi fold lain, berulang untuk semua fold — semua baris akhirnya ter-encode oleh encoder yang tidak melihatnya, tanpa perlu menyisihkan 25% data hanya untuk encoding.
5. Karena fitur tanpa informasi tidak membantu dan dapat memicu overfitting; membuangnya di notebook justru memberi kenaikan performa moderat.

---

## Ringkasan 1 paragraf

Course ini membangun satu gagasan tunggal — fitur yang baik adalah fitur yang hubungannya dengan target bisa dipelajari model — lalu menyediakan perkakas bertingkat untuk mencapainya. Lesson 1 menetapkan disiplin kerja: tentukan tujuan, buat baseline cross-validated, dan ingat bahwa transformasi fitur pada dasarnya memperluas kapasitas model. Lesson 2 memberi cara memilih titik mulai secara sistematis lewat mutual information, metrik univariate yang menangkap hubungan non-linear tetapi buta terhadap interaksi sehingga harus dilengkapi visualisasi dan domain knowledge. Lesson 3 mengajarkan transformasi Pandas sehari-hari (aritmetika, log, count, pemecahan/penggabungan string, group transform) beserta panduan model mana yang diuntungkan oleh tipe fitur mana, plus keharusan menghitung group transform hanya dari training set. Lesson 4 dan 5 menambahkan dua metode unsupervised: k-means yang mengubah kedekatan spasial menjadi label cluster atau jarak centroid, dan PCA yang mengubah struktur variasi menjadi komponen beserta loadings yang bisa dibaca sebagai resep fitur baru. Lesson 6 menutup dengan target encoding untuk fitur kategorikal high-cardinality, lengkap dengan mekanisme smoothing m-estimate dan keharusan encoding split terpisah. Bonus project merangkai semuanya menjadi pipeline modular pada kompetisi House Prices — preprocessing, penyaringan MI, feature creation, clustering/PCA, `CrossFoldEncoder`, tuning, dan submission dengan target log — yang menunjukkan bahwa nilai sebenarnya bukan pada satu teknik, melainkan pada siklus ukur–transformasi–ukur ulang terhadap baseline.

---

## Sumber

Halaman yang berhasil dibuka dan dibaca:

- `https://www.kaggle.com/learn/feature-engineering` — daftar resmi lesson (6 lesson + bonus), durasi, instructor.
- `https://www.kaggle.com/code/ryanholbrook/mutual-information/tutorial` — teks lesson penuh (dibaca via render frame notebook).
- `https://www.kaggle.com/code/ryanholbrook/clustering-with-k-means/tutorial` — teks lesson penuh (dibaca via render frame notebook).
- `https://www.kaggle.com/code/ryanholbrook/what-is-feature-engineering/tutorial` — metadata halaman + daftar section (Welcome, The Goal of Feature Engineering, A Guiding Principle, Example – Concrete Formulations, Continue).
- `https://www.kaggle.com/code/ryanholbrook/creating-features/tutorial` — metadata halaman + daftar section (Mathematical Transforms, Counts, Building-Up and Breaking-Down Features, Group Transforms).
- `https://www.kaggle.com/code/ryanholbrook/principal-component-analysis/tutorial` — metadata halaman + daftar section (Principal Component Analysis, PCA for Feature Engineering, Example – 1985 Automobiles).
- `https://www.kaggle.com/code/ryanholbrook/target-encoding/tutorial` — metadata halaman + daftar section (Target Encoding, Smoothing, Example – MovieLens1M).

Catatan akses: halaman notebook Kaggle merender isi di dalam frame `kaggleusercontent` yang tidak selalu bisa diambil (beberapa percobaan gagal/timeout, dan `web_extract` biasa hanya mengembalikan kerangka halaman). Untuk lesson 1, 3, 5, 6, dan bonus, teks di atas dicocokkan dengan salinan notebook resmi yang dipublikasikan publik, dan akurasinya diverifikasi dengan membandingkan teks salinan terhadap teks halaman resmi untuk lesson 2 dan lesson 4 — keduanya identik kata per kata. Judul section pada semua lesson juga dicek terhadap daftar isi halaman resmi.

Dokumentasi resmi pendukung:

- `https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_regression.html` — parameter `discrete_features`, sifat MI non-negatif, estimasi berbasis k-NN.
- `https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html` — `explained_variance_ratio_`, `components_`, catatan bahwa input di-center tetapi tidak di-scale.
- `https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html` — `n_clusters`, `n_init`, `max_iter`, `transform` (cluster-distance space), kecenderungan local minimum.
- `https://contrib.scikit-learn.org/category_encoders/mestimate.html` — `MEstimateEncoder`, parameter `m`, `handle_unknown`/`handle_missing`.

Tidak ada sertifikat atau klaim kelulusan dalam catatan ini; ini murni materi belajar.
