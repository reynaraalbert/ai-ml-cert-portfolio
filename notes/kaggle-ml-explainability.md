# Machine Learning Explainability — Kaggle Learn

Materi belajar lengkap untuk course **Machine Learning Explainability** (slug: `machine-learning-explainability`) di Kaggle Learn. Disusun per lesson sesuai urutan resmi course, dengan poin inti, contoh kode, kesalahan umum, dan drill.

| Item | Keterangan |
|---|---|
| Course | Machine Learning Explainability (Kaggle Learn) |
| Slug | `machine-learning-explainability` |
| Instructor | Dan Becker (DanB) |
| Estimasi durasi | ± 4 jam |
| Level | Intermediate |
| Bahasa | Python |
| Dataset contoh | FIFA 2018 Statistics (Man of the Match) |
| Library inti | `scikit-learn`, `eli5`, `shap`, `matplotlib` |

> Catatan: dokumen ini adalah **materi belajar**, bukan bukti penyelesaian course. Tidak ada klaim kelulusan, badge, atau sertifikat di sini.

## Peta Course

| # | Lesson | Pertanyaan yang dijawab | Level insight |
|---|---|---|---|
| 1 | Use Cases for Model Insights | Kapan dan mengapa insight model dibutuhkan? | Orientasi |
| 2 | Permutation Importance | Fitur mana yang paling berpengaruh bagi model? | Global |
| 3 | Partial Plots | Bagaimana sebuah fitur memengaruhi prediksi? | Global (bentuk efek) |
| 4 | SHAP Values | Mengapa model membuat prediksi *ini*? | Local (per baris) |
| 5 | Advanced Uses of SHAP Values | Pola apa yang muncul saat SHAP diagregasi? | Local → Global |

Alur logikanya: lesson 2–3 memberi gambaran besar (global), lesson 4 turun ke satu prediksi (local), lesson 5 naik lagi ke global tetapi dengan detail yang lebih kaya daripada lesson 2–3.

## Persiapan

```bash
pip install scikit-learn pandas numpy matplotlib shap eli5 graphviz
```

**Catatan versi (penting).** Sejak `shap >= 0.45`, output `explainer.shap_values()` untuk model multi-output (termasuk klasifikasi biner scikit-learn) berubah dari *list of arrays* menjadi *ndarray* dengan dimensi kelas di posisi terakhir. Kode di tutorial Kaggle memakai gaya lama (`shap_values[1]`). Supaya aman di dua versi, pakai helper berikut di semua contoh:

```python
def positive_class_values(sv):
    """Normalisasi output shap_values.

    shap lama  : list [kelas_0, kelas_1]  -> ambil sv[1]
    shap >=0.45: ndarray (n, fitur, kelas) -> ambil [:, :, 1]
    """
    return sv[1] if isinstance(sv, list) else sv[:, :, 1]
```

**Setup bersama** (dipakai di lesson 2–5; lesson 1 tidak punya kode di tutorial resmi):

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('FIFA 2018 Statistics.csv')
y = (data['Man of the Match'] == 'Yes')                      # target: "Yes"/"No" -> bool
feature_names = [c for c in data.columns if data[c].dtype in [np.int64]]  # hanya fitur numerik
X = data[feature_names]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

model = RandomForestClassifier(n_estimators=100, random_state=0).fit(train_X, train_y)
```

---

## Use Cases for Model Insights

Lesson pembuka ini tidak mengajarkan algoritma, melainkan **kerangka berpikir**: insight apa yang mungkin diambil dari model, dan untuk apa.

### Poin inti

- Klaim "model itu black box" tidak sepenuhnya benar. Prediksinya bisa bagus sementara logikanya tidak terbaca, tetapi itu lebih karena **praktisi belum menguasai teknik ekstraksi insight**, bukan karena modelnya mustahil dijelaskan.
- Ada **tiga jenis insight** yang dibahas sepanjang course: (1) fitur mana yang paling penting bagi model, (2) bagaimana tiap fitur memengaruhi **satu** prediksi tertentu, (3) bagaimana sebuah fitur memengaruhi prediksi **secara umum**. Ketiganya berbeda level: (1) dan (3) bersifat **global** (perilaku model secara keseluruhan), (2) bersifat **local** (satu baris data).
- Lima kegunaan utama insight: **debugging**, **informing feature engineering**, **directing future data collection**, **informing human decision-making**, dan **building trust**.
- **Debugging** adalah use case paling bernilai. Data nyata kotor dan tidak konsisten; kode preprocessing menambah sumber error; ditambah risiko **target leakage**. Insight membantu melihat saat pola yang ditemukan model bertentangan dengan pengetahuan domain — biasanya itu langkah pertama menemukan bug.
- **Feature engineering**: ketika fitur berjumlah ratusan dan tidak punya nama bermakna (contoh kompetisi loan default: `f1`, `f2`, …), insight menunjukkan fitur mana yang penting dan mana yang perannya **saling terkait**, sehingga mengarahkan pencarian transformasi. Di kompetisi itu, seorang peserta menemukan fitur `f527 - f528` yang jauh lebih kuat daripada fitur aslinya.
- **Directing data collection**: banyak organisasi bisa menambah jenis data yang dikumpulkan, tetapi mahal. Insight menunjukkan fitur mana yang sudah bernilai dan mana yang berpotensi bernilai, jadi keputusan investasi data tidak buta.
- **Informing human decision-making**: sebagian keputusan tetap diambil manusia (mis. penyedia layanan kesehatan memutuskan intervensi). Di situ **insight lebih berguna daripada prediksi**, karena manusia butuh alasan untuk bertindak.
- **Building trust**: pemangku kepentingan akan memverifikasi beberapa fakta dasar sebelum memercayai model. Menunjukkan insight yang **konsisten dengan pemahaman umum mereka** adalah cara termurah membangun kepercayaan, bahkan pada audiens non-teknis.

### Contoh kode

Lesson 1 tidak menyertakan kode. Yang berguna di tahap ini adalah menyiapkan objek yang nanti dipakai berulang, plus pertanyaan yang harus dijawab sebelum memilih teknik:

```python
# Setup bersama (lihat bagian Persiapan) — dipakai ulang di lesson 2-5
print(f"jumlah fitur: {len(feature_names)}")
print(f"train: {train_X.shape}, validasi: {val_X.shape}")
print(f"baseline akurasi model: {model.score(val_X, val_y):.3f}")

# Tiga pertanyaan yang memetakan ke tiga teknik di course ini:
# 1. Fitur mana yang paling penting?              -> permutation importance (Lesson 2)
# 2. Bagaimana efek tiap fitur secara umum?       -> partial dependence plot (Lesson 3)
# 3. Mengapa prediksi baris INI seperti ini?      -> SHAP values (Lesson 4)
```

Sebelum memilih teknik, jawab dulu: siapa audiensnya (analis, regulator, manajer), keputusan apa yang akan diambil, dan apakah yang dibutuhkan alasan **global** atau alasan untuk **satu kasus**.

### Kesalahan umum

- **Menganggap feature importance sebagai efek kausal.** Insight menjelaskan *apa yang dipelajari model*, bukan *apa yang terjadi di dunia nyata*. Model bisa mengandalkan proxy yang berkorelasi dengan target tanpa menjadi penyebabnya.
- **Menambang insight dari model yang belum divalidasi.** Kalau performa model di holdout buruk, polanya kemungkinan besar noise atau artefak; insight-nya menyesatkan, bukan informatif.
- **Menyerahkan plot teknis mentah ke stakeholder non-teknis.** Nilai insight justru hilang kalau tidak diterjemahkan menjadi pernyataan yang bisa ditindaklanjuti ("fitur ini menaikkan skor risiko sekian"), bukan sekadar gambar.

### Drill

1. Sebutkan tiga tipe insight yang dibahas course dan jelaskan perbedaan level analisis antara ketiganya.
2. Sebuah bank menolak pinjaman seorang nasabah dan wajib menjelaskan dasar penolakan **untuk nasabah itu**. Tipe insight mana yang dibutuhkan, dan mengapa permutation importance tidak memadai?
3. Jelaskan bagaimana insight model bisa membantu menemukan **target leakage**. Beri satu contoh pola yang mencurigakan.
4. Pada kompetisi loan default, fitur `f527` dan `f528` sama-sama menonjol dan perannya "tightly entangled". Teknik mana yang bisa menunjukkan keterkaitan itu, dan transformasi apa yang akhirnya ditemukan peserta?
5. Mengapa insight dari model dengan skor cross-validation rendah tidak layak dipakai untuk mengarahkan pengumpulan data baru?

---

## Permutation Importance

Lesson ini menjawab pertanyaan paling dasar: **fitur mana yang paling berpengaruh terhadap prediksi model?**

### Poin inti

- **Permutation importance dihitung setelah model di-fit.** Model tidak dilatih ulang dan tidak diubah; yang diubah hanya data yang diberikan kepadanya.
- Mekanismenya: **acak (shuffle) satu kolom** pada data validasi, biarkan target dan kolom lain tetap, lalu ukur seberapa besar performa (loss/metric) memburuk. Penurunan performa itulah ukuran kepentingan fitur tersebut.
- Alasannya masuk akal: setelah diacak, hubungan antara kolom itu dan target hancur, sehingga data "tidak lagi berkorespondensi dengan apa pun yang teramati di dunia nyata". Kalau model sangat bergantung pada kolom itu, prediksinya rusak; kalau kolomnya tidak berguna (contoh di tutorial: *socks owned*), prediksi hampir tidak terpengaruh.
- Prosedurnya: (1) ambil model terlatih, (2) acak satu kolom → prediksi → hitung kerugian, (3) kembalikan urutan data, ulangi untuk kolom berikutnya sampai semua kolom selesai.
- Karena pengacakan bersifat random, hasilnya bervariasi. Proses diulang beberapa kali; nilai setelah tanda **±** adalah variasi antar-pengulangan (standar deviasi).
- Nilai bisa **negatif**: prediksi pada data teracak justru kebetulan lebih akurat daripada data asli. Artinya fitur itu semestinya bernilai ≈ 0, tetapi keberuntungan acak menggesernya ke bawah. Ini lebih sering muncul pada dataset kecil.
- Kelebihan yang disebut course: **cepat dihitung**, **luas dipakai dan dipahami**, serta **konsisten dengan properti yang kita inginkan** dari sebuah ukuran feature importance.
- Interpretasi baris: nilai atas = paling penting, nilai bawah = paling tidak penting. Pada contoh soccer, fitur terpenting adalah **Goals scored** — masuk akal secara domain.

### Contoh kode

```python
import eli5
from eli5.sklearn import PermutationImportance

# Model sudah terlatih. .fit() di sini TIDAK melatih ulang model:
# data yang diberikan (val_X, val_y) adalah data yang dipakai untuk mengukur penurunan performa.
perm = PermutationImportance(model, random_state=1).fit(val_X, val_y)

# Tabel HTML: baris atas = paling penting, angka = penurunan skor, ± = variasi antar-shuffle
eli5.show_weights(perm, feature_names=val_X.columns.tolist())
```

Padanan modern tanpa `eli5` (hasil sama secara konsep, lebih mudah diotomasi):

```python
from sklearn.inspection import permutation_importance

r = permutation_importance(
    model, val_X, val_y,
    n_repeats=10,          # jumlah pengulangan shuffle per fitur -> untuk std
    random_state=1,
    scoring='accuracy',    # default classifier = akurasi; ganti utk data imbalanced
)

for i in r.importances_mean.argsort()[::-1]:          # urutkan dari terpenting
    print(f"{val_X.columns[i]:25s} {r.importances_mean[i]:.3f} ± {r.importances_std[i]:.3f}")
```

### Kesalahan umum

- **Menghitung importance di training set.** Model sudah "melihat" data itu, sehingga fitur yang dihafal (termasuk fitur ID-like) tampak sangat penting. Hitung selalu di data yang belum dilihat model (validation/holdout).
- **Membaca importance sebagai arah pengaruh.** Nilai tinggi hanya berarti "model bergantung pada fitur ini", bukan "fitur ini menaikkan prediksi". Arah dan bentuk efek adalah tugas partial dependence plot (Lesson 3) dan SHAP (Lesson 4).
- **Mengabaikan korelasi antar fitur dan pilihan metric.** Pada fitur duplikat/berkorelasi, informasi bisa "dipinjam" dari kolom kembarannya sehingga keduanya tampak tidak penting. Selain itu default scoring untuk classifier adalah akurasi — pada data imbalanced, akurasi menyembunyikan masalah; set `scoring` secara eksplisit (mis. `roc_auc`).

### Drill

1. Permutation importance dihitung sebelum atau sesudah model di-fit? Apa konsekuensinya bagi cara kita menginterpretasikan angkanya?
2. Data validasi berisi 500 baris dan akurasi model 0,80. Setelah kolom `passes` diacak, akurasi turun menjadi 0,62. Berapa nilai importance-nya dan apa artinya dalam kalimat biasa?
3. Mengapa permutation importance bisa bernilai negatif, dan pada kondisi data seperti apa ini paling sering muncul?
4. Jelaskan mengapa menghitung permutation importance pada training set menyesatkan, khususnya untuk fitur seperti nomor urut baris atau ID pelanggan.
5. Dua fitur identik (satu salinan dari yang lain) sama-sama masuk model. Apa yang terjadi pada nilai importance keduanya, dan apa implikasinya ketika kamu menyimpulkan "fitur mana yang penting"?

---

## Partial Plots

Kalau permutation importance menjawab **fitur mana**, partial dependence plot (PDP) menjawab **bagaimana** sebuah fitur memengaruhi prediksi.

### Poin inti

- PDP menunjukkan **bentuk dan arah** efek sebuah fitur: apakah efeknya naik, turun, datar, atau berbelok pada ambang tertentu.
- Pertanyaan yang cocok dijawab PDP: "Dengan mengontrol fitur rumah lainnya, seberapa besar pengaruh longitude dan latitude terhadap harga rumah?" atau "Apakah perbedaan prediksi kesehatan antar dua kelompok disebabkan diet, atau faktor lain?"
- Bagi yang mengenal regresi linear/logistik, PDP bisa dibaca mirip **koefisien**: arah dan besar efek satu variabel. Bedanya, PDP pada model kompleks mampu menangkap pola **non-linear** yang tidak bisa diwakili satu koefisien.
- PDP juga dihitung **setelah model di-fit**, pada model yang dilatih dengan data asli tanpa manipulasi.
- Cara kerjanya: ambil satu baris data, lalu **ubah-ubah nilai satu variabel** pada sederet nilai (grid), prediksi setiap kali, dan gambar hasilnya — mis. tim dengan ball possession 40%, 50%, 60%, sambil fitur lain dibiarkan apa adanya. Karena interaksi bisa membuat plot satu baris tidak tipikal, eksperimen itu diulang pada **banyak baris** dan yang digambar adalah **rata-rata** prediksinya.
- Bentuk kurva membuka struktur model: decision tree dangkal menghasilkan **step function** (tangga), sedangkan random forest menghasilkan kurva **halus** yang umumnya lebih masuk akal. Pada contoh `Distance Covered (Kms)`, random forest menilai peluang menang tertinggi sekitar 100 km, lalu menurun bila berlebihan.
- **2D PDP** memakai pasangan fitur untuk melihat interaksi. Contoh: prediksi tertinggi saat tim mencetak ≥ 1 gol **dan** berlari mendekati 100 km; kalau 0 gol, jarak tempuh tidak berpengaruh.
- Catatan teknis: pada `scikit-learn`, sumbu y sebenarnya adalah **rata-rata prediksi** (untuk classifier: probabilitas, karena default `response_method='auto'` memakai `predict_proba`). Tutorial Kaggle menginterpretasikannya sebagai **perubahan prediksi relatif terhadap nilai paling kiri/baseline**; bentuk kurvanya identik, hanya titik nolnya bergeser. Gunakan `centered=True` bila ingin sumbu y benar-benar berupa selisih terhadap rata-rata.

### Contoh kode

```python
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import PartialDependenceDisplay

# Model sederhana supaya strukturnya mudah dibaca
tree_model = DecisionTreeClassifier(
    random_state=0, max_depth=5, min_samples_split=5
).fit(train_X, train_y)

# PDP 1D: bagaimana "Goal Scored" mengubah prediksi?
fig, ax = plt.subplots(figsize=(7, 4))
PartialDependenceDisplay.from_estimator(tree_model, val_X, ['Goal Scored'], ax=ax)
plt.show()
# Bacaan: mencetak 1 gol menaikkan peluang secara tajam; gol tambahan hampir tidak berpengaruh.
```

```python
# PDP 2D: interaksi dua fitur -> berikan tuple, bukan string
PartialDependenceDisplay.from_estimator(
    tree_model, val_X, [('Goal Scored', 'Distance Covered (Kms)')]
)
```

```python
# Bandingkan dengan model yang lebih kompleks + lihat variasi antar-baris (ICE)
rf_model = RandomForestClassifier(random_state=0).fit(train_X, train_y)

PartialDependenceDisplay.from_estimator(
    rf_model, val_X, ['Distance Covered (Kms)'],
    kind='both',        # 'average' (PDP), 'individual' (ICE), atau 'both'
    centered=True,      # sumbu y = selisih terhadap rata-rata
    subsample=50,       # batasi jumlah kurva ICE agar plot terbaca
)
```

### Kesalahan umum

- **Membaca PDP sebagai efek kausal.** PDP adalah rata-rata prediksi model pada nilai fitur tertentu, bukan hasil intervensi nyata. Ia mewarisi semua bias dan korelasi spurius yang dipelajari model.
- **Mengabaikan asumsi independensi fitur.** PDP membuat grid nilai untuk satu fitur sambil membiarkan fitur lain apa adanya, sehingga bisa menghasilkan kombinasi yang **tidak pernah ada di data** (mis. rumah 10 kamar di area yang seluruhnya rumah kecil). Kalau fitur berkorelasi kuat, kurvanya bisa menyesatkan.
- **Hanya melihat garis rata-rata.** Kurva rata-rata bisa tampak rapi padahal efeknya sangat berbeda antar-baris (interaksi kuat). Selalu cek `kind='individual'`/ICE untuk melihat sebaran efek, dan gunakan 2D PDP bila mencurigai interaksi.

### Drill

1. Pertanyaan apa yang dijawab PDP tetapi tidak dijawab permutation importance?
2. Tuliskan langkah komputasi PDP 1D dalam tiga langkah ringkas.
3. Apa arti sumbu y pada PDP, dan apa yang berubah jika `centered=True`?
4. Mengapa PDP bisa menyesatkan ketika fitur yang diplot berkorelasi dengan fitur lain? Beri satu contoh konkret.
5. Pada contoh `Distance Covered (Kms)`, decision tree menghasilkan kurva bertangga dan random forest menghasilkan kurva halus. Apa yang bisa kamu simpulkan tentang kedua model dari perbedaan bentuk itu, dan kapan 2D PDP lebih tepat dipakai?

---

## SHAP Values

Lesson ini turun dari level model ke level **satu prediksi**: mengapa model memprediksi nilai ini untuk baris ini?

### Poin inti

- **SHAP** adalah singkatan dari **SHapley Additive exPlanations**, berakar pada teori permainan kooperatif (Shapley values): setiap fitur dianggap "pemain" yang berkontribusi pada satu hasil, dan kontribusinya dihitung secara adil.
- SHAP mengukur dampak **nilai tertentu** sebuah fitur **dibandingkan dengan baseline**. Pertanyaan "seberapa besar prediksi didorong oleh fakta tim mencetak 3 gol" lebih mudah dijawab bila dirumuskan sebagai "… dibandingkan sejumlah gol baseline".
- Properti kunci — **additivitas**: `sum(SHAP values semua fitur) = prediksi - nilai baseline`. Jadi kontribusi semua fitur tepat menjelaskan selisih prediksi terhadap baseline, tidak lebih dan tidak kurang.
- Visualisasinya `force_plot`: fitur yang **menaikkan** prediksi berwarna pink, yang **menurunkan** berwarna biru, dan panjang batang menunjukkan besar efeknya. Pada contoh tutorial: prediksi 0,7 dengan `base_value` 0,4979; pendorong terbesar adalah `Goal Scored` = 2, sementara ball possession menurunkan prediksi secara berarti. Selisih total panjang batang pink dan biru sama dengan jarak dari base value ke output.
- Pemilihan explainer: **`TreeExplainer`** untuk model berbasis pohon (cepat dan exact), **`DeepExplainer`** untuk model deep learning, **`KernelExplainer`** untuk model apa pun (paling fleksibel, tetapi lebih lambat dan hanya **aproksimasi**).
- Untuk klasifikasi, `shap_values` memuat satu array per kelas. Yang biasanya dipakai adalah array untuk **kelas positif** (prediksi "menang"), yaitu indeks 1 pada gaya lama.
- Use case yang disebut course bersifat regulatif dan operasional: bank yang **wajib secara hukum** menjelaskan dasar setiap penolakan kredit, dan penyedia layanan kesehatan yang ingin tahu faktor risiko tiap pasien agar bisa melakukan intervensi tepat sasaran.
- Batasan konseptual: SHAP menjelaskan **model**, bukan dunia. Nilainya juga bergantung pada baseline/background yang dipilih, dan pada fitur berkorelasi kredit kontribusi bisa terbagi tidak intuitif.

### Contoh kode

```python
import shap

row_to_show = 5
data_for_prediction = val_X.iloc[row_to_show]

# Konteks dulu: prediksi mentah sebelum melihat dekomposisinya
print(model.predict_proba(data_for_prediction.values.reshape(1, -1)))   # mis. [[0.3, 0.7]]

explainer = shap.TreeExplainer(model)      # exact & cepat untuk model pohon
shap_values = explainer.shap_values(data_for_prediction)

shap.initjs()
# Index [1] = kelas positif ("menang"). Gaya lama: shap_values[1]
shap.force_plot(explainer.expected_value[1], positive_class_values(shap_values), data_for_prediction)
```

```python
# Model apa pun: KernelExplainer (lambat, hasil aproksimasi)
# Beri background sample yang representatif, bukan seluruh training set.
background = shap.sample(train_X, 100)

k_explainer = shap.KernelExplainer(model.predict_proba, background)
k_shap_values = k_explainer.shap_values(data_for_prediction)
shap.force_plot(k_explainer.expected_value[1], k_shap_values[1], data_for_prediction)
```

```python
# API modern (shap >= 0.36): explainer dipanggil langsung, hasilnya shap.Explanation
explanation = explainer(data_for_prediction)
print(explanation.values.shape)      # (n_baris, n_fitur, n_kelas)
shap.plots.force(explanation[0, :, 1])   # kelas positif untuk baris pertama
```

### Kesalahan umum

- **Salah meng-index kelas.** Untuk klasifikasi biner, memakai `shap_values` tanpa memilih kelas akan menghasilkan plot/angka yang membingungkan atau error. Ambil kelas positif (`[1]` pada gaya lama, `[:, :, 1]` pada ndarray modern) dan pastikan `expected_value` diambil dari indeks yang sama.
- **Memakai `KernelExplainer` secara boros.** Ia memanggil model berkali-kali, jadi sangat lambat pada data besar dan hasilnya hanya aproksimasi. Gunakan `TreeExplainer` bila modelnya berbasis pohon; untuk `KernelExplainer`, beri background sample kecil (ratusan baris) yang representatif.
- **Membandingkan SHAP values lintas model/skala secara langsung.** Skala nilai bergantung pada `model_output` (log-odds/raw margin vs probabilitas) dan pada baseline. Angka dari dua model berbeda belum tentu apple-to-apple, dan menjumlahkan kontribusi tanpa memperhatikan baseline akan salah total.

### Drill

1. Tuliskan persamaan additivitas SHAP dan jelaskan arti setiap sukunya.
2. Sebuah prediksi bernilai 0,70 dengan `base_value` 0,50. `Goal Scored` menyumbang +0,30 dan `Ball Possession %` menyumbang −0,12. Berapa total kontribusi fitur-fitur lainnya?
3. Kapan kamu memilih `TreeExplainer`, `DeepExplainer`, dan `KernelExplainer`? Sebutkan trade-off utamanya.
4. Mengapa `shap_values` untuk klasifikasi biner berisi dua array, dan array mana yang biasanya dipakai? Apa yang terjadi kalau salah pilih?
5. Sebutkan dua keterbatasan konseptual SHAP yang harus disebut saat melaporkan hasilnya ke stakeholder.

---

## Advanced Uses of SHAP Values

Lesson terakhir mengagregasi SHAP values untuk mendapatkan insight **global** yang lebih detail daripada permutation importance dan PDP.

### Poin inti

- Pengingat definisi: SHAP value menunjukkan seberapa besar sebuah fitur mengubah prediksi dibandingkan bila fitur itu bernilai baseline. Pada model mainan `y = 4*x1 + 2*x2`, jika `x1 = 2` sedangkan baseline 0, maka SHAP value `x1` = 8.
- Agregasi banyak SHAP values menghasilkan alternatif yang lebih kaya untuk dua teknik sebelumnya: **summary plot** (pengganti/pelengkap permutation importance) dan **dependence contribution plot** (pengganti/pelengkap PDP).
- **Summary plot** (beeswarm) menampilkan satu titik per baris data, dengan tiga atribut: **posisi vertikal** = fitur apa, **warna** = nilai fitur itu tinggi atau rendah, **posisi horizontal** = apakah nilai tersebut mendorong prediksi naik atau turun.
- Yang bisa dibaca dari summary plot: fitur yang **diabaikan model** (pada contoh: `Red` dan `Yellow & Red` menumpuk di 0), fitur yang biasanya tidak berpengaruh tetapi punya **kasus ekstrem** (`Yellow Card`), dan hubungan monoton (`Goal Scored` tinggi → prediksi tinggi).
- Keunggulan dibanding permutation importance: importance tingkat sedang bisa berarti **efek besar untuk sedikit prediksi** atau **efek sedang untuk semua prediksi**. Summary plot membedakan keduanya karena menampilkan sebaran, bukan satu angka.
- **Dependence contribution plot**: sumbu x = nilai fitur sebenarnya, sumbu y = SHAP value fitur itu (dampaknya pada prediksi), warna = fitur interaksi. Titik-titiknya adalah baris data.
- **Sebaran vertikal** pada dependence plot adalah sinyal **interaksi**: dua tim dengan ball possession hampir sama bisa mendapat efek berlawanan, sedangkan regresi linear akan menghasilkan garis lurus tanpa sebaran. Warna membantu menemukan sumber interaksi (pada contoh: titik menyimpang berwarna ungu = tim yang hanya mencetak 1 gol); `interaction_index` boleh tidak diisi karena SHAP memilih fitur interaksi secara otomatis lewat heuristik.
- Catatan performa: menghitung SHAP untuk seluruh dataset bisa lambat. Pengecualiannya model `xgboost`, yang punya optimasi khusus di SHAP sehingga jauh lebih cepat.

### Contoh kode

```python
import shap

explainer = shap.TreeExplainer(model)

# Hitung untuk SELURUH val_X (bukan satu baris) supaya plot punya cukup titik
shap_values = explainer.shap_values(val_X)

# Summary plot (beeswarm): y=fitur, warna=nilai fitur, x=arah & besar efek
shap.summary_plot(positive_class_values(shap_values), val_X)
```

```python
# Dependence contribution plot untuk satu fitur + satu fitur interaksi
shap.dependence_plot(
    'Ball Possession %',
    positive_class_values(shap_values),
    val_X,
    interaction_index='Goal Scored',   # boleh None -> SHAP memilih otomatis
)
# Bacaan: tren naik (makin banyak possession, makin tinggi prediksi),
# tetapi sebaran vertikal -> ada interaksi; titik ungu (1 gol) menyimpang ke bawah.
```

```python
# Dataset besar: subsample dulu, jangan langsung seluruh data
sample_X = val_X.sample(200, random_state=1)
sv = explainer.shap_values(sample_X)
shap.summary_plot(positive_class_values(sv), sample_X, max_display=10)
```

### Kesalahan umum

- **Membaca sumbu y dependence plot sebagai prediksi absolut.** Sumbu y adalah **SHAP value** — selisih prediksi terhadap baseline, bukan probabilitas akhir. Nilai 0 berarti "fitur ini tidak mengubah apa pun", bukan "peluangnya nol".
- **Menganggap warna pada dependence plot sebagai nilai target atau prediksi.** Warna adalah **nilai fitur interaksi** untuk baris tersebut. Salah membaca warna berarti salah menyimpulkan interaksi.
- **Menjalankan SHAP pada seluruh dataset besar tanpa subsample.** Hasilnya lambat, memakan memori, dan plotnya terlalu padat untuk dibaca. Subsample ratusan hingga ribuan baris, atau manfaatkan optimasi `xgboost` bila modelnya memungkinkan.

### Drill

1. Sebutkan tiga karakteristik yang dikodekan oleh setiap titik pada SHAP summary plot dan apa yang ditunjukkan masing-masing.
2. Bagaimana summary plot menjawab kelemahan permutation importance yang berbunyi "importance sedang bisa berarti dua hal yang sangat berbeda"?
3. Pada dependence contribution plot, apa arti sumbu x, sumbu y, dan warna titik? Apa arti titik yang menyimpang jauh dari tren dan berwarna berbeda dari mayoritas?
4. Mengapa dependence contribution plot lebih informatif daripada PDP biasa ketika ada interaksi antar fitur?
5. Mengapa pembuatan summary plot bisa lambat pada dataset besar, dan optimasi apa yang disebut course untuk mengatasinya?

---

## Ringkasan 1 paragraf

Course ini membangun satu rangkaian alat untuk membuka "kotak hitam" model, bergerak dari gambaran global ke penjelasan per prediksi lalu kembali ke global dengan detail lebih kaya. **Permutation importance** menjawab fitur mana yang paling diandalkan model dengan mengacak satu kolom pada data validasi dan mengukur kemerosotan performa — cepat dan model-agnostic, tetapi tidak memberi arah pengaruh dan bisa menyesatkan bila fitur berkorelasi atau dihitung di training set. **Partial dependence plot** melengkapi dengan menunjukkan bentuk dan arah efek sebuah fitur, termasuk versi 2D untuk interaksi, dengan catatan bahwa ia mengasumsikan fitur yang diplot relatif independen dan hanya menampilkan rata-rata. **SHAP values** turun ke level satu baris: berkat sifat additivitas, kontribusi semua fitur tepat menjumlah menjadi selisih antara prediksi dan baseline, sehingga alasan penolakan kredit atau faktor risiko seorang pasien bisa diuraikan per fitur. **Agregasi SHAP** kemudian menghasilkan summary plot yang memisahkan "efek besar untuk sedikit kasus" dari "efek sedang untuk semua kasus", serta dependence contribution plot yang menyingkap interaksi lewat sebaran vertikal dan warna. Benang merahnya: semua teknik ini menjelaskan **perilaku model**, bukan kausalitas dunia nyata — karena itu insight selalu harus diuji dengan pengetahuan domain, dan model yang belum terbukti performanya tidak layak dijadikan sumber insight.

## Sumber

- Course page — Machine Learning Explainability: https://www.kaggle.com/learn/machine-learning-explainability
- Lesson 1 — Use Cases for Model Insights: https://www.kaggle.com/code/dansbecker/use-cases-for-model-insights
- Lesson 2 — Permutation Importance: https://www.kaggle.com/code/dansbecker/permutation-importance
- Lesson 3 — Partial Plots: https://www.kaggle.com/code/dansbecker/partial-plots
- Lesson 4 — SHAP Values: https://www.kaggle.com/code/dansbecker/shap-values
- Lesson 5 — Advanced Uses of SHAP Values: https://www.kaggle.com/code/dansbecker/advanced-uses-of-shap-values
- Dataset — FIFA 2018 Statistics: https://www.kaggle.com/datasets/abecklas/fifa-world-cup
- eli5 documentation (PermutationImportance): https://eli5.readthedocs.io/en/latest/
- SHAP documentation — TreeExplainer: https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html
- SHAP — Migrating to the new Explanation API: https://shap.readthedocs.io/en/stable/example_notebooks/api_examples/migrating-to-new-api.html
- scikit-learn — Permutation feature importance: https://scikit-learn.org/stable/modules/permutation_importance.html
- scikit-learn — PartialDependenceDisplay: https://scikit-learn.org/stable/modules/generated/sklearn.inspection.PartialDependenceDisplay.html

Catatan akses: halaman tutorial Kaggle dirender dengan JavaScript sehingga teksnya tidak terbaca oleh pengambil halaman biasa. Isi lesson di atas diambil dari **sumber resmi notebook tutorial** melalui endpoint publik Kaggle (`/api/v1/kernels/pull?userName=dansbecker&kernelSlug=...`), lalu diverifikasi silang dengan dokumentasi resmi `shap` dan `scikit-learn` untuk bagian API dan catatan versi.

---

## Kunci Jawaban

### Use Cases for Model Insights

1. (a) Fitur mana yang paling penting bagi model — global; (b) bagaimana tiap fitur memengaruhi **satu** prediksi — local; (c) bagaimana sebuah fitur memengaruhi prediksi secara umum/efek tipikal — global. Poin pentingnya: (a) dan (c) berbicara tentang perilaku model secara keseluruhan, sedangkan (b) hanya tentang satu baris data.
2. Yang dibutuhkan adalah insight **local** — dekomposisi prediksi untuk nasabah tersebut, yaitu SHAP values. Permutation importance hanya memberi peringkat kepentingan fitur secara global; ia tidak bisa mengatakan "untuk pemohon ini, rasio utang menurunkan skor sebesar 0,12".
3. Dengan membandingkan pola yang dipelajari model terhadap pengetahuan domain; penyimpangan adalah petunjuk bug atau leakage. Contoh pola mencurigakan: sebuah fitur yang secara logika baru diketahui **setelah** kejadian target (mis. `jumlah_panggilan_penagihan` atau `tanggal_pelunasan`) muncul sebagai fitur paling penting untuk memprediksi gagal bayar.
4. Teknik yang menunjukkan keterkaitan: permutation importance/SHAP yang menampilkan kedua fitur sama-sama penting (dan sering muncul bersama pada dependence plot). Peserta menemukan fitur turunan **`f527 - f528`** (selisih dua fitur) yang jauh lebih kuat daripada masing-masing fitur aslinya.
5. Karena insight menjelaskan **apa yang dipelajari model**, bukan struktur dunia nyata. Model dengan skor validasi rendah kemungkinan besar mempelajari noise atau artefak; menginvestasikan biaya pengumpulan data berdasarkan pola semacam itu berisiko mengumpulkan data yang salah.

### Permutation Importance

1. **Sesudah** model di-fit. Konsekuensinya: angka tersebut mengukur seberapa besar **model yang sudah ada** bergantung pada fitur itu, bukan seberapa prediktif fitur itu secara intrinsik. Model lain bisa memberi peringkat berbeda pada data yang sama.
2. Importance = 0,80 − 0,62 = **0,18** (turun 18 poin persentase). Artinya: ketika hubungan kolom `passes` dengan target dihancurkan, akurasi model turun 0,18 — jadi model cukup bergantung pada fitur itu.
3. Karena pengacakan bersifat acak: kadang data teracak justru kebetulan lebih mudah diprediksi. Nilai negatif menandakan fitur tersebut semestinya ≈ 0. Paling sering muncul pada **dataset kecil**, karena ruang untuk keberuntungan acak lebih besar.
4. Pada training set, model sudah menghafal/menyesuaikan diri pada data itu, sehingga fitur yang overfit — termasuk ID atau nomor urut yang unik per baris — tampak sangat penting. Di data baru fitur seperti itu tidak berguna, jadi importance-nya akan turun drastis; angka di training set memberi gambaran yang terlalu optimistis.
5. Keduanya akan mendapat importance yang **rendah**, karena ketika satu kolom diacak, informasi yang sama masih tersedia di kolom kembarannya sehingga performa tidak turun banyak. Implikasinya: importance rendah tidak otomatis berarti "fitur tidak relevan"; bisa jadi informasinya redundan. Untuk kesimpulan yang aman, kelompokkan fitur berkorelasi dan nilai kepentingannya sebagai satu grup.

### Partial Plots

1. PDP menjawab **bagaimana** sebuah fitur memengaruhi prediksi: bentuk, arah, dan titik belok efeknya. Permutation importance hanya memberi satu angka kepentingan tanpa arah.
2. (1) Ambil baris data (atau banyak baris) dan pilih satu fitur; (2) ganti nilai fitur itu pada sederet nilai grid sambil membiarkan fitur lain tetap, lalu prediksi setiap kali; (3) ulangi untuk banyak baris dan **rata-ratakan** prediksinya, lalu plot nilai fitur (x) versus rata-rata prediksi (y).
3. Sumbu y adalah rata-rata prediksi model pada nilai fitur tersebut (untuk classifier: probabilitas), yang dalam tutorial dibaca sebagai **perubahan prediksi** relatif terhadap nilai paling kiri/baseline. Dengan `centered=True`, sumbu y menjadi selisih prediksi terhadap rata-rata prediksi keseluruhan, sehingga garis nol berarti "tidak ada efek".
4. Karena PDP membuat kombinasi nilai yang mungkin **tidak pernah ada di data nyata**. Contoh: memplot `jumlah_kamar` pada rumah sambil membiarkan `luas_tanah` apa adanya bisa menghasilkan "rumah 10 kamar dengan luas 40 m²" — kombinasi mustahil yang prediksinya tidak bermakna, sehingga kurvanya menyesatkan.
5. Perbedaan bentuk menunjukkan kompleksitas struktur yang dipelajari: tree dangkal memotong ruang fitur menjadi beberapa wilayah konstan (step function), sedangkan random forest merata-ratakan banyak pohon sehingga transisinya halus. 2D PDP dipakai ketika dua fitur kemungkinan **berinteraksi** — yaitu ketika efek satu fitur bergantung pada nilai fitur lain (contoh: jarak tempuh hanya berpengaruh bila tim mencetak gol).

### SHAP Values

1. `sum(SHAP values untuk semua fitur) = prediksi − nilai baseline`. Suku kiri adalah total kontribusi seluruh fitur; `prediksi` adalah output model untuk baris itu; `baseline` (dinamakan `expected_value`) adalah prediksi rata-rata model pada background data.
2. 0,70 − 0,50 = 0,20 total yang harus dijelaskan. Sudah terpakai: +0,30 − 0,12 = +0,18. Maka kontribusi fitur lainnya = 0,20 − 0,18 = **+0,02**.
3. `TreeExplainer` untuk model berbasis pohon (XGBoost, LightGBM, CatBoost, scikit-learn tree/forest) — cepat dan exact. `DeepExplainer` untuk model deep learning. `KernelExplainer` untuk model apa pun — paling fleksibel, tetapi paling lambat dan hanya aproksimasi, sehingga butuh background sample kecil yang representatif.
4. Karena classifier biner memprediksi dua keluaran (tidak menang / menang), dan SHAP menghitung kontribusi untuk masing-masing. Yang dipakai biasanya array **kelas positif** (indeks 1). Kalau salah pilih, kamu menjelaskan sisi keputusan yang berlawanan: tanda kontribusi bisa terbalik dan narasinya ("fitur ini menaikkan peluang") menjadi keliru.
5. (a) SHAP menjelaskan **model**, bukan kausalitas — kontribusi besar bukan bukti pengaruh sebab-akibat; (b) hasilnya bergantung pada pilihan **baseline/background**, dan pada fitur yang berkorelasi kredit kontribusi bisa terbagi secara tidak intuitif antar fitur. (Tambahan: skalanya bergantung pada `model_output`, sehingga tidak selalu bisa dibandingkan langsung antar model.)

### Advanced Uses of SHAP Values

1. (a) **Posisi vertikal** = fitur mana yang digambarkan; (b) **warna** = apakah nilai fitur itu tinggi atau rendah untuk baris tersebut; (c) **posisi horizontal** = apakah nilai itu mendorong prediksi lebih tinggi atau lebih rendah, dan seberapa besar.
2. Karena summary plot menampilkan **setiap baris sebagai satu titik**, bukan satu angka agregat. Importance sedang yang berasal dari "efek besar pada sedikit kasus" akan terlihat sebagai segelintir titik jauh di ujung dengan mayoritas menumpuk di nol; importance sedang yang berasal dari "efek sedang pada semua kasus" akan terlihat sebagai awan titik yang menyebar merata. Pola distribusinya langsung terlihat.
3. Sumbu x = **nilai fitur sebenarnya** pada baris itu; sumbu y = **SHAP value** fitur tersebut (dampaknya terhadap prediksi, bukan prediksi absolut); warna = **nilai fitur interaksi**. Titik yang menyimpang dari tren dan berwarna berbeda menandakan **interaksi**: efek fitur utama berbalik/berubah justru karena nilai fitur interaksi tertentu (contoh: possession tinggi biasanya menaikkan prediksi, tetapi tidak bila tim hanya mencetak 1 gol).
4. Karena PDP hanya menampilkan **rata-rata**, sehingga interaksi tersembunyi di balik satu garis. Dependence contribution plot mempertahankan setiap baris sebagai titik, sehingga sebaran vertikal pada nilai x yang hampir sama langsung menyingkap bahwa efeknya bergantung pada fitur lain, dan pewarnaan membantu menebak fitur mana penyebabnya.
5. Karena menghitung SHAP berarti menjalankan prosedur atribusi untuk setiap baris dan setiap fitur — biayanya tumbuh seiring jumlah baris dan fitur, sehingga lambat dan boros memori pada data besar. Course menyebut **`xgboost`** sebagai pengecualian karena SHAP memiliki optimasi khusus untuk model tersebut. Solusi praktis lain: subsample data sebelum memplot.
