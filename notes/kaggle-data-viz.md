# Catatan Belajar — Kaggle Learn: Data Visualization

**Course:** Data Visualization (`slug: data-visualization`)
**Instructor:** Alexis Cook
**Jumlah lesson:** 8 (7 lesson materi + 1 lesson workflow)
**Estimasi waktu:** ±4 jam
**Library utama:** seaborn (`sns`), matplotlib (`plt`), pandas (`pd`)

> Catatan ini adalah materi belajar mandiri yang disusun ulang dari halaman tutorial resmi Kaggle Learn.
> Isi lesson mengikuti teks aslinya (contoh dataset, nama kolom, dan urutan pembahasan).
> Tidak ada klaim sertifikat atau kelulusan di dokumen ini.

**Cara pakai:** kerjakan `### Drill` di setiap lesson **sebelum** melihat bagian **Kunci Jawaban**.
Kunci jawaban sengaja diletakkan di bagian paling bawah file agar bisa dipakai untuk self-test.

**Catatan path file:** semua contoh memakai path gaya Kaggle Notebook (`../input/nama_file.csv`).
Di luar Kaggle (laptop/Colab), ganti path itu dengan lokasi file CSV milikmu — sisa kodenya identik.

---

## Hello, Seaborn

Lesson pembuka: pengenalan lingkungan notebook, cara memuat CSV dengan pandas, dan chart pertama dengan seaborn.

### Poin inti

- **seaborn** adalah library visualisasi Python yang dibangun di atas matplotlib; satu baris kode sudah cukup untuk membuat chart yang rapi.
- Halaman tutorial berupa **Jupyter notebook**: campuran sel teks, sel kode (kotak abu-abu), dan output yang muncul tepat di bawah sel kode.
- Ada **boilerplate setup** yang harus dijalankan di awal setiap notebook: import pandas, matplotlib, seaborn, plus magic `%matplotlib inline`, lalu `print("Setup Complete")`.
- `pd.read_csv()` memuat dataset ke sebuah variabel; parameternya bisa mengatur kolom index dan tipe data.
- `index_col="Date"` memakai kolom `Date` sebagai label baris, sehingga tanggal tidak menjadi kolom biasa.
- `parse_dates=True` membuat pandas membaca label baris sebagai **datetime**, bukan teks/angka.
- Baris yang diawali `#` adalah **komentar** — diabaikan komputer, hanya untuk manusia. Sisanya **executable code**.
- `.head()` menampilkan lima baris pertama untuk memastikan data termuat benar.

### Contoh kode

```python
# --- Boilerplate: dijalankan di sel paling atas setiap notebook ---
import pandas as pd
pd.plotting.register_matplotlib_converters()   # daftarkan converter tanggal pandas -> matplotlib
import matplotlib.pyplot as plt
%matplotlib inline                             # magic Jupyter: gambar tampil di dalam notebook
import seaborn as sns
print("Setup Complete")                        # output yang diharapkan: Setup Complete

# --- Muat dataset FIFA (6 negara: ARG, BRA, ESP, FRA, GER, ITA) ---
fifa_filepath = "../input/fifa.csv"            # di luar Kaggle: ganti path ini
fifa_data = pd.read_csv(fifa_filepath, index_col="Date", parse_dates=True)

# --- Cek data: 5 baris pertama ---
fifa_data.head()

# --- Chart pertama: line chart semua kolom ---
plt.figure(figsize=(16, 6))                    # lebar 16 inci, tinggi 6 inci
sns.lineplot(data=fifa_data)
```

### Kesalahan umum

- **Lupa `parse_dates=True`** (atau lupa `index_col="Date"`): sumbu x jadi deretan angka `0, 1, 2, ...` alih-alih tanggal, dan garis jadi tidak berurutan secara waktu.
- **Menjalankan sel chart sebelum sel setup**: muncul `NameError: name 'sns' is not defined` karena import belum dieksekusi pada sesi notebook itu.
- **Menganggap `%matplotlib inline` itu Python biasa**: baris itu *magic command* Jupyter; kalau ditempel ke file `.py` yang dijalankan lewat terminal, hasilnya `SyntaxError`.

### Drill

1. Perintah untuk menampilkan lima baris pertama sebuah DataFrame adalah…
   a. `data.tail()`  b. `data.head()`  c. `data.first()`  d. `data.top()`
2. Apa fungsi `parse_dates=True` pada `pd.read_csv(...)`?
   a. Menghapus kolom tanggal  b. Mengurutkan data berdasarkan tanggal  c. Membaca kolom tanggal sebagai tipe datetime  d. Mengubah format tanggal ke `dd/mm/yyyy`
3. Setelah sel setup dijalankan, output apa yang tercetak?
   a. `Done`  b. `Setup Complete`  c. Tidak ada output  d. `True`
4. Baris kode yang diawali tanda `#` disebut… dan perlakuannya saat kode dijalankan adalah…
   a. string; dicetak  b. komentar; diabaikan  c. operator; dihitung  d. docstring; disimpan
5. **(pilih semua yang benar)** Tentang `index_col="Date"`:
   a. Kolom `Date` dipakai sebagai label baris
   b. Kolom `Date` tetap muncul sebagai kolom biasa
   c. Kolom `Date` tidak lagi menjadi kolom data biasa
   d. Wajib dipakai bersama `parse_dates=True` agar tanggal dikenali sebagai datetime

---

## Line Charts

Line chart untuk melihat **tren sepanjang waktu**; dataset Spotify berisi stream harian lima lagu populer 2017–2018.

### Poin inti

- Chart tren dibuat dengan **`sns.lineplot`**; polanya selalu `sns.lineplot(data=nama_dataframe)`.
- Jika DataFrame punya beberapa kolom numerik, seaborn otomatis menggambar **satu garis per kolom** beserta legendanya; untuk dataset lain cukup ganti nama variabel di dalam `data=`.
- `plt.figure(figsize=(lebar, tinggi))` mengatur ukuran figure dalam inci; harus dipanggil **sebelum** fungsi plot.
- `plt.title("...")` memberi judul; teks judul **wajib** diapit tanda kutip.
- Untuk memplot satu kolom saja: `sns.lineplot(data=spotify_data['Shape of You'], label="Shape of You")` — nama kolom dalam kutip tunggal di dalam kurung siku.
- `label=` menambahkan entri pada **legend**; tanpa `label=`, garis hasil subset tidak muncul di legenda.
- `list(df.columns)` mencetak semua nama kolom — cara aman memastikan ejaan nama kolom sebelum dipakai.
- Nilai kosong pada dataset tampil sebagai **`NaN`** ("Not a Number"), misalnya lagu yang belum rilis pada tanggal tersebut.

### Contoh kode

```python
import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

# --- Muat data Spotify (stream harian per lagu) ---
spotify_filepath = "../input/spotify.csv"
spotify_data = pd.read_csv(spotify_filepath, index_col="Date", parse_dates=True)

spotify_data.head()        # 5 baris pertama; NaN = lagu belum rilis
spotify_data.tail()        # 5 baris terakhir
list(spotify_data.columns) # cek nama kolom sebelum memilih subset

# --- Semua kolom dalam satu chart ---
plt.figure(figsize=(14, 6))
plt.title("Daily Global Streams of Popular Songs in 2017-2018")
sns.lineplot(data=spotify_data)

# --- Hanya dua lagu ---
plt.figure(figsize=(14, 6))
plt.title("Daily Global Streams of Popular Songs in 2017-2018")
sns.lineplot(data=spotify_data['Shape of You'], label="Shape of You")
sns.lineplot(data=spotify_data['Despacito'], label="Despacito")
plt.xlabel("Date")         # label sumbu horizontal
```

### Kesalahan umum

- **Lupa `label=` saat memplot subset**: garis muncul, tetapi tidak ada keterangan lagu di legend sehingga pembaca tidak tahu garis mana milik siapa.
- **Salah eja nama kolom** (termasuk beda huruf besar/kecil dan spasi): hasilnya `KeyError: 'shape of you'` — selalu cek dulu dengan `list(df.columns)`.
- **`plt.figure()` diletakkan setelah fungsi plot**: ukuran figure tidak berubah, karena `figsize` hanya berlaku untuk figure yang aktif saat plot digambar.

### Drill

1. Fungsi seaborn untuk membuat line chart adalah…
   a. `sns.lineplot`  b. `sns.plotline`  c. `sns.line`  d. `sns.plot`
2. Apa fungsi argumen `label="Despacito"` pada `sns.lineplot(...)`?
   a. Memberi judul chart  b. Memberi label sumbu x  c. Menampilkan nama garis di legend  d. Mengganti nama kolom di DataFrame
3. Kode untuk memplot **hanya** kolom `'HUMBLE.'` adalah…
   a. `sns.lineplot(data=spotify_data.HUMBLE.)`  b. `sns.lineplot(data=spotify_data['HUMBLE.'], label="HUMBLE.")`  c. `sns.lineplot(data='HUMBLE.')`  d. `sns.lineplot(x='HUMBLE.')`
4. Perintah untuk melihat daftar seluruh nama kolom pada `spotify_data` adalah…
   a. `spotify_data.head()`  b. `list(spotify_data.columns)`  c. `spotify_data.index`  d. `print(spotify_data)`
5. **(pilih semua yang benar)** Tentang `plt.figure(figsize=(14, 6))`:
   a. Lebar 14 inci, tinggi 6 inci
   b. Harus dipanggil sebelum fungsi plot agar berpengaruh
   c. Mengubah jumlah baris data
   d. Nilainya boleh diganti sesuai kebutuhan

---

## Bar Charts and Heatmaps

Bar chart membandingkan kuantitas antar kategori; heatmap memakai warna untuk menemukan pola pada tabel angka. Dataset: keterlambatan penerbangan AS 2015.

### Poin inti

- **`sns.barplot(x=..., y=...)`**: panjang/tinggi batang ditentukan oleh `y`, sedangkan `x` menentukan kategori.
- Untuk data yang barisnya di-index oleh sebuah kolom, kategori diambil lewat **`flight_data.index`**, bukan `flight_data['Month']`.
- Saat memuat data penerbangan dipakai `index_col="Month"` **tanpa** `parse_dates=True`, karena `Month` berisi angka 1–12, bukan tanggal.
- Isi dataset: rata-rata **arrival delay** (menit) per maskapai per bulan; nilai **negatif** berarti rata-rata pesawat tiba lebih awal.
- **`sns.heatmap(data=flight_data, annot=True)`** mewarnai seluruh tabel sekaligus; `annot=True` menampilkan angka di setiap sel.
- Tanpa `annot=True`, heatmap hanya menampilkan gradasi warna tanpa angka — lebih ringkas, tapi kehilangan presisi nilai.
- Membaca pola: bulan 9–11 tampak lebih gelap untuk hampir semua maskapai, artinya jadwal kedatangan relatif lebih baik pada bulan-bulan itu.
- Dataset kecil (12 baris) sehingga seluruh isinya bisa dicetak hanya dengan menulis nama variabelnya: `flight_data`.

### Contoh kode

```python
import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

# --- Muat data: index = Month, kolom = kode maskapai ---
flight_filepath = "../input/flight_delays.csv"
flight_data = pd.read_csv(flight_filepath, index_col="Month")  # bukan tanggal -> tanpa parse_dates

flight_data          # dataset kecil: cetak seluruh isinya

# --- Bar chart: delay maskapai NK (Spirit) per bulan ---
plt.figure(figsize=(10, 6))
plt.title("Average Arrival Delay for Spirit Airlines Flights, by Month")
sns.barplot(x=flight_data.index, y=flight_data['NK'])   # index, BUKAN flight_data['Month']
plt.ylabel("Arrival delay (in minutes)")

# --- Heatmap: semua maskapai x semua bulan ---
plt.figure(figsize=(14, 7))
plt.title("Average Arrival Delay for Each Airline, by Month")
sns.heatmap(data=flight_data, annot=True)   # annot=True -> angka tampil di sel
plt.xlabel("Airline")
```

### Kesalahan umum

- **Memakai `x=flight_data['Month']`**: langsung gagal dengan `KeyError: 'Month'`, karena kolom itu sudah dipindahkan menjadi index saat `read_csv`. Selalu pakai `flight_data.index`.
- **Lupa `annot=True`**: angka hilang dari sel heatmap sehingga pembaca hanya melihat warna tanpa nilai pasti.
- **Memberi heatmap data non-numerik** (misalnya kolom berisi teks): muncul `ValueError: could not convert string to float` — pilih hanya kolom numerik sebelum membuat heatmap.

### Drill

1. Mengapa `sns.barplot(x=flight_data['Month'], ...)` menghasilkan error?
   a. `barplot` tidak menerima argumen `x`  b. Kolom `Month` sudah menjadi index, bukan kolom biasa  c. Tipe datanya salah  d. Harus memakai `hue=`
2. Apa efek `annot=True` pada `sns.heatmap`?
   a. Menambahkan judul  b. Menampilkan nilai angka di setiap sel  c. Mengubah colormap  d. Mengurutkan kolom
3. Nilai delay **negatif** pada dataset berarti…
   a. Data hilang  b. Penerbangan dibatalkan  c. Rata-rata pesawat tiba lebih awal  d. Kesalahan input
4. Untuk membuat heatmap dari seluruh tabel `flight_data` beserta angkanya…
   a. `sns.heatmap(data=flight_data)`  b. `sns.heatmap(data=flight_data, annot=True)`  c. `sns.barplot(data=flight_data)`  d. `sns.heatmap(flight_data.index)`
5. **(pilih semua yang benar)** Tentang lesson ini:
   a. Bar chart cocok membandingkan kuantitas antar grup
   b. `x` pada bar chart bisa diisi index DataFrame
   c. Heatmap memerlukan data numerik
   d. `parse_dates=True` wajib dipakai untuk kolom `Month`

---

## Scatter Plots

Scatter plot memanfaatkan bidang koordinat untuk melihat **hubungan antar variabel**. Dataset: biaya asuransi sintetis (`insurance.csv`).

### Poin inti

- **`sns.scatterplot(x=..., y=...)`** menampilkan hubungan dua variabel kontinu; pola menanjak = **positif berkorelasi**.
- **`sns.regplot`** = scatter plot + **garis regresi** (best fit) untuk mengecek kekuatan hubungan linear.
- **`hue=`** mewarnai titik berdasarkan kategori, sehingga satu chart bisa menampilkan **tiga variabel** sekaligus.
- **`sns.lmplot`** menggambar beberapa garis regresi sekaligus (satu per grup `hue`) — di sinilah terlihat bahwa slope perokok jauh lebih curam.
- Gaya penulisan `lmplot` berbeda: `x`, `y`, dan `hue` diisi **nama kolom sebagai string** (`x="bmi"`), lalu dataset diberikan lewat `data=insurance_data`.
- **`sns.swarmplot`** membuat *categorical scatter plot*: satu sumbu berisi variabel kategorikal (mis. `smoker`), sumbu lain kontinu (mis. `charges`).
- Setiap titik pada swarmplot adalah satu observasi; titik "diacak" sedikit secara horizontal agar tidak saling menumpuk.
- Interpretasi dataset: non-perokok membayar lebih sedikit; pemilik biaya tertinggi adalah perokok dan biaya terendah adalah non-perokok.

### Contoh kode

```python
import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

# --- Muat data asuransi ---
insurance_filepath = "../input/insurance.csv"
insurance_data = pd.read_csv(insurance_filepath)
insurance_data.head()

# --- Scatter plot dasar: BMI vs biaya ---
sns.scatterplot(x=insurance_data['bmi'], y=insurance_data['charges'])

# --- Scatter + garis regresi ---
sns.regplot(x=insurance_data['bmi'], y=insurance_data['charges'])

# --- Tiga variabel: warna = status perokok ---
sns.scatterplot(x=insurance_data['bmi'], y=insurance_data['charges'],
                hue=insurance_data['smoker'])

# --- Dua garis regresi (satu per grup): perhatikan gaya argumennya ---
sns.lmplot(x="bmi", y="charges", hue="smoker", data=insurance_data)

# --- Categorical scatter: variabel kategorikal di sumbu x ---
sns.swarmplot(x=insurance_data['smoker'], y=insurance_data['charges'])
```

### Kesalahan umum

- **Mencampur dua gaya penulisan**: `sns.lmplot(x=insurance_data['bmi'], ...)` gagal (`KeyError`) karena `lmplot` mengharapkan **nama kolom string** + `data=`, bukan Series.
- **Berharap `plt.figure(figsize=...)` mengubah ukuran `lmplot`**: `lmplot` adalah *figure-level function* dan membuat figure sendiri (default 5×5 inci); atur ukuran lewat `height=`/`aspect=` pada `lmplot` itu sendiri.
- **Menaruh variabel kategorikal di `x`/`y` tanpa mempertimbangkan penumpukan titik**: gunakan `swarmplot` (atau `stripplot`) agar distribusi tiap kategori tetap terbaca, bukan `scatterplot` biasa.

### Drill

1. Menambahkan garis regresi pada scatter plot dilakukan dengan…
   a. `sns.lineplot`  b. `sns.regplot`  c. `sns.lmplot` saja  d. `plt.plot`
2. Argumen `hue=` pada scatter plot berfungsi untuk…
   a. Mengubah ukuran titik  b. Mewarnai titik berdasarkan kategori  c. Menambah judul  d. Mengatur sumbu y
3. Perbedaan utama `sns.lmplot` dibanding `sns.regplot` adalah…
   a. Tidak butuh data  b. Bisa menggambar beberapa garis regresi per grup `hue`  c. Hanya untuk data kategorikal  d. Tidak bisa memakai `hue`
4. Chart untuk hubungan variabel kontinu vs variabel kategorikal adalah…
   a. `sns.swarmplot`  b. `sns.heatmap`  c. `sns.lineplot`  d. `sns.histplot`
5. **(pilih semua yang benar)** Tentang `sns.lmplot(x="bmi", y="charges", hue="smoker", data=insurance_data)`:
   a. `x` dan `y` berisi nama kolom, bukan Series
   b. Dataset diberikan lewat `data=`
   c. Menghasilkan dua garis regresi
   d. Ukurannya mengikuti `plt.figure(figsize=...)` yang dipanggil sebelumnya

---

## Distributions

Histogram dan density plot untuk melihat **sebaran nilai** sebuah variabel. Dataset: 150 bunga iris dari tiga spesies.

### Poin inti

- **`sns.histplot`** mengelompokkan nilai numerik ke dalam bin dan menghitung frekuensinya; cukup beri satu kolom.
- **KDE (kernel density estimate)** plot adalah versi halus dari histogram; dibuat dengan **`sns.kdeplot`**.
- Parameter pewarnaan area di bawah kurva pada materi asli adalah `shade=True`; pada seaborn modern parameternya **`fill=True`** (`shade` sudah deprecated).
- **2D KDE** dibuat dengan **`sns.jointplot(..., kind="kde")`**: pusat figure berisi KDE dua dimensi, sedangkan kurva di atas dan di kanan adalah KDE marginal untuk masing-masing sumbu.
- Pada area 2D KDE, bagian yang lebih gelap menandakan kombinasi nilai yang **lebih mungkin** muncul.
- Untuk membandingkan antar kategori: `sns.histplot(data=iris_data, x='Petal Length (cm)', hue='Species')` — `data` = DataFrame, `x` = nama kolom, `hue` = kolom pemisah.
- `sns.kdeplot` menerima kombinasi `data`, `x`, `hue` yang sama, sehingga distribusi tiap spesies bisa ditumpuk dalam satu figure.
- Temuan dataset: *Iris setosa* terpisah jelas dari dua spesies lain; panjang petal < 2 cm hampir pasti *Iris setosa*.

### Contoh kode

```python
import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

# --- Muat data iris: index = Id ---
iris_filepath = "../input/iris.csv"
iris_data = pd.read_csv(iris_filepath, index_col="Id")
iris_data.head()

# --- Histogram satu variabel ---
sns.histplot(iris_data['Petal Length (cm)'])

# --- KDE (versi halus histogram); seaborn modern: fill=True ---
sns.kdeplot(data=iris_data['Petal Length (cm)'], fill=True)  # materi asli: shade=True

# --- 2D KDE + KDE marginal di atas dan di kanan ---
sns.jointplot(x=iris_data['Petal Length (cm)'],
              y=iris_data['Sepal Width (cm)'], kind="kde")

# --- Distribusi per spesies ---
sns.histplot(data=iris_data, x='Petal Length (cm)', hue='Species')
plt.title("Histogram of Petal Lengths, by Species")

sns.kdeplot(data=iris_data, x='Petal Length (cm)', hue='Species', fill=True)
plt.title("Distribution of Petal Lengths, by Species")
```

### Kesalahan umum

- **Menulis `sns.histplot(a=iris_data['Petal Length (cm)'])`**: argumen `a=` milik `distplot` versi lama; di `histplot` hasilnya `AttributeError` (unexpected keyword argument 'a'). Berikan kolom secara posisional atau lewat `x=`.
- **Menyalin `shade=True` ke seaborn baru**: muncul peringatan deprecation (`shade` is now deprecated in favor of `fill`) dan akan error pada rilis berikutnya — pakai `fill=True`.
- **Memakai KDE pada data diskrit / banyak nilai berulang**: kurva halus bisa menyesatkan (bisa melewati batas nilai yang mustahil, mis. negatif); histogram lebih jujur untuk data seperti itu.

### Drill

1. Fungsi untuk membuat histogram adalah…
   a. `sns.kdeplot`  b. `sns.histplot`  c. `sns.barplot`  d. `sns.jointplot`
2. KDE plot paling tepat digambarkan sebagai…
   a. Histogram dengan bin lebih banyak  b. Versi halus (smoothed) dari histogram  c. Scatter plot dua variabel  d. Diagram batang kategorikal
3. Untuk membuat 2D KDE beserta distribusi marginal tiap sumbu, dipakai…
   a. `sns.jointplot(x=..., y=..., kind="kde")`  b. `sns.heatmap`  c. `sns.histplot`  d. `sns.lmplot`
4. Parameter untuk mewarnai area di bawah kurva KDE pada seaborn modern adalah…
   a. `color=True`  b. `fill=True`  c. `area=True`  d. `annot=True`
5. **(pilih semua yang benar)** Tentang `sns.histplot(data=iris_data, x='Petal Length (cm)', hue='Species')`:
   a. `data` menunjuk DataFrame sumber
   b. `x` berisi nama kolom yang diplot
   c. `hue` memisahkan data menjadi beberapa histogram
   d. `hue` wajib berupa Series, bukan nama kolom

---

## Choosing Plot Types and Custom Styles

Lesson rangkuman: memetakan jenis pertanyaan ke jenis chart, lalu mengganti tema tampilan dengan satu baris kode.

### Poin inti

- Chart dikelompokkan menjadi tiga kategori besar: **Trends**, **Relationship**, dan **Distribution** — pilih berdasarkan pertanyaan yang ingin dijawab.
- **Trends → `sns.lineplot`**: pola perubahan sepanjang waktu; beberapa garis bisa membandingkan beberapa grup.
- **Relationship**: `sns.barplot` (bandingkan kuantitas antar grup), `sns.heatmap` (pola warna pada tabel angka), `sns.scatterplot` (dua variabel kontinu, bisa ditambah variabel kategorikal lewat warna).
- Masih di Relationship: `sns.regplot` (scatter + garis regresi linear), `sns.lmplot` (beberapa garis regresi untuk beberapa grup berwarna), `sns.swarmplot` (kontinu vs kategorikal).
- **Distribution**: `sns.histplot` (satu variabel numerik), `sns.kdeplot` (distribusi halus satu atau dua variabel), `sns.jointplot` (2D KDE + KDE marginal).
- **`sns.set_style("...")`** mengganti tema seluruh chart berikutnya hanya dengan satu baris.
- Seaborn punya **lima tema**: `"darkgrid"`, `"whitegrid"`, `"dark"`, `"white"`, dan `"ticks"`; `"darkgrid"` adalah default.
- Nama tema harus persis; nilai lain langsung ditolak dengan `ValueError` yang menyebutkan daftar tema valid.

### Contoh kode

```python
import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

# --- Data contoh: stream harian Spotify ---
spotify_filepath = "../input/spotify.csv"
spotify_data = pd.read_csv(spotify_filepath, index_col="Date", parse_dates=True)

# --- Tema default (darkgrid) ---
plt.figure(figsize=(12, 6))
sns.lineplot(data=spotify_data)

# --- Ganti tema dengan satu baris; berlaku untuk figure berikutnya ---
sns.set_style("dark")
plt.figure(figsize=(12, 6))
sns.lineplot(data=spotify_data)

# --- Lima tema yang tersedia ---
for tema in ["darkgrid", "whitegrid", "dark", "white", "ticks"]:
    sns.set_style(tema)
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=spotify_data)
    plt.title(f"style: {tema}")

sns.set_style("darkgrid")   # kembalikan ke default
```

### Kesalahan umum

- **Salah ketik nama tema** (`"dargkrid"`, `"greygrid"`): langsung `ValueError: style must be one of white, dark, whitegrid, darkgrid, ticks`.
- **Mengira `set_style` hanya mengubah satu chart**: efeknya **global** untuk semua figure yang dibuat setelahnya di sesi itu, jadi kembalikan ke default bila perlu.
- **Memilih chart yang tidak cocok dengan pertanyaan**: tren waktu dengan `barplot` (atau heatmap) menyembunyikan bentuk perubahan; gunakan `lineplot` untuk trends.

### Drill

1. Chart paling tepat untuk menunjukkan tren penjualan per bulan selama dua tahun adalah…
   a. `sns.barplot`  b. `sns.lineplot`  c. `sns.heatmap`  d. `sns.histplot`
2. `sns.heatmap` paling berguna untuk…
   a. Distribusi satu variabel  b. Menemukan pola warna pada tabel angka  c. Hubungan kontinu vs kategorikal  d. Tren waktu
3. Berapa jumlah tema bawaan seaborn?
   a. 3  b. 4  c. 5  d. 7
4. Mana yang **bukan** nama tema seaborn yang valid?
   a. `darkgrid`  b. `whitegrid`  c. `greygrid`  d. `ticks`
5. **(pilih semua yang benar)** Tentang `sns.set_style(...)`:
   a. Cukup satu baris kode
   b. Mempengaruhi figure yang dibuat setelahnya
   c. Hanya mempengaruhi figure yang sedang aktif
   d. `darkgrid` adalah default seaborn

---

## Final Project

Lesson tanpa kode baru: memperkenalkan **Kaggle Datasets** agar bisa memakai dataset sendiri, lalu latihan membuat visualisasi bebas.

### Poin inti

- Kaggle Datasets diakses di **https://www.kaggle.com/datasets**; berisi puluhan ribu dataset publik.
- Jenis file beragam: CSV, JSON, SQLite, hingga BigQuery — course ini fokus pada dataset yang punya **file CSV**.
- Pencarian dataset memakai search bar (contoh di lesson: ketik `"comic"` lalu pilih *FiveThirtyEight Comic Characters Dataset*).
- Halaman dataset punya **Data Explorer** di kiri untuk melihat daftar file dan preview isinya; file pertama (sering `README.md`) terpilih sebagai default.
- Untuk memakai data sendiri: klik **"New Dataset"** di kaggle.com/datasets, lalu unggah file.
- Kaggle otomatis mengonversi data tabular (mis. Google Spreadsheet atau Excel) menjadi **CSV**.
- Pola kerja tetap sama seperti lesson sebelumnya: muat dengan `pd.read_csv`, cek isi, baru memilih jenis chart sesuai pertanyaan.

### Contoh kode

```python
import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

# --- Ganti dengan file CSV dari dataset Kaggle yang kamu pilih ---
filepath = "../input/dc-wikia-data.csv"   # contoh: salah satu file Comic Characters
data = pd.read_csv(filepath)

print(data.shape)   # (jumlah baris, jumlah kolom) -> cek ukuran dulu
data.head()         # lihat 5 baris pertama sebelum memplot apa pun
list(data.columns)  # daftar kolom: pastikan nama & tipe datanya masuk akal

# --- Setelah paham isinya, pilih chart sesuai pertanyaan ---
# Contoh: distribusi satu kolom numerik
sns.histplot(data['YEAR'])
plt.title("Distribusi tahun kemunculan karakter komik")
```

> Untuk latihan di luar Kaggle (butuh internet), dataset contoh bawaan seaborn bisa dipakai sebagai pengganti:
> `data = sns.load_dataset("tips")` — API pemanggilan chart-nya sama persis.

### Kesalahan umum

- **Memilih dataset tanpa file CSV** (hanya JSON/SQLite/BigQuery): `pd.read_csv` gagal; di lesson ini selalu pilih dataset yang punya minimal satu file CSV.
- **Path file salah** (nama file beda huruf, atau lupa folder `../input/`): muncul `FileNotFoundError: [Errno 2] No such file or directory`.
- **Langsung memplot tanpa memeriksa data**: tipe kolom salah (angka terbaca sebagai teks), banyak `NaN`, atau kolom tanggal belum diparse — selalu jalankan `shape`, `head()`, dan `columns` lebih dulu.

### Drill

1. Alamat halaman Kaggle Datasets adalah…
   a. `kaggle.com/datasets`  b. `kaggle.com/data`  c. `kaggle.com/learn/datasets`  d. `kaggle.com/files`
2. Jenis file yang dipakai sepanjang course ini adalah…
   a. JSON  b. CSV  c. SQLite  d. Parquet
3. Untuk mengunggah dataset milik sendiri, tombol yang ditekan di halaman datasets adalah…
   a. "New Notebook"  b. "New Dataset"  c. "Upload Model"  d. "Fork"
4. Jika kamu mengunggah file Excel atau Google Spreadsheet, Kaggle akan…
   a. Menolaknya  b. Mengubahnya menjadi CSV  c. Mengubahnya menjadi JSON  d. Menyimpannya sebagai gambar
5. **(pilih semua yang benar)** Sebelum memvisualisasikan dataset baru, langkah yang masuk akal adalah:
   a. Cari dataset dan pilih yang punya file CSV
   b. Buka Data Explorer untuk melihat preview file
   c. Jalankan `head()`/`shape` untuk memeriksa isi
   d. Langsung membuat heatmap tanpa memeriksa tipe kolom

---

## Creating Your Own Notebooks

Lesson penutup: workflow membuat notebook sendiri di Kaggle, dari notebook kosong sampai menyimpan gambar untuk presentasi.

### Poin inti

- Notebook baru dibuat di **https://www.kaggle.com/code** lewat tombol **"+ New Notebook"** di kiri atas.
- Langkah pertama: pastikan bahasa notebook adalah **Python** melalui menu **File > Language**.
- Hapus kode default yang disediakan, lalu ganti dengan **boilerplate setup** yang sama seperti di semua exercise (`print("Setup Complete")` sebagai penanda berhasil).
- Setelah environment siap, **attach dataset** (lewat panel Input/Add Data) sebelum menulis kode visualisasi.
- Setelah figure jadi, simpan sebagai file gambar agar mudah ditempel ke presentasi/laporan — di matplotlib dipakai **`plt.savefig("nama_file.png")`**.
- Alur kerjanya berulang: setup → attach data → load & cek → plot → simpan.
- Ini lesson terakhir course; isinya workflow, bukan API baru.

### Contoh kode

```python
import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
print("Setup Complete")

# --- Muat data yang sudah di-attach ke notebook ---
# Di Kaggle:  filepath = "../input/nama_file.csv"   (nama file dari dataset yang di-attach)
# Demo lokal (butuh internet): pakai dataset bawaan seaborn
data = sns.load_dataset("tips")
data.head()

# --- Buat figure, lalu simpan sebagai gambar ---
plt.figure(figsize=(10, 6))
plt.title("Judul Chart")
sns.lineplot(data=data)
plt.savefig("chart_saya.png", dpi=150, bbox_inches="tight")  # simpan SEBELUM plt.show()
```

### Kesalahan umum

- **Lupa meng-attach dataset**: `FileNotFoundError` saat `read_csv`, meski path terlihat benar — file harus ditambahkan ke panel Input notebook dulu.
- **Menyimpan gambar setelah `plt.show()`**: figure sudah ditutup/ dikosongkan sehingga file yang dihasilkan kosong; urutannya `savefig` dulu, baru `show`.
- **Membiarkan kode default / salah bahasa**: notebook baru berisi template yang bukan seaborn, dan bila bahasa bukan Python maka `import seaborn` gagal.

### Drill

1. Notebook baru di Kaggle dibuat melalui…
   a. `kaggle.com/code` → "+ New Notebook"  b. `kaggle.com/datasets` → "New Dataset"  c. menu "File > Export"  d. `kaggle.com/learn`
2. Menu untuk memeriksa/mengubah bahasa notebook adalah…
   a. File > Language  b. Edit > Preferences  c. Run > All  d. View > Language
3. Perintah matplotlib untuk menyimpan figure sebagai file gambar adalah…
   a. `plt.export()`  b. `plt.savefig("chart.png")`  c. `plt.write()`  d. `plt.image()`
4. Kode default pada notebook baru sebaiknya…
   a. Dibiarkan  b. Dihapus dan diganti boilerplate setup  c. Dijalankan berkali-kali  d. Diubah ke R
5. **(pilih semua yang benar)** Urutan workflow yang benar untuk notebook baru:
   a. Cek bahasa = Python
   b. Ganti kode default dengan boilerplate setup
   c. Attach dataset sebelum load data
   d. Simpan gambar dengan `savefig` setelah `plt.show()`

---

## Ringkasan 1 paragraf

Course **Data Visualization** di Kaggle Learn membangun satu keterampilan inti: menerjemahkan pertanyaan tentang data menjadi chart yang tepat dengan seaborn, dimulai dari boilerplate `import pandas / matplotlib / seaborn` + `%matplotlib inline`, lalu memuat CSV memakai `pd.read_csv` dengan `index_col` (dan `parse_dates=True` bila kolomnya tanggal) serta memeriksanya lewat `head()`. Dari situ materi bergerak melalui tiga keluarga chart: **trends** dengan `sns.lineplot` (termasuk memplot subset kolom dengan `label=` agar legendanya jelas), **relationship** dengan `sns.barplot` (kategori dari `df.index`), `sns.heatmap(annot=True)` untuk pola pada tabel angka, `sns.scatterplot`/`sns.regplot`/`sns.lmplot` untuk korelasi dua variabel kontinu plus variabel kategorikal lewat `hue=`, dan `sns.swarmplot` untuk kontinu vs kategorikal; serta **distribution** dengan `sns.histplot`, `sns.kdeplot` (parameter area kini `fill=True`, bukan `shade=True`), dan `sns.jointplot(kind="kde")` untuk 2D KDE beserta marginalnya. Dua lesson terakhir tidak menambah API baru, melainkan memperluas cakupan data (Kaggle Datasets, upload CSV sendiri) dan menutup dengan workflow notebook mandiri: buat notebook di kaggle.com/code, pastikan bahasa Python, tempel boilerplate, attach dataset, buat chart, lalu simpan dengan `plt.savefig()` sebelum `plt.show()`. Pola yang berulang di seluruh course adalah keseragaman sintaks — hampir semua chart cukup satu baris `sns.<chart>(...)` — sehingga kesalahan yang paling sering muncul bukan soal konsep, melainkan detail kecil: nama kolom salah ketik, memakai `df['Kolom']` padahal kolom itu sudah jadi index, lupa `label=`/`annot=`, atau memakai nama parameter versi lama.

## Sumber

Halaman yang benar-benar dibuka saat menyusun catatan ini:

**Course & lesson (Kaggle Learn / Kaggle Code)**
- Course index: https://www.kaggle.com/learn/data-visualization
- 1. Hello, Seaborn: https://www.kaggle.com/code/alexisbcook/hello-seaborn
- 2. Line Charts: https://www.kaggle.com/code/alexisbcook/line-charts
- 3. Bar Charts and Heatmaps: https://www.kaggle.com/code/alexisbcook/bar-charts-and-heatmaps
- 4. Scatter Plots: https://www.kaggle.com/code/alexisbcook/scatter-plots
- 5. Distributions: https://www.kaggle.com/code/alexisbcook/distributions
- 6. Choosing Plot Types and Custom Styles: https://www.kaggle.com/code/alexisbcook/choosing-plot-types-and-custom-styles
- 7. Final Project: https://www.kaggle.com/code/alexisbcook/final-project
- 8. Creating Your Own Notebooks: https://www.kaggle.com/code/alexisbcook/creating-your-own-notebooks

**Catatan metodologi:** halaman lesson Kaggle merender isi notebook lewat JavaScript, sehingga teksnya tidak terbaca oleh extractor HTML biasa (yang terambil hanya judul, daftar "Table of Contents", dan metadata). Isi tiap lesson di atas diambil dari **sumber notebook resmi yang sama** melalui Kaggle public kernel API:
`https://www.kaggle.com/api/v1/kernels/pull?userName=alexisbcook&kernelSlug=<slug>` (untuk 8 slug di atas), lalu sel markdown dan sel kode notebook dibaca langsung.

**Dokumentasi pendukung (untuk verifikasi API versi terbaru)**
- seaborn `kdeplot` (parameter `fill`, catatan bandwidth/distribusi): https://seaborn.pydata.org/generated/seaborn.kdeplot.html
- pandas `register_matplotlib_converters`: https://pandas.pydata.org/docs/reference/api/pandas.plotting.register_matplotlib_converters.html

**Verifikasi eksekusi:** semua blok kode di catatan ini dijalankan ulang pada lingkungan lokal (pandas 3.1.1, seaborn 0.13.2, matplotlib, Python 3.14) memakai data sintetis dengan skema kolom yang sama seperti dataset lesson; seluruh pemanggilan chart berhasil, dan tiga perilaku error yang diklaim di bagian "Kesalahan umum" (`flight_data['Month']` → `KeyError`, `histplot(a=...)` → `AttributeError`, `shade=True` → peringatan deprecation) juga dikonfirmasi langsung.

---

## Kunci Jawaban

> Bagian ini sengaja diletakkan paling bawah. Kerjakan drill tiap lesson dulu, baru cocokkan di sini.

### Hello, Seaborn
1. **b** — `data.head()` menampilkan 5 baris pertama.
2. **c** — `parse_dates=True` membuat label baris dibaca sebagai datetime.
3. **b** — `Setup Complete`.
4. **b** — komentar, diabaikan saat kode dijalankan.
5. **a, c, d** — `Date` menjadi index (bukan kolom biasa), dan `parse_dates=True` membuatnya dikenali sebagai tanggal.

### Line Charts
1. **a** — `sns.lineplot`.
2. **c** — `label=` mengisi entri legend untuk garis tersebut.
3. **b** — subset kolom dalam kurung siku + `label=`.
4. **b** — `list(spotify_data.columns)`.
5. **a, b, d** — `figsize` = (lebar, tinggi) dalam inci, harus dipanggil sebelum plot, nilainya bebas diubah; ia tidak menyentuh data.

### Bar Charts and Heatmaps
1. **b** — `Month` sudah dipindah menjadi index oleh `index_col="Month"`, jadi harus `flight_data.index`.
2. **b** — `annot=True` menampilkan angka di tiap sel.
3. **c** — negatif = rata-rata tiba lebih awal dari jadwal.
4. **b** — `sns.heatmap(data=flight_data, annot=True)`.
5. **a, b, c** — `Month` berisi angka 1–12, jadi `parse_dates=True` tidak relevan (bukan tanggal).

### Scatter Plots
1. **b** — `sns.regplot`.
2. **b** — mewarnai titik berdasarkan kategori (variabel ketiga).
3. **b** — `lmplot` bisa menggambar beberapa garis regresi, satu per grup `hue`.
4. **a** — `sns.swarmplot` (categorical scatter plot).
5. **a, b, c** — `lmplot` adalah figure-level function; `plt.figure(figsize=...)` tidak mengubah ukurannya (pakai `height`/`aspect`).

### Distributions
1. **b** — `sns.histplot`.
2. **b** — KDE = histogram yang dihaluskan.
3. **a** — `sns.jointplot(x=..., y=..., kind="kde")`.
4. **b** — `fill=True` (materi asli memakai `shade=True`, kini deprecated).
5. **a, b, c** — `hue` cukup diberi **nama kolom** (string); bukan wajib Series.

### Choosing Plot Types and Custom Styles
1. **b** — `sns.lineplot` untuk trends over time.
2. **b** — menemukan pola warna pada tabel angka.
3. **c** — lima tema.
4. **c** — `greygrid` tidak ada; yang valid: `darkgrid`, `whitegrid`, `dark`, `white`, `ticks`.
5. **a, b, d** — `set_style` bersifat global untuk figure berikutnya, bukan hanya figure aktif.

### Final Project
1. **a** — `kaggle.com/datasets`.
2. **b** — CSV.
3. **b** — "New Dataset".
4. **b** — dikonversi otomatis menjadi CSV.
5. **a, b, c** — memplot tanpa memeriksa tipe kolom adalah sumber error klasik.

### Creating Your Own Notebooks
1. **a** — `kaggle.com/code` → "+ New Notebook".
2. **a** — File > Language.
3. **b** — `plt.savefig("chart.png")`.
4. **b** — hapus, ganti dengan boilerplate setup.
5. **a, b, c** — `savefig` harus dipanggil **sebelum** `plt.show()`, bukan sesudahnya.
