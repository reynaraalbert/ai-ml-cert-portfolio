# Kaggle Learn — Time Series

Catatan belajar lengkap untuk course **Time Series** (slug: `time-series`) di Kaggle Learn.

| Item | Keterangan |
|---|---|
| Instructor | Ryan Holbrook |
| Jumlah lesson | 6 (masing-masing: Tutorial + Exercise) |
| Estimasi waktu | ± 5 jam |
| Prasyarat | Intermediate Machine Learning |
| Dataset course | `ts-course-data` (`book_sales.csv`, `tunnel.csv`, `reserve.csv`, `flu-trends.csv`, `us-retail-sales.csv`) |
| Kompetisi latihan | Store Sales — Time Series Forecasting (Corporación Favorita, ± 1800 kategori produk) |
| Biaya | Gratis |

**Cara pakai catatan ini:** baca *Poin inti* → jalankan *Contoh kode* → cek *Kesalahan umum* → kerjakan *Drill* tanpa melihat kunci. Kunci jawaban ada di bagian paling bawah.

**Tiga komponen yang jadi benang merah course ini:**

```
series = trend + seasons + cycles + error
```

Lesson 1–3 membangun fitur dari **time index** (time dependence). Lesson 4 membangun fitur dari **nilai masa lalu target** (serial dependence). Lesson 5 menggabungkan dua algoritma. Lesson 6 mengubah semuanya menjadi dataset forecasting multi-step yang valid.

---

## 1. Linear Regression With Time Series

### Poin inti

- **Time series** = himpunan observasi yang direkam menurut waktu, biasanya dengan frekuensi teratur (harian, mingguan, bulanan). Bentuk kanonis di pandas: satu kolom observasi + `DatetimeIndex`/`PeriodIndex`.
- **Linear regression** mempelajari kombinasi linear fitur: `target = w1*f1 + w2*f2 + bias`. Parameter dilatih dengan *ordinary least squares* (meminimalkan squared error). `weight` disebut *regression coefficients*, `bias` disebut *intercept*.
- Ada dua jenis fitur yang khas time series: **time-step features** (diturunkan langsung dari time index) dan **lag features** (dibuat dengan menggeser nilai target).
- **Time dummy** adalah time-step feature paling dasar: `np.arange(len(df.index))` — menghitung langkah waktu dari awal sampai akhir. Model `target = w*time + b` memodelkan **time dependence** (nilai bisa diprediksi dari *kapan* ia terjadi).
- **Lag feature** dibuat dengan `df['y'].shift(1)` — menggeser observasi agar tampak terjadi lebih lambat, sehingga nilai masa lalu sejajar satu baris dengan target. Model `target = w*lag + b` memodelkan **serial dependence** (nilai bisa diprediksi dari nilai sebelumnya).
- **Time plot** (x = waktu) untuk melihat time dependence; **lag plot** (x = lag, y = nilai sekarang) untuk melihat serial dependence. Lag plot yang membentuk garis diagonal rapat = lag feature akan berguna.
- Di pandas, `PeriodIndex` (hasil `to_period()`) merepresentasikan kuantitas yang terakumulasi sepanjang periode dan lebih mudah dipakai untuk forecasting daripada `DatetimeIndex` (`Timestamp` = pengukuran pada satu momen).
- Adaptasi ML ke time series pada dasarnya adalah **feature engineering dengan time index dan lag**; algoritma apa pun (linear regression, GBDT, NN) bisa memakai fitur yang sama.

### Contoh kode

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# --- Siapkan series: Date jadi index, satu kolom observasi ---
df = pd.read_csv("../input/ts-course-data/book_sales.csv",
                 index_col="Date", parse_dates=["Date"]).drop("Paperback", axis=1)

# --- Time-step feature: time dummy ---
df["Time"] = np.arange(len(df.index))          # 0,1,2,... sepanjang series

X = df.loc[:, ["Time"]]                        # features (harus 2D)
y = df.loc[:, "Hardcover"]                     # target
model = LinearRegression().fit(X, y)
y_pred = pd.Series(model.predict(X), index=X.index)   # pertahankan index waktu

# --- Lag feature: geser target 1 langkah ---
df["Lag_1"] = df["Hardcover"].shift(1)         # baris pertama jadi NaN

X = df.loc[:, ["Lag_1"]]
X.dropna(inplace=True)                         # buang NaN akibat shifting
y = df.loc[:, "Hardcover"]
y, X = y.align(X, join="inner")                # buang tanggal yang sama di target
model = LinearRegression().fit(X, y)
```

Pola loading data course (dipakai berulang di lesson 2–5):

```python
from pathlib import Path
tunnel = pd.read_csv(Path("../input/ts-course-data") / "tunnel.csv", parse_dates=["Day"])
tunnel = tunnel.set_index("Day").to_period()   # DatetimeIndex -> PeriodIndex
```

### Kesalahan umum

- **Lupa `.align()` setelah `dropna()`.** `X.dropna()` membuang baris pertama, tapi `y` masih punya baris itu → jumlah sampel tidak cocok atau index tidak sinkron. Selalu `y, X = y.align(X, join='inner')`.
- **Menganggap time dummy selalu valid.** `np.arange(len(index))` hanya benar bila tidak ada tanggal yang hilang; kalau ada gap, urutan waktu dan nilai `Time` tidak lagi sesuai. Cek kontinuitas index atau pakai `DeterministicProcess`.
- **Membaca lag plot sebagai bukti kausalitas/trend.** Lag plot hanya menunjukkan korelasi dengan nilai sebelumnya; series dengan trend kuat juga memberi lag plot diagonal karena keduanya naik — di lesson 2 hal ini ditangani dengan detrending/moving average.

### Drill

1. Apa perbedaan **time dependence** dan **serial dependence**? Fitur mana yang memodelkan masing-masing?
2. Series harian 30 hari tanpa tanggal hilang. Tulis satu baris pandas untuk membuat time dummy.
3. Kenapa `X` pada `model.fit(X, y)` harus berupa DataFrame 2D (`df[['Lag_1']]`) dan bukan Series (`df['Lag_1']`)?
4. Sebuah lag plot 1-langkah membentuk awan titik tanpa pola. Apa implikasinya untuk penggunaan `Lag_1` sebagai feature?
5. Setelah `shift(2)`, berapa baris NaN di awal series, dan apa dua opsi penanganannya menurut tutorial?

---

## 2. Trend

### Poin inti

- **Trend** = perubahan jangka panjang yang persisten pada **mean** series; komponen paling lambat bergerak, skala waktu terbesar. Trend bisa juga berupa perubahan persisten pada *variation*, tapi course ini fokus pada trend di mean.
- **Moving average plot** dipakai untuk *melihat* bentuk trend: rata-rata nilai dalam sliding window, sehingga fluktuasi jangka pendek hilang. Lebar window harus **lebih panjang dari satu periode musiman** (mis. window 12 untuk data bulanan dengan musim tahunan; 365 untuk data harian).
- `rolling(window=365, center=True, min_periods=183).mean()` — `center=True` menempatkan rata-rata di tengah window (tidak menggeser trend ke belakang); `min_periods` ± setengah window agar ujung series tetap terisi.
- Setelah bentuk trend diketahui, trend dimodelkan sebagai **transformasi time dummy**: linear `target = a*time + b`, kuadratik `target = a*time**2 + b*time + c`. Linear regression bisa memfit kurva, bukan hanya garis, selama fitur kurvanya disediakan.
- **`DeterministicProcess`** (statsmodels) adalah cara resmi membuat fitur waktu: `constant=True` (intercept), `order=1` (linear), `order=2` (kuadratik), `order=3` (kubik). Ini menghindari "tricky failure cases" yang muncul saat membuat time dummy manual. Istilah *deterministic process* sendiri berarti series yang non-random/sepenuhnya ditentukan (seperti `const` dan `trend`); fitur turunan time index umumnya deterministik.
- `dp.in_sample()` membuat fitur untuk tanggal training; `dp.out_of_sample(steps=n)` membuat fitur untuk masa depan → inilah cara membuat forecast murni tanpa lag.
- Karena `DeterministicProcess(constant=True)` sudah menyediakan intercept, model harus dilatih dengan **`LinearRegression(fit_intercept=False)`** agar tidak ada dua fitur konstan (kolinier).
- Trend model berguna sebagai **baseline**, dan sebagai komponen **hybrid model** untuk algoritma yang tidak bisa mengekstrapolasi trend (XGBoost, random forest).

### Contoh kode

```python
import pandas as pd
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.deterministic import DeterministicProcess

tunnel = pd.read_csv("../input/ts-course-data/tunnel.csv", parse_dates=["Day"])
tunnel = tunnel.set_index("Day").to_period()
y = tunnel["NumVehicles"]

# --- 1. Lihat trend: moving average 365 hari ---
moving_average = tunnel.rolling(window=365, center=True, min_periods=183).mean()

# --- 2. Fitur trend: linear (order=1) ---
dp = DeterministicProcess(
    index=tunnel.index,   # tanggal training
    constant=True,        # intercept
    order=1,              # 1=linear, 2=kuadratik, 3=kubik
    drop=True,            # buang term agar tidak kolinier
)
X = dp.in_sample()

model = LinearRegression(fit_intercept=False)   # intercept sudah ada di X
model.fit(X, y)
y_pred = pd.Series(model.predict(X), index=X.index)

# --- 3. Forecast 30 hari ke depan (out of sample) ---
X_fore = dp.out_of_sample(steps=30)
y_fore = pd.Series(model.predict(X_fore), index=X_fore.index)
```

### Kesalahan umum

- **Memakai `fit_intercept=True` bersama `constant=True`.** Dua kolom konstan → kolinier; `LinearRegression` "behaves badly with duplicated features". Pilih satu: `constant=True` + `fit_intercept=False`, atau tanpa `constant` + `fit_intercept=True`.
- **Window moving average lebih pendek dari satu siklus musiman.** Trend akan tercemar musim (garis moving average ikut bergelombang). Window harus menutupi minimal satu periode musim penuh.
- **Order polinomial terlalu tinggi untuk horizon panjang.** Polinomial orde tinggi meledak di luar rentang data (ekstrapolasi kuadratik/kubik jauh ke depan tidak stabil) — ini risiko yang dieksplorasi di exercise lesson 2. Pilih order serendah mungkin yang menangkap bentuk trend.

### Drill

1. Kenapa window moving average untuk data harian dipilih 365, bukan 7 atau 30?
2. Apa fungsi `center=True` dan `min_periods=183` pada `rolling(...)`?
3. Tulis argumen `DeterministicProcess` untuk trend kuadratik dengan intercept dan pembuangan term kolinier.
4. Apa beda `dp.in_sample()` dan `dp.out_of_sample(steps=30)`?
5. Model dilatih dengan `constant=True`. Kenapa `fit_intercept=False` wajib, dan apa gejala numeriknya kalau dilanggar?

---

## 3. Seasonality

### Poin inti

- **Seasonality** = perubahan periodik yang teratur pada **mean** series, mengikuti jam/kalender: harian, mingguan, tahunan. Berbeda dari trend (lambat, tak berulang) dan dari cycles (tidak terikat tanggal).
- **Seasonal plot** = potongan series yang diplot terhadap periode bersama (mis. semua nilai per jam-dalam-seminggu). Ini alat *diagnostik* sebelum memilih fitur.
- **Seasonal indicators** = one-hot encoding dari periode musiman sebagai fitur kategorikal. Cocok bila musim punya **sedikit observasi** (musim mingguan pada data harian: 7 indikator). Indicators bekerja sebagai on/off switch: paling banyak satu bernilai 1 per baris; linear regression mempelajari baseline untuk kategori yang di-drop lalu menambahkan adjustment.
- **Fourier features** = pasangan kurva sin dan cos, satu pasang per frekuensi dalam musim, mulai dari frekuensi terpanjang (1×/tahun, 2×/tahun, 3×/tahun, ...). Cocok bila musim punya **banyak observasi** (musim tahunan pada data harian) karena hanya butuh beberapa fitur, bukan ratusan.
- Efisiensi: 4 pasang Fourier (8 fitur) cukup mengaproksimasi musim tahunan, dibanding 365 indikator harian → komputasi lebih cepat dan risiko overfitting lebih kecil.
- **Periodogram** mengukur kekuatan tiap frekuensi; nilai y = `(a**2 + b**2)/2` dengan `a`, `b` koefisien sin/cos pada frekuensi itu. Cara memilih orde Fourier: lihat di frekuensi mana periodogram "drops off". Contoh course: turun antara *Bimonthly (6)* dan *Monthly (12)* → pakai 10 pasang; frekuensi *Weekly (52)* diabaikan karena lebih baik dimodelkan dengan indicators.
- **`CalendarFourier(freq="A", order=n)`** membuat pasangan Fourier untuk musim tahunan; `DeterministicProcess(seasonal=True, additional_terms=[fourier])` menggabungkan indikator musiman dari frekuensi index + Fourier sebagai term tambahan.
- **Correlogram ↔ periodogram**: periodogram untuk memilih fitur Fourier, correlogram (lesson 4) untuk memilih lag.

### Contoh kode

```python
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess

tunnel = pd.read_csv("../input/ts-course-data/tunnel.csv", parse_dates=["Day"])
tunnel = tunnel.set_index("Day").to_period("D")
y = tunnel["NumVehicles"]

# --- Diagnostik: seasonal plot mingguan & tahunan, lalu periodogram ---
X = tunnel.copy()
X["day"] = X.index.dayofweek        # x-axis (freq) untuk musim mingguan
X["week"] = X.index.week            # period musim mingguan
X["dayofyear"] = X.index.dayofyear  # x-axis (freq) untuk musim tahunan
X["year"] = X.index.year            # period musim tahunan
# lalu: seasonal_plot(X, y="NumVehicles", period="week", freq="day")
#       seasonal_plot(X, y="NumVehicles", period="year", freq="dayofyear")

from scipy.signal import periodogram
freqs, spectrum = periodogram(y, fs=365, detrend="linear",
                              window="boxcar", scaling="spectrum")
# sumbu x log + tick 1,2,4,6,12,26,52,104 -> baca di frekuensi mana spektrum turun

# --- Fitur musiman: indikator mingguan + 10 pasang Fourier tahunan ---
fourier = CalendarFourier(freq="A", order=10)   # "A" = annual

dp = DeterministicProcess(
    index=tunnel.index,
    constant=True,                # intercept
    order=1,                      # trend linear
    seasonal=True,                # indikator mingguan (mengikuti freq index harian)
    additional_terms=[fourier],   # musim tahunan via Fourier
    drop=True,                    # hindari kolinier
)
X = dp.in_sample()

model = LinearRegression(fit_intercept=False).fit(X, y)
y_pred = pd.Series(model.predict(X), index=y.index)

X_fore = dp.out_of_sample(steps=90)             # forecast 90 hari
y_fore = pd.Series(model.predict(X_fore), index=X_fore.index)
```

Rumus manual Fourier (opsional, untuk memahami isi `CalendarFourier`):

```python
def fourier_features(index, freq, order):
    time = np.arange(len(index), dtype=np.float32)
    k = 2 * np.pi * (1 / freq) * time            # sudut per langkah waktu
    features = {}
    for i in range(1, order + 1):                # harmonik ke-1..order
        features[f"sin_{freq}_{i}"] = np.sin(i * k)
        features[f"cos_{freq}_{i}"] = np.cos(i * k)
    return pd.DataFrame(features, index=index)
```

### Kesalahan umum

- **Memakai indikator untuk musim beresolusi tinggi.** Musim tahunan pada data harian butuh 365 kolom; hasilnya lambat, rawan overfitting, dan banyak kolom nyaris konstan. Gunakan Fourier.
- **Lupa membuang satu indikator (dummy trap).** K indikator + intercept = kolinier sempurna. Tutorial membuang satu kategori (mis. Monday). `DeterministicProcess(drop=True)` menangani ini otomatis.
- **Memilih orde Fourier tanpa melihat periodogram.** Menebak orde terlalu rendah → musim tidak tertangkap; terlalu tinggi → fitur noise ikut dipelajari. Orde ditentukan dari titik di mana spektrum turun.

### Drill

1. Kapan memakai seasonal indicators dan kapan memakai Fourier features? Beri satu contoh masing-masing.
2. Berapa fitur yang dihasilkan 4 pasang Fourier, dan berapa yang dibutuhkan pendekatan indikator harian untuk musim tahunan?
3. Apa arti nilai y pada periodogram, dan bagaimana ia dipakai memilih `order`?
4. Pada `CalendarFourier(freq="A", order=10)`, apa arti `freq` dan `order`?
5. Kenapa salah satu indikator harus di-drop saat memakai indikator musiman bersama intercept?

---

## 4. Time Series as Features

### Poin inti

- **Serial dependence**: nilai saat ini bisa diprediksi dari nilai-nilai sebelumnya (bukan dari *kapan* terjadinya). Beberapa series hanya menampakkan struktur ini di **lag plot**, tidak di time plot.
- **Cycles** = pola growth/decay yang muncul dari ketergantungan nilai sekarang pada nilai sebelumnya, tanpa terikat time step. Cycles khas sistem yang mempengaruhi dirinya sendiri (ekonomi, epidemi, populasi hewan). Bedanya dengan seasonality: cycles **tidak harus time dependent**, jadi jauh lebih ireguler (puncak flu bisa lebih awal/lebih lambat, lebih besar/kecil).
- **Lagging** = menggeser nilai ke depan (atau menggeser waktu di index ke belakang) sehingga nilai masa lalu tampak *contemporaneous* dengan target di baris yang sama. `y.shift(1)`, `y.shift(2)`.
- **Lag plot** = nilai series vs lag-nya (hubungan linear rapat → lag feature berguna); ukuran kuantitatifnya adalah **autocorrelation** = korelasi series dengan salah satu lag-nya (US Unemployment: 0.99 di lag 1, 0.98 di lag 2).
- **Partial autocorrelation** = korelasi sebuah lag *setelah* memperhitungkan semua lag sebelumnya — yaitu kontribusi "baru" lag tersebut. Ini dasar memilih lag: pada US Unemployment lag 1–6 keluar dari interval "no correlation" (95%), lag 11 dianggap false positive. Plot-nya disebut **correlogram** (padanan periodogram untuk lag).
- Autocorrelation dan partial autocorrelation hanya mengukur **dependensi linear**. Selalu periksa lag plot (atau ukuran dependensi umum seperti *mutual information*) karena series nyata sering non-linear (contoh: Sunspots).
- **Leading indicators**: series lain yang bergerak lebih dulu dan bisa memberi "early warning" (mis. volume pencarian Google untuk istilah flu naik beberapa minggu sebelum kunjungan dokter naik). Menambahkan lag dari leading indicator membuat model bisa mengantisipasi lonjakan.
- Keterbatasan penting: dengan lag features, horizon forecast **terbatas pada ketersediaan lag**. Lag 1 hari Senin tidak bisa memforecast Rabu karena nilai Selasa belum ada. Solusinya dibahas di lesson 6.

### Contoh kode

```python
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from statsmodels.graphics.tsaplots import plot_pacf

flu = pd.read_csv("../input/ts-course-data/flu-trends.csv")
flu.set_index(pd.PeriodIndex(flu.Week, freq="W"), inplace=True)
flu.drop("Week", axis=1, inplace=True)
y = flu.FluVisits.copy()

# --- Diagnostik: lag plot (manual) + partial autocorrelation ---
plot_pacf(flu.FluVisits, lags=12)        # pilih lag yang keluar dari pita 95%

# --- Fitur: 4 lag pertama ---
def make_lags(ts, lags):
    return pd.concat({f"y_lag_{i}": ts.shift(i) for i in range(1, lags + 1)}, axis=1)

X = make_lags(flu.FluVisits, lags=4).fillna(0.0)   # NaN di awal diisi 0.0

# --- Split kronologis: JANGAN shuffle ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=60, shuffle=False)

model = LinearRegression()               # fit_intercept=True: tanpa DeterministicProcess
model.fit(X_train, y_train)
y_fore = pd.Series(model.predict(X_test), index=y_test.index)

# --- Versi dengan leading indicators (Google Trends) ---
terms = ["FluContagious", "FluCough", "FluFever", "InfluenzaA",
         "TreatFlu", "IHaveTheFlu", "OverTheCounterFlu", "HowLongFlu"]
X0 = make_lags(flu[terms], lags=3)
X0.columns = [" ".join(c).strip() for c in X0.columns]   # rapikan nama kolom MultiIndex
X1 = make_lags(flu["FluVisits"], lags=4)
X = pd.concat([X0, X1], axis=1).fillna(0.0)
```

### Kesalahan umum

- **`train_test_split` dengan `shuffle=True` (default).** Ini merusak urutan waktu — model "melihat masa depan". Selalu `shuffle=False` dan split berdasarkan waktu (test = periode terakhir).
- **Memasukkan semua lag yang autokorelasinya tinggi.** Lag 2 sering hanya "sisa" informasi lag 1. Pakai **partial** autocorrelation untuk memilih lag yang menambah informasi baru, dan cek lag plot untuk dependensi non-linear.
- **Mengisi NaN hasil `shift` dengan cara yang membocorkan informasi.** Backfill dari nilai masa depan = kebocoran. Isi `0.0`, atau drop barisnya, atau gunakan nilai yang memang diketahui pada saat itu.

### Drill

1. Apa perbedaan cycles dan seasonality, dan kenapa perbedaan itu penting untuk pilihan fitur?
2. Jelaskan perbedaan autocorrelation dan partial autocorrelation dengan kata sendiri.
3. Series mingguan, kamu punya `Lag_1`..`Lag_4`. Bisa berapa langkah ke depan kamu memforecast tanpa strategi tambahan? Kenapa?
4. Kenapa `shuffle=False` wajib pada `train_test_split` untuk data time series?
5. Apa itu leading indicator, dan bagaimana cara memasukkannya sebagai fitur?

---

## 5. Hybrid Models

### Poin inti

- **Dekomposisi aditif**: `series = trend + seasons + cycles + error`. Setiap suku disebut **component**; sisanya adalah error yang tak terprediksi.
- **Residuals** = target − prediksi. Residual memuat semua yang belum dipelajari model dari fitur yang diberikan. Proses belajar komponen bisa dibayangkan iteratif: pelajari trend → kurangkan → pelajari musim dari residual → kurangkan → pelajari cycles → sisanya error.
- **Motivasi hybrid**: linear regression unggul **mengekstrapolasi trend** tapi tidak bisa mempelajari interaksi; XGBoost unggul mempelajari **interaksi** tapi tidak bisa mengekstrapolasi trend. Digabung, kelemahan saling menutupi.
- **Prosedur hybrid (boosted)**: (1) latih model 1, prediksi; (2) latih model 2 pada **residual** `y_train - y_pred_1`; (3) jumlahkan `y_pred = y_pred_1 + y_pred_2`. Biasanya dua model memakai feature set berbeda (kalau model 1 sudah menangani trend, model 2 tidak perlu fitur trend).
- Dua cara algoritma regresi bekerja: **feature-transforming** (linear regression, neural net — mempelajari fungsi dari fitur) vs **target-transforming** (decision tree, nearest neighbors — mengelompokkan lalu merata-ratakan target). Feature transformer bisa **extrapolate**; target transformer selalu terkurung dalam rentang nilai training.
- Konsekuensi: **decision tree, random forest, dan GBDT (XGBoost) tidak bisa mengekstrapolasi trend** — dengan time dummy yang terus naik, tree hanya mengulang nilai prediksi dari langkah terakhir training.
- Dua gaya hybrid: **boosted** (model 2 dilatih pada residual — sama seperti mekanisme gradient boosting) dan **stacked** (prediksi model 1 dipakai sebagai *feature* model 2; cocok untuk neural net yang merupakan feature transformer). Praktik umumnya satu model sederhana (biasanya linear) sebagai "helper" diikuti learner non-linear yang kuat; lebih dari dua model biasanya tidak menambah manfaat signifikan.
- Data untuk XGBoost harus **long format**: `stack()` memindahkan label kolom (Industries) menjadi label baris; kategori diubah ke angka dengan `factorize()`; musim tahunan diambil dari `X.index.month`.

### Contoh kode

```python
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from statsmodels.tsa.deterministic import DeterministicProcess
from xgboost import XGBRegressor

retail = pd.read_csv("../input/ts-course-data/us-retail-sales.csv",
                     usecols=["Month", "BuildingMaterials", "FoodAndBeverage"],
                     parse_dates=["Month"], index_col="Month").to_period("D")
y = pd.concat({"Sales": retail}, names=[None, "Industries"], axis=1)

# --- 1. Model 1: trend kuadratik (linear regression) ---
dp = DeterministicProcess(index=y.index, constant=True, order=2, drop=True)
X = dp.in_sample()
idx_train, idx_test = train_test_split(y.index, test_size=12 * 4, shuffle=False)

model = LinearRegression(fit_intercept=False).fit(X.loc[idx_train], y.loc[idx_train])
y_fit  = pd.DataFrame(model.predict(X.loc[idx_train]), index=idx_train,  columns=y.columns)
y_pred = pd.DataFrame(model.predict(X.loc[idx_test]),  index=idx_test,   columns=y.columns)

# --- 2. Siapkan data long format untuk XGBoost ---
X = y.stack()                          # wide -> long: Industries pindah ke baris
y_long = X.pop("Sales")                # ambil target, sisakan fitur kategori
X = X.reset_index("Industries")
for col in X.select_dtypes(["object", "category"]):
    X[col], _ = X[col].factorize()     # label encoding kategori
X["Month"] = X.index.month             # musim tahunan sebagai fitur

X_train, X_test = X.loc[idx_train, :], X.loc[idx_test, :]
y_train = y_long.loc[idx_train]

# --- 3. Model 2 dilatih pada residual, lalu dijumlahkan ---
y_fit_long  = y_fit.stack().squeeze()
y_pred_long = y_pred.stack().squeeze()
y_resid = y_train - y_fit_long         # yang belum dipelajari model 1

xgb = XGBRegressor().fit(X_train, y_resid)
y_fit_boosted  = xgb.predict(X_train) + y_fit_long
y_pred_boosted = xgb.predict(X_test)  + y_pred_long
```

### Kesalahan umum

- **Mengharapkan XGBoost memperbaiki trend yang salah.** XGBoost hanya memodelkan residual; kalau trend linear regression bias (mis. `BuildingMaterials` pada contoh), hybrid tetap bias. Kualitas trend model 1 adalah batas atas kualitas hybrid.
- **Memberi time dummy ke XGBoost dan berharap ekstrapolasi.** Tree tidak bisa keluar dari rentang training — prediksi masa depan akan flat di nilai terakhir. Trend harus ditangani model linear (atau fitur ditransformasi agar trend hilang).
- **Lupa menyelaraskan bentuk wide vs long.** `y_train` (wide, MultiIndex kolom) dikurangi `y_fit` yang sudah di-`stack()` akan menghasilkan NaN/mismatch. Konsistenkan format sebelum aritmetika residual.

### Drill

1. Tulis rumus dekomposisi aditif dan jelaskan apa itu residual.
2. Kenapa XGBoost tidak bisa mengekstrapolasi trend, sedangkan linear regression bisa? Jelaskan lewat perbedaan feature-transforming vs target-transforming.
3. Tulis tiga langkah prosedur hybrid boosted dalam bentuk pseudo-code.
4. Apa beda **boosted hybrid** dan **stacked hybrid**, dan kapan memilih masing-masing?
5. Kenapa data harus diubah ke long format (`stack()`) sebelum masuk XGBoost pada contoh US Retail Sales?

---

## 6. Forecasting With Machine Learning

### Poin inti

- Begitu lag features dipakai, forecast butuh nilai lag yang **sudah diketahui** pada saat prediksi. Karena itu perlu mendefinisikan tugas forecasting secara eksplisit.
- **Forecast origin** = waktu saat forecast dibuat (praktisnya: waktu terakhir yang datanya tersedia). Semua data sampai origin boleh dipakai untuk fitur.
- **Forecast horizon** = periode yang diramalkan (mis. 8 langkah ke depan). **Lead time** (latency) = jarak origin → awal horizon (mis. 2 langkah karena keterlambatan pengumpulan data). Origin menjelaskan fitur; horizon menjelaskan target.
- **Persiapan data**: satu baris = satu forecast. Index baris = waktu pertama dalam horizon, dan nilai seluruh horizon diletakkan di kolom-kolom baris itu → model menghasilkan **multiple output**.
- Empat strategi multistep:
  - **Multioutput model** — satu model yang secara alami menghasilkan banyak output (linear regression, neural net). Sederhana dan efisien, tapi tidak semua algoritma bisa (XGBoost tidak).
  - **Direct** — satu model terpisah untuk tiap langkah horizon. Tiap langkah adalah masalah berbeda sehingga bisa lebih akurat, tapi biaya training banyak model. Untuk XGBoost: bungkus dengan `MultiOutputRegressor`.
  - **Recursive** — satu model 1-step, prediksinya diumpan balik sebagai lag untuk langkah berikutnya. Hanya satu model, tapi error terakumulasi sehingga horizon panjang tidak akurat.
  - **DirRec** — gabungan: satu model per langkah **plus** forecast langkah sebelumnya sebagai lag baru; tiap model mendapat set lag yang selalu ter-update. Menangkap serial dependence lebih baik daripada Direct, tapi tetap rawan propagasi error. Di scikit-learn: `RegressorChain`.
- Helper penting: `make_lags(ts, lags, lead_time=1)` menghasilkan lag mulai `lead_time` sampai `lags + lead_time - 1`; `make_multistep_target(ts, steps)` menghasilkan `y_step_1..y_step_n` via `shift(-i)`. Setelah `shift`, index fitur dan target tidak lagi cocok → `y, X = y.align(X, join='inner', axis=0)`.
- Evaluasi memakai RMSE pada train dan test; pada contoh Flu Trends, XGBoost overfit di train tapi menangkap dinamika musim flu lebih baik di test (dan akan membaik dengan hyperparameter tuning).
- Untuk Recursive, strategi harus ditulis sendiri (tidak ada wrapper siap pakai di scikit-learn).

### Contoh kode

```python
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

flu = pd.read_csv("../input/ts-course-data/flu-trends.csv")
flu.set_index(pd.PeriodIndex(flu.Week, freq="W"), inplace=True)
flu.drop("Week", axis=1, inplace=True)
y_raw = flu.FluVisits.copy()

# --- Fitur: lag 1..4 (lead_time default 1) ---
def make_lags(ts, lags, lead_time=1):
    return pd.concat({f"y_lag_{i}": ts.shift(i)
                      for i in range(lead_time, lags + lead_time)}, axis=1)

# --- Target: 8 langkah ke depan dalam satu baris ---
def make_multistep_target(ts, steps):
    return pd.concat({f"y_step_{i + 1}": ts.shift(-i) for i in range(steps)}, axis=1)

X = make_lags(y_raw, lags=4).fillna(0.0)
y = make_multistep_target(y_raw, steps=8).dropna()
y, X = y.align(X, join="inner", axis=0)          # hanya waktu yang punya fitur & target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False)

# --- Strategi Multioutput ---
model = LinearRegression().fit(X_train, y_train)
y_pred = pd.DataFrame(model.predict(X_test), index=X_test.index, columns=y.columns)
# RMSE gabungan untuk seluruh horizon (scikit-learn >= 1.4: root_mean_squared_error)
print("Test RMSE:", mean_squared_error(y_test, y_pred) ** 0.5)

# --- Strategi Direct dengan XGBoost ---
direct = MultiOutputRegressor(XGBRegressor()).fit(X_train, y_train)
y_pred_direct = pd.DataFrame(direct.predict(X_test), index=X_test.index, columns=y.columns)
```

### Kesalahan umum

- **Melupakan lead time saat membuat lag.** Kalau data baru tersedia 2 langkah setelah terjadi, `Lag_1` dan `Lag_2` tidak tersedia di origin. Set `lead_time=2` agar lag termuda yang dipakai adalah `y_lag_2`.
- **Membiarkan NaN hasil `shift(-i)` pada target.** Baris terakhir tidak punya masa depan → `dropna()` pada target, lalu `align(join='inner')` agar fitur dan target konsisten.
- **Memakai Recursive untuk horizon panjang tanpa memeriksa propagasi error.** Setiap langkah memakai prediksi langkah sebelumnya; error menumpuk. Bandingkan dengan Direct/DirRec pada validation set sebelum memilih.

### Drill

1. Definisikan forecast origin, forecast horizon, dan lead time untuk kasus: hari ini Senin, data penjualan terakhir yang bersih adalah Jumat, dan kamu harus meramal 7 hari mulai Rabu (minggu berikutnya).
2. Kenapa satu baris dataset forecasting multi-step berisi beberapa kolom target?
3. Sebutkan kelebihan dan kekurangan strategi Recursive.
4. Algoritma mana yang tidak bisa memakai strategi Multioutput secara native, dan wrapper apa yang dipakai untuk menerapkan Direct padanya?
5. Apa fungsi `align(X, join='inner', axis=0)` setelah `make_lags` dan `make_multistep_target`?

---

## Kunci Jawaban

### Lesson 1 — Linear Regression With Time Series

1. **Time dependence**: nilai dapat diprediksi dari *kapan* ia terjadi → dimodelkan dengan **time-step features** (time dummy, `np.arange`). **Serial dependence**: nilai dapat diprediksi dari nilai-nilai sebelumnya → dimodelkan dengan **lag features** (`shift`).
2. `df['Time'] = np.arange(len(df.index))`
3. scikit-learn mengharapkan `X` berbentuk 2D `(n_samples, n_features)`. Series 1D akan ditolak/diinterpretasi salah; `df[['Lag_1']]` menghasilkan DataFrame satu kolom.
4. Tidak ada hubungan linear yang jelas pada lag 1 → `Lag_1` kemungkinan tidak informatif sebagai fitur (perlu cek lag lain, transformasi, atau dependensi non-linear).
5. Dua baris NaN di awal. Opsi: **isi** (0.0 atau backfill dengan nilai diketahui pertama) atau **drop** baris tersebut beserta target pada tanggal yang sama (yang dipakai tutorial).

### Lesson 2 — Trend

1. Karena data harian memiliki **musim tahunan**; window harus lebih panjang dari satu periode musim agar musim tersaring dan hanya perubahan jangka panjang yang tersisa. Window 7/30 hanya akan menyisakan pola mingguan/bulanan.
2. `center=True` menempatkan nilai rata-rata di tengah window sehingga garis tidak bergeser ke belakang; `min_periods=183` (± setengah window) membuat ujung series tetap terhitung tanpa memaksa window penuh.
3. `DeterministicProcess(index=tunnel.index, constant=True, order=2, drop=True)`
4. `in_sample()` membuat fitur untuk tanggal yang ada di `index` (data training); `out_of_sample(steps=30)` membuat fitur untuk 30 langkah **setelah** index berakhir → dipakai untuk forecast.
5. Karena `constant=True` sudah membuat kolom konstanta untuk intercept; menambah intercept lagi membuat dua kolom identik (kolinier) sehingga solusi OLS tidak unik/koefisien tidak stabil. Gejala: koefisien ekstrem/berlawanan tanda, warning matriks singular, prediksi tetap tapi interpretasi rusak.

### Lesson 3 — Seasonality

1. **Indicators** untuk musim dengan sedikit observasi per siklus — mis. musim mingguan (7 kategori) pada data harian. **Fourier** untuk musim dengan banyak observasi — mis. musim tahunan pada data harian (365 nilai) yang diringkas 8–20 fitur.
2. 4 pasang Fourier = **8 fitur** (sin+cos per pasang); pendekatan indikator harian untuk musim tahunan butuh **± 365 fitur** (satu per hari, dikurangi satu jika di-drop).
3. Nilai y = `(a**2 + b**2)/2`, yaitu kekuatan (variance) frekuensi tersebut dengan `a`,`b` koefisien sin/cos. Pilih `order` sampai titik frekuensi di mana spektrum **drops off** (contoh course: 10 pasang karena turun antara Bimonthly dan Monthly).
4. `freq` = periode musim dalam alias frekuensi pandas (`"A"` annual, juga `"M"`, `"W"`, `"D"`, `"H"`); `order` = jumlah pasangan sin/cos (harmonik) yang dibuat.
5. K indikator + intercept menimbulkan **kolinearitas sempurna** (jumlah semua indikator = 1 = kolom intercept). Membuang satu kategori membuat baseline diserap intercept dan koefisien lain teridentifikasi.

### Lesson 4 — Time Series as Features

1. Seasonality terikat kalender dan berulang teratur; cycles bergantung pada nilai masa lalu, tidak terikat tanggal, dan jauh lebih ireguler. Karena itu seasonality dimodelkan fitur waktu (indikator/Fourier), sedangkan cycles dimodelkan **lag features**.
2. **Autocorrelation** = korelasi series dengan lag-nya (total, termasuk informasi yang diwarisi lag sebelumnya). **Partial autocorrelation** = korelasi lag tersebut setelah memperhitungkan semua lag sebelumnya — kontribusi informasi "baru".
3. **Satu langkah**. Untuk memprediksi langkah ke-2 dibutuhkan nilai langkah ke-1 yang belum terjadi; lag 4 pun tidak menutupi kebutuhan nilai masa depan tersebut. Perlu strategi multistep (lesson 6).
4. Karena time series berurutan; shuffling membuat data masa depan masuk ke training (leakage) dan metrik test jadi terlalu optimistis.
5. Series lain yang bergerak **lebih dulu** daripada target sehingga memberi peringatan dini (mis. volume pencarian "FluCough" sebelum kunjungan dokter). Cara: buat lag dari kolom leading indicator (`make_lags(flu[terms], lags=3)`), gabungkan dengan lag target, isi NaN, lalu latih model.

### Lesson 5 — Hybrid Models

1. `series = trend + seasons + cycles + error`. **Residual** = target − prediksi model = bagian target yang belum dipelajari model dari fitur yang diberikan.
2. **Feature-transforming** (linear regression, NN) mempelajari fungsi matematis dari fitur sehingga bisa menghasilkan nilai di luar rentang training (garis trend berlanjut). **Target-transforming** (tree, kNN) memprediksi dengan merata-ratakan target dalam grup; rata-rata selalu berada dalam rentang nilai training, jadi dengan time dummy yang naik terus pun prediksi tree berhenti di level terakhir.
3. (1) `model_1.fit(X_train_1, y_train)` lalu `y_pred_1 = model_1.predict(X_train_1)`; (2) `model_2.fit(X_train_2, y_train - y_pred_1)` lalu `y_pred_2 = model_2.predict(X_train_2)`; (3) `y_pred = y_pred_1 + y_pred_2`. Model 2 dilatih pada residual, bukan pada target asli.
4. **Boosted**: model 2 dilatih pada residual (mekanisme seperti gradient boosting) — umum untuk kombinasi linear + GBDT. **Stacked**: prediksi model 1 menjadi *feature* model 2 (stacking) — dipakai untuk feature transformer seperti neural net yang bisa memanfaatkan prediksi sebagai input.
5. Karena XGBoost tidak mendukung multi-output regression pada data wide (satu kolom per series). `stack()` mengubah wide → long sehingga satu model bisa belajar banyak series sekaligus, dengan series dibedakan lewat fitur kategori hasil `factorize()`.

### Lesson 6 — Forecasting With Machine Learning

1. **Origin** = Jumat (waktu data terakhir tersedia; hari ini Senin tidak menambah data). **Lead time** = 5 langkah dari origin ke awal horizon (Sabtu, Minggu, Senin, Selasa, Rabu). **Horizon** = 7 langkah (Rabu sampai Selasa berikutnya). Karena baris data di-index pada awal horizon, lag termuda yang tersedia di origin adalah `y_lag_5`; lag 1–4 akan menunjuk ke tanggal setelah origin sehingga belum diketahui.
2. Karena satu baris merepresentasikan satu *forecast call* yang harus menghasilkan nilai untuk seluruh horizon; tiap langkah horizon menjadi satu kolom target (`y_step_1..y_step_n`) sehingga model belajar multi-output.
3. Kelebihan: hanya satu model dilatih (murah). Kekurangan: prediksi diumpan balik sebagai lag, sehingga error terakumulasi dan akurasi menurun untuk horizon panjang.
4. **XGBoost** (dan umumnya GBDT) tidak mendukung multi-output native. Direct diterapkan dengan `sklearn.multioutput.MultiOutputRegressor`; untuk DirRec gunakan `RegressorChain`.
5. Menyelaraskan index fitur dan target setelah `shift` ke dua arah — hanya menyisakan waktu yang memiliki fitur **dan** target lengkap, sehingga tidak ada NaN atau baris tak berpasangan saat training.

---

## Ringkasan 1 paragraf

Course Time Series menempatkan forecasting sebagai masalah regresi dengan feature engineering yang disiplin. Lesson 1 memperkenalkan dua fitur dasar — time dummy (time dependence) dan lag (serial dependence) — beserta cara membacanya lewat time plot dan lag plot. Lesson 2 menambahkan trend: didiagnosis dengan moving average plot (window lebih panjang dari satu musim) dan dimodelkan lewat `DeterministicProcess` ber-`order` polinomial, dengan `in_sample`/`out_of_sample` sebagai jalur forecast. Lesson 3 menangani musim: indikator one-hot untuk musim pendek, Fourier (`CalendarFourier`) untuk musim panjang, dengan orde dipilih dari periodogram. Lesson 4 beralih ke serial dependence dan cycles: lag plot, autocorrelation, partial autocorrelation (correlogram) untuk memilih lag, serta leading indicators sebagai early warning. Lesson 5 menyusun hybrid: dekomposisi `trend + seasons + cycles + error`, model linear menangkap trend yang bisa diekstrapolasi, lalu XGBoost dilatih pada residual untuk menangkap interaksi yang tidak bisa dipelajari model linear. Lesson 6 menutup dengan formalisasi tugas forecasting (origin, horizon, lead time) dan empat strategi multistep — Multioutput, Direct, Recursive, DirRec — sehingga model yang sudah dibangun bisa menghasilkan ramalan banyak langkah yang valid secara metodologis. Benang merahnya: pahami komponen series, pilih fitur yang tepat untuk tiap komponen, dan hormati struktur waktu saat menyiapkan data serta mengevaluasi model.

---

## Sumber

- Kaggle Learn — Time Series (halaman course): https://www.kaggle.com/learn/time-series
- Lesson 1 — Linear Regression With Time Series: https://www.kaggle.com/code/ryanholbrook/linear-regression-with-time-series
- Lesson 2 — Trend: https://www.kaggle.com/code/ryanholbrook/trend
- Lesson 3 — Seasonality: https://www.kaggle.com/code/ryanholbrook/seasonality
- Lesson 4 — Time Series as Features: https://www.kaggle.com/code/ryanholbrook/time-series-as-features
- Lesson 5 — Hybrid Models: https://www.kaggle.com/code/ryanholbrook/hybrid-models
- Lesson 6 — Forecasting With Machine Learning: https://www.kaggle.com/code/ryanholbrook/forecasting-with-machine-learning
- Dataset course: `ts-course-data` (Kaggle dataset `ryanholbrook/ts-course-data`)
- Kompetisi latihan: Store Sales — Time Series Forecasting, https://www.kaggle.com/c/store-sales-time-series-forecasting
- Forum diskusi course: https://www.kaggle.com/learn/time-series/discussion

*Catatan teknis: isi lesson diambil dari notebook tutorial resmi (diakses 27 November 2026). Beberapa API yang dipakai tutorial sudah bergeser versi — `plt.style.use("seaborn-whitegrid")` (gaya lama Matplotlib; di Matplotlib 3.6+ harus `seaborn-v0_8-whitegrid`, sudah diuji gagal di 3.11) dan `mean_squared_error(..., squared=False)` (deprecated di scikit-learn ≥ 1.4, dihapus di 1.6 — gunakan `root_mean_squared_error`). Adapun `X.index.week` pada lesson 3 tetap valid karena index-nya `PeriodIndex`; pada `DatetimeIndex` properti itu sudah dihapus di pandas 2.0 dan diganti `index.isocalendar().week`. Ini catatan kompatibilitas, bukan bagian dari materi asli.*
