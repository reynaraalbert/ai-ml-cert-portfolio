# Kaggle Learn — Intro to AI Ethics

**Slug:** `intro-to-ai-ethics` · **Penyusun materi:** Var Shankar & Alexis Cook · **Estimasi waktu (menurut halaman kursus):** 4 jam · **Prasyarat:** tidak ada (tidak diasumsikan latar belakang pemrograman)

Kursus ini adalah kursus **bacaan + latihan (exercise)**: tiap lesson berupa tutorial notebook, lalu satu exercise notebook berisi kode dan pertanyaan. Materi di bawah merangkum isi tutorial resmi per lesson, ditambah glosarium, studi kasus, dan drill untuk self-check.

**Peta kursus (5 lesson):**

1. Introduction to AI Ethics
2. Human-Centered Design for AI
3. Identifying Bias in AI
4. AI Fairness
5. Model Cards

---

## 1. Introduction to AI Ethics

### Poin inti

- AI dipakai untuk hal-hal berdampak langsung: keputusan kartu kredit, pengawasan video di bandara, dan menginformasikan operasi militer. Teknologi ini bisa **membantu atau melukai** orang yang dilayaninya.
- Langkah inti kursus: memakai "lensa etika" untuk **mengidentifikasi potensi bahaya**, lalu mendesain sistem yang mengurangi bahaya itu — atau memutuskan **tidak membangunnya** sama sekali.
- Kursus tidak punya prasyarat dan tidak mengasumsikan kemampuan coding; di exercise peserta tetap menjalankan kode untuk melatih dan menyelidiki model.
- Empat topik yang dibahas berurutan: **human-centered design** (memastikan sistem melayani kebutuhan orang), **bias** (bagaimana sistem bisa belajar mendiskriminasi kelompok), **fairness** (mengukur seberapa besar bias itu), dan **model cards** (kerangka akuntabilitas publik untuk model).
- Kursus ini menekankan sisi praktis: kalau kamu sudah berada di tim yang membangun produk AI dan bingung mulai dari mana, materinya memberi saran yang bisa langsung dieksekusi.
- Sikap yang ditanamkan: *"Ethics is a conversation"* — etika bukan checklist yang selesai, melainkan percakapan yang terus berkembang seiring berkembangnya bidang AI.
- Kursus ini **tidak** membahas dampak AI terhadap kesenjangan digital kaya–miskin, terhadap lapangan kerja, dan terhadap kapabilitas pemerintah otoriter. Keterbatasan cakupan ini disebut eksplisit agar peserta tidak menganggap kursus ini menyeluruh.

### Studi kasus / contoh nyata

- **Kartu kredit, bandara, militer:** tiga contoh pemakaian AI yang disebut di tutorial sebagai ilustrasi bahwa keputusan model menyentuh akses ekonomi, kebebasan bergerak, dan keselamatan nyawa — bukan sekadar akurasi di atas kertas.
- **Kursus sebagai titik awal, bukan akhir:** tutorial menutup dengan anjuran melanjutkan ke kursus *Machine Learning Explainability* dan memantau diskusi etika yang terus berubah; ini contoh nyata bahwa materi etika AI cepat kedaluwarsa dan perlu diperbarui.

### Istilah kunci

- **AI ethics** — kajian tentang dampak baik/buruk sistem AI terhadap orang dan masyarakat, serta cara mendesain agar dampak buruknya berkurang.
- **Ethical lens (lensa etika)** — kebiasaan memeriksa sebuah sistem dari sisi "siapa yang diuntungkan dan siapa yang dirugikan", bukan hanya dari sisi performa teknis.
- **Human-centered design (HCD)** — pendekatan mendesain sistem dengan melibatkan orang di setiap tahap, diperkenalkan di lesson 2.
- **Bias** — dalam kursus ini: konsekuensi negatif yang tidak diinginkan dari aplikasi ML, terutama bila dampaknya timpang ke kelompok tertentu.
- **Model card** — dokumen ringkas berisi informasi kunci sebuah model ML, dibahas di lesson 5.

### Drill

1. Menurut tutorial, apa dua kemungkinan keputusan yang sah setelah tim menilai potensi bahaya sebuah sistem AI?
   a. Menambah data / menambah GPU
   b. Mendesain ulang untuk mengurangi bahaya / tidak membangunnya
   c. Menurunkan ambang akurasi / mengganti vendor
   d. Menambah dokumentasi / menambah jumlah pengguna

2. Pernyataan mana yang paling sesuai dengan sikap kursus terhadap etika AI?
   a. Etika AI adalah checklist kepatuhan yang selesai setelah audit
   b. Etika AI hanya urusan tim legal
   c. Etika AI adalah percakapan yang terus berlangsung dan ikut berubah bersama bidangnya
   d. Etika AI hanya relevan untuk model generatif

3. Topik mana yang **secara eksplisit** dinyatakan di luar cakupan kursus ini?
   a. Bias pada data latih
   b. Model cards
   c. Dampak AI terhadap lapangan kerja
   d. Pengukuran fairness

4. Urutan empat topik utama kursus adalah…
   a. Model cards → Fairness → Bias → HCD
   b. HCD → Bias → Fairness → Model cards
   c. Bias → Model cards → HCD → Fairness
   d. Fairness → HCD → Model cards → Bias

5. Apa prasyarat kursus ini menurut halaman resminya?
   a. Python tingkat menengah
   b. Statistika dasar
   c. Tidak ada prasyarat
   d. Pengalaman membangun model produksi

---

## 2. Human-Centered Design for AI

### Poin inti

- **Human-centered design (HCD)** adalah pendekatan mendesain sistem yang melayani kebutuhan orang. HCD melibatkan orang di **setiap langkah** proses desain, dan sebaiknya dimulai sedini mungkin — idealnya sejak tim mulai mempertimbangkan kemungkinan membangun sistem AI.
- **Langkah 1 — Pahami kebutuhan orang untuk mendefinisikan masalah.** Amati orang memakai alat yang ada, lakukan wawancara, bentuk focus group, baca umpan balik pengguna. Seluruh tim (termasuk data scientist dan engineer) harus terlibat agar semua orang memahami siapa yang dilayani; tim juga harus beragam latar belakangnya (ras, gender, dan lain-lain).
- **Langkah 2 — Tanya apakah AI memberi nilai tambah.** Empat pertanyaan penyaring: apakah hasil yang dituju disepakati sebagai hal baik? Apakah solusi non-AI (mis. rule-based, lebih mudah dibuat/diaudit/dirawat) jauh lebih lemah? Apakah tugasnya membosankan, repetitif, atau sulit dijaga konsentrasinya? Apakah AI sudah terbukti lebih baik pada kasus serupa? **Jika ada satu saja jawaban "tidak", solusi AI mungkin tidak perlu atau tidak tepat.**
- **Langkah 3 — Pertimbangkan potensi bahaya** di sepanjang pipeline: pengumpulan dan pelabelan data, pelatihan model, hingga deployment. Pertimbangkan dampak pada pengguna dan masyarakat; libatkan tim privasi (mis. *differential privacy*, *federated learning*). Jika bahaya diperkirakan melebihi manfaat, **jangan bangun sistemnya**.
- **Langkah 4 — Prototipe, mulai dari solusi non-AI.** Prototipe non-AI lebih cepat, murah, dan mudah; memberi informasi awal soal ekspektasi pengguna. UI harus memudahkan orang memahami cara kerja sistem, mengubah pengaturan, dan memberi umpan balik. Pemberi umpan balik harus beragam dan memahami serta menyetujui apa yang mereka bantu.
- **Langkah 5 — Beri jalan bagi orang untuk menantang sistem.** Pengguna harus bisa menantang rekomendasi atau *opt out* dengan mudah. Sediakan mekanisme menerima, memantau, dan menindaklanjuti tantangan — misalnya meminta penjelasan, mengubah input, mematikan fitur tertentu, atau menghubungi tim produk.
- **Langkah 6 — Bangun ukuran keselamatan.** Keselamatan hanya tercapai lewat evaluasi dan pengujian yang ekstensif dan berkelanjutan: pantau performa, ketercapaian manfaat, pengurangan bahaya, metrik fairness, dan perubahan cara pakai nyata. Unsur kunci: **human oversight** — *red team* internal, pemantauan pasca-rilis, sistem yang cepat memberi sinyal ke manusia saat menghadapi kasus sulit, dan kanal pelaporan bagi pengguna.
- HCD bersifat kontekstual: apa arti HCD bagi sebuah tim bergantung pada industri, sumber daya, organisasi, dan orang yang hendak dilayani.

### Studi kasus / contoh nyata

- **Dosis imunosupresan pasca-transplantasi hati (langkah 1):** perusahaan mengamati dokter dan perawat sepanjang proses transplantasi, mewawancarai mereka soal penentuan dosis yang masih bergantung pada pedoman tertulis + penilaian manusia, membagikan klip wawancara ke seluruh tim, menelaah studi riset, dan membentuk focus group mantan pasien beserta keluarganya — lalu seluruh tim ikut sesi brainstorming bebas.
- **Penilaian esai otomatis (langkah 3, keputusan berhenti):** perusahaan edukasi online ingin AI memberi skor esai siswa. Komite peninjau bahaya merekomendasikan **agar sistem tidak dibangun**, karena risiko: model menyerap dan memperbesar bias bahasa dari data latih (merugikan kelompok pengguna pola bahasa tertentu), mendorong siswa "mengakali" algoritma alih-alih memperbaiki tulisan, serta menggeser peran pendidik ke peran teknisi.

### Istilah kunci

- **Human-centered design (HCD)** — pendekatan desain yang menempatkan kebutuhan, konteks, dan keterlibatan manusia di pusat seluruh proses.
- **Rule-based system** — solusi berbasis aturan eksplisit; lebih mudah dibuat, diaudit, dan dirawat, sehingga sering menjadi pembanding sebelum memilih AI.
- **Differential privacy** — teknik privasi yang menambahkan derau terkontrol agar kesimpulan statistik tetap berguna tanpa membocorkan data individu.
- **Federated learning** — pelatihan model terdistribusi di perangkat/organisasi tanpa memusatkan data mentah.
- **Red team** — tim internal yang berperan sebagai aktor jahat untuk memancing perilaku menyimpang sistem; temuannya dipakai memperkuat sistem lewat *adversarial training* (melatih model dengan input yang sengaja dirancang menipunya).

### Drill

1. Sebuah organisasi di Kamboja ingin mengurangi sampah plastik di sistem Sungai Mekong. Menurut HCD, langkah awal yang tepat adalah…
   a. Mengumpulkan citra satelit dan melatih model deteksi sampah
   b. Mengamati, mewawancarai, dan membentuk focus group dengan orang yang saat ini sudah menangani masalah tersebut
   c. Membeli API visi komputer komersial
   d. Menetapkan KPI pengurangan sampah 30% dalam setahun

2. Tim ingin membuat alat bantu patologis mendeteksi kanker payudara dari citra slide jaringan secara real-time. Sistem berbasis aturan sudah dicoba dan tidak mampu memberi nilai pada peninjauan citra. Apakah AI memberi nilai tambah?
   a. Tidak, karena AI selalu lebih buruk dari aturan
   b. Ya, karena tugasnya repetitif dan melelahkan, butuh "mata ahli", dan solusi rule-based sudah terbukti tidak memadai
   c. Tidak, karena data pasien bersifat sensitif
   d. Ya, karena AI selalu lebih murah

3. Bank memakai AI untuk menandai transfer internasional yang mencurigakan. Sistem lebih efektif dari proses lama, tetapi sering menandai transaksi sah. Praktik HCD yang paling tepat adalah…
   a. Menurunkan sensitivitas sampai tidak ada false positive
   b. Menyerahkan keputusan sepenuhnya ke model
   c. Menyediakan jalur peninjauan manusia dan mekanisme bagi nasabah untuk menantang/mengklarifikasi status transaksi, serta memantau dampaknya
   d. Menghapus fitur kewarganegaraan dari data

4. Saat memprototipe chatbot kesehatan publik pada masa wabah, urutan yang dianjurkan HCD adalah…
   a. Bangun versi AI selengkap mungkin, lalu uji ke pengguna beragam
   b. Bangun prototipe non-AI dengan cepat, uji ke kelompok pengguna beragam, lalu iterasi
   c. Uji ke pengguna internal saja, lalu rilis
   d. Tunggu data satu tahun sebelum membuat prototipe

5. Manakah yang **bukan** termasuk ukuran keselamatan (langkah 6) menurut tutorial?
   a. Membentuk red team internal permanen
   b. Memantau performa dan metrik fairness setelah sistem live
   c. Menyediakan cara bagi pengguna untuk melaporkan masalah keselamatan
   d. Menyembunyikan cara kerja sistem agar tidak dieksploitasi pengguna

---

## 3. Identifying Bias in AI

### Poin inti

- **Bias** dalam kursus ini didefinisikan sebagai konsekuensi negatif yang tidak diinginkan dari aplikasi ML, terutama bila konsekuensinya **timpang** ke kelompok tertentu. Aplikasi ML sudah tercatat mendiskriminasi berdasarkan ras, jenis kelamin, agama, status sosial-ekonomi, dan kategori lain. Kursus memakai kerangka **enam tipe bias** dari paper Suresh & Guttag (arXiv:1901.10002) karena kosakata bersama membuat tim bisa berdiskusi dan memitigasi bias secara produktif.
- "Garbage in, garbage out" hanya **satu** jalur bias (historical bias). Bias juga bisa muncul dari cara model didefinisikan, dari cara model dibandingkan dengan model lain, dan dari cara pengguna menafsirkan hasilnya — **bahaya bisa datang dari mana saja dalam proses ML**.
- **Historical bias** — muncul ketika kondisi dunia saat data dihasilkan memang sudah cacat. Contoh: hanya 7,4% CEO Fortune 500 (2020) adalah perempuan; menghapus manusia dari proses hiring dan memakai AI tidak otomatis adil, karena model belajar dari keputusan historis yang bias.
- **Representation bias** — dataset latih tidak mewakili orang yang akan dilayani model. Contoh: data dari aplikasi ponsel under-represent orang di atas 65 tahun; sistem deteksi wajah yang dilatih dominan kulit terang gagal untuk kulit gelap.
- **Measurement bias** — akurasi data berbeda antar kelompok, sering karena **proxy variable**. Contoh: model risiko kesehatan memakai biaya medis sebagai proxy risiko; karena pasien Black menghadapi hambatan akses dan biaya lebih rendah pada kondisi kesehatan yang sama, model justru kurang memilih pasien Black yang layak.
- **Aggregation bias** — kelompok digabung secara tidak tepat sehingga model tidak bagus untuk kelompok mana pun, atau hanya bagus untuk mayoritas. Paling sering muncul di aplikasi medis: prevalensi diabetes pada populasi Hispanik berbeda, sehingga model perlu dibuat sensitif terhadap perbedaan ini (memasukkan etnisitas sebagai fitur, atau model terpisah).
- **Evaluation bias** — data benchmark untuk membandingkan model tidak mewakili populasi yang dilayani. **Deployment bias** — masalah yang hendak dipecahkan model berbeda dari cara model benar-benar dipakai; jika pengguna akhir memakai model di luar maksud desainnya, tidak ada jaminan performanya bertahan.
- Tipe-tipe ini **tidak saling eksklusif**: satu aplikasi bisa kena beberapa sekaligus. Contoh: perangkat fitness wearable bisa kena representation bias (data kurang kulit gelap), measurement bias (sensor bekerja lebih buruk pada kulit gelap), dan evaluation bias (dataset benchmark juga kurang kulit gelap).

### Studi kasus / contoh nyata

- **Model deteksi teks toksik (Jigsaw / Civil Comments):** data ~2 juta komentar publik Civil Comments yang dirilis akhir 2017 dan dianotasi lewat kompetisi Kaggle 2019. Model sederhana mencapai akurasi sekitar 93%, tetapi "I have a muslim friend" dan "I have a black friend" ikut ditandai toksik sementara versi "christian"/"white" tidak — tanda model menyerap asosiasi identitas dari data, bukan memahami makna kalimat.
- **Gender Shades:** dua benchmark analisis wajah yang luas dipakai (IJB-A dan Adience) ternyata didominasi subjek berkulit terang (79,6% dan 86,2%). Klasifikasi gender komersial tampak "state of the art" di benchmark itu, tetapi tingkat errornya jauh lebih tinggi untuk orang berkulit gelap — evaluation bias yang berdampak nyata.

### Istilah kunci

- **Bias (dalam kursus ini)** — konsekuensi negatif tak diinginkan dari aplikasi ML, terutama yang dampaknya tidak proporsional pada kelompok tertentu.
- **Proxy variable** — variabel pengganti untuk hal yang tidak bisa diukur langsung (mis. biaya medis sebagai proxy risiko kesehatan); berbahaya bila hubungan proxy–target berbeda antar kelompok.
- **Historical bias** — bias yang berasal dari kondisi dunia yang sudah timpang saat data dikumpulkan; "garbage in, garbage out".
- **Measurement bias** — bias akibat kualitas/akurasi pengukuran (termasuk kualitas proxy) berbeda antar kelompok.
- **Deployment bias** — bias akibat penggunaan nyata berbeda dari maksud desain, sehingga jaminan performa gugur.

### Drill

1. Sebuah komunitas online terbukti islamofobik, sehingga komentar yang menyebut Islam lebih sering berlabel toksik. Model dilatih pada data itu. Tipe bias apa yang paling mungkin muncul?
   a. Aggregation bias
   b. Historical bias
   c. Deployment bias
   d. Evaluation bias

2. Tim menerjemahkan semua komentar non-Inggris ke Inggris dengan alat penerjemah, lalu memperlakukan semuanya seolah ditulis dalam bahasa Inggris. Tipe bias utama yang muncul adalah…
   a. Measurement bias (kualitas "pengukuran" teks berbeda antar kelompok bahasa), dan berpotensi aggregation bias karena perbedaan bahasa diperlakukan seragam
   b. Hanya deployment bias
   c. Hanya representation bias
   d. Tidak ada bias, karena semua data jadi satu bahasa

3. Model dilatih dan dievaluasi pada komentar pengguna di Inggris Raya, performanya bagus, lalu dipakai perusahaan di Australia dan anjlok karena perbedaan British vs Australian English. Tipe bias yang terlibat…
   a. Evaluation bias dan deployment bias
   b. Historical bias saja
   c. Aggregation bias saja
   d. Tidak ada, itu sekadar distribusi data baru

4. Sebuah model risiko kesehatan memakai biaya medis sebagai proxy tingkat risiko. Karena hambatan akses, pasien dari kelompok tertentu berbiaya lebih rendah pada kondisi kesehatan yang sama, sehingga model kurang memilih mereka. Ini contoh…
   a. Representation bias
   b. Measurement bias
   c. Deployment bias
   d. Aggregation bias

5. Manakah pernyataan yang **paling** sesuai dengan isi tutorial?
   a. Bias hanya berasal dari data latih
   b. Enam tipe bias saling eksklusif, satu model hanya bisa kena satu tipe
   c. Bias bisa muncul di berbagai tahap pipeline dan satu model dapat mengalami beberapa tipe sekaligus
   d. Menghapus atribut sensitif dari data selalu menghilangkan bias

---

## 4. AI Fairness

### Poin inti

- Tidak ada satu definisi tunggal "model yang adil". Pertanyaan pembuka tutorial: apakah adil bila tingkat persetujuan sama antar gender, atau lebih baik bila gender dihapus dari data? Empat kriteria di bawah adalah titik awal, bukan daftar lengkap, dan semuanya berlaku untuk model yang **menyeleksi individu untuk suatu hasil** (pinjaman, penerimaan universitas, tawaran kerja) — bukan untuk tugas seperti pengenalan wajah atau generasi teks.
- **Demographic parity / statistical parity:** model adil bila komposisi orang yang dipilih mencerminkan persentase keanggotaan kelompok pada populasi pelamar. Contoh: 50% peserta konferensi perempuan → 50% kandidat pembicara yang dipilih perempuan.
- **Equal opportunity:** proporsi orang yang *seharusnya* dipilih ("positif") dan benar-benar dipilih harus sama di tiap kelompok. Proporsi ini adalah **true positive rate (TPR)** alias **sensitivity**.
- **Equal accuracy:** persentase klasifikasi benar harus sama antar kelompok (mis. 98% akurat untuk satu kelompok berarti 98% juga untuk kelompok lain).
- **Group unaware / fairness through unawareness:** hapus seluruh informasi keanggotaan kelompok dari dataset. Jebakannya: harus hati-hati menghapus **proxy** — di kota yang segregatif secara ras, kode pos adalah proxy kuat untuk ras; dan pendekatan ini umumnya bukan solusi untuk historical bias.
- **Confusion matrix** dipakai untuk melihat performa per kelompok. Contoh pembuka: model dengan akurasi 80% (8 dari 10 benar) dan TPR 83% (5 dari 6 positif benar).
- **Tidak mungkin memenuhi lebih dari satu kriteria sekaligus** (Impossibility Theorem of Machine Fairness, arXiv:2007.06024). Contoh-contoh toy di tutorial memang masing-masing hanya memenuhi satu kriteria. Karena itu memilih kriteria adalah **percakapan panjang** dengan seluruh tim, bukan keputusan teknis sepihak.
- Di proyek nyata, model biasanya **tidak** memenuhi definisi fairness secara sempurna — target 50% bisa berakhir 48% atau 53%. Yang penting adalah mengukur dan menyadari selisihnya.

### Studi kasus / contoh nyata

- **Model persetujuan kartu kredit (dataset sintetis di exercise):** model menyetujui 38.246 orang — 8.028 dari Group A, 30.218 dari Group B. Akurasi 94,56% (A) vs 95,02% (B); TPR 77,23% (A) vs 98,03% (B). Artinya Group B diuntungkan pada ketiga kriteria sekaligus: representasi persetujuan, akurasi, dan kesempatan lolos bagi yang layak.
- **Group unaware yang tidak menolong:** setelah `Group` dihapus dari data latih, model tetap tidak serta-merta setara, karena fitur lain (pendapatan, kepemilikan rumah/mobil) membawa sinyal kelompok. Model ketiga — dengan ambang berbeda per kelompok (*group thresholds*) untuk menyamakan representasi persetujuan — memang memperbaiki demographic parity, tetapi mengorbankan kriteria lain. Ini ilustrasi langsung trade-off antar definisi fairness.

### Istilah kunci

- **Demographic parity** — kesetaraan komposisi hasil seleksi terhadap komposisi pelamar.
- **Equal opportunity** — kesetaraan **true positive rate (TPR) / sensitivity** antar kelompok, yaitu proporsi positif yang benar-benar terdeteksi sebagai positif.
- **Equal accuracy** — kesetaraan persentase klasifikasi benar antar kelompok.
- **Group unaware** — menghapus informasi keanggotaan kelompok (beserta proxy-nya) dari data.
- **Confusion matrix** — tabel silang prediksi vs aktual yang menjadi dasar penghitungan metrik per kelompok.

### Drill

1. Sebuah model beasiswa dipakai untuk memilih 200 penerima. Pelamar 60% perempuan, 40% laki-laki, dan panitia ingin komposisi penerima mencerminkan itu. Kriteria fairness yang dipakai adalah…
   a. Equal opportunity
   b. Demographic parity
   c. Equal accuracy
   d. Group unaware

2. Sebuah alat bantu dokter dirancang agar proporsi pasien berisiko yang berhasil terdeteksi sama di setiap kelompok demografis. Ini adalah…
   a. Equal opportunity (TPR sama antar kelompok)
   b. Demographic parity
   c. Group unaware
   d. Equal accuracy

3. Mengapa "hapus kolom gender/ras" sering tidak cukup?
   a. Karena model tidak bisa dilatih tanpa kolom itu
   b. Karena variabel lain bisa menjadi proxy (mis. kode pos untuk ras), dan pendekatan ini lemah terhadap historical bias
   c. Karena metrik fairness jadi tidak bisa dihitung
   d. Karena data selalu bersih

4. Menurut tutorial, berapa banyak kriteria fairness yang bisa dioptimalkan sebuah model secara bersamaan?
   a. Keempatnya
   b. Tiga
   c. Dua
   d. Praktis hanya satu; memenuhi lebih dari satu tidak memungkinkan (Impossibility Theorem)

5. Model A punya akurasi 95% untuk kedua kelompok, tetapi TPR-nya 70% vs 95%. Penilaian yang tepat…
   a. Model adil menurut equal accuracy, tetapi tidak adil menurut equal opportunity
   b. Model adil menurut semua kriteria
   c. Model tidak adil menurut equal accuracy maupun equal opportunity
   d. Tidak bisa dinilai tanpa melihat kode pos

---

## 5. Model Cards

### Poin inti

- **Model card** adalah dokumen singkat berisi informasi kunci tentang sebuah model ML: maksud penggunaan, cara kerja model, dan performanya dalam berbagai situasi. Tujuannya **meningkatkan transparansi** ke audiens luas.
- Analogi yang dipakai tutorial: model card mirip **label gizi pada kemasan makanan** — ringkas, terstandar, dan membantu orang membuat keputusan tanpa harus memahami seluruh proses produksi.
- Model cards diperkenalkan lewat paper 2019 (arXiv:1810.03993). Contoh yang dianjurkan dibaca: model card Salesforce, model card GPT-3 dari OpenAI, dan contoh model card Google Cloud (deteksi wajah).
- **Audiens menentukan gaya penulisan.** Model card harus seimbang antara mudah dipahami dan tetap memuat informasi teknis penting. Contoh: model card untuk sistem bantu baca rontgen ortopedi akan dibaca dokter, peneliti, pasien, pembuat kebijakan, dan developer sistem serupa — sehingga boleh mengasumsikan pemahaman kesehatan dan AI.
- Sembilan bagian menurut paper aslinya: **(1) Model Details** (latar belakang, developer, versi), **(2) Intended Use** (use case in-scope, pengguna yang dituju, use case out-of-scope), **(3) Factors** (faktor yang memengaruhi dampak model: demografis, lingkungan, instrumentasi), **(4) Metrics** (metrik apa dan mengapa; untuk klasifikasi: FPR, FNR, false discovery rate, false omission rate), **(5) Evaluation Data**, **(6) Training Data**, **(7) Quantitative Analyses** (performa per faktor **dan irisan** faktor), **(8) Ethical Considerations** (data sensitif, implikasi bagi nyawa/kesehatan/keselamatan, mitigasi risiko, potensi bahaya), **(9) Caveats and Recommendations**.
- Organisasi boleh menambah, mengurangi, atau menyusun ulang bagian sesuai kebutuhan — sembilan bagian itu cetak biru, bukan aturan mati.
- Hambatan nyata: organisasi sering enggan membuka proses, data proprietary, atau rahasia dagang. Jalan tengahnya adalah membuat model card yang tetap berguna dan memberdayakan tanpa memuat informasi sensitif. Beberapa tim memakai format lain seperti **FactSheets** untuk mencatat informasi model.

### Studi kasus / contoh nyata

- **Simple Zoom (skenario exercise):** alat penyunting video yang otomatis melakukan zoom ke presenter. Model card-nya harus ditulis untuk beberapa audiens sekaligus (pengguna, integrator, dsb.), dan bagian *Intended Use* menyatakan alat ini **tidak cocok** untuk presentasi dengan lebih dari satu presenter atau presenter yang tertutup sebagian.
- **Integrasi Simple Zoom ke Presenter Pro (skenario exercise):** tim produk membaca model card dan menemukan risiko nyata — pelanggan yang memakai properti besar sehingga presenter kadang tertutup sebagian berada di luar cakupan *Intended Use*, sehingga integrasi bisa gagal untuk mereka; solusinya bisa berupa pembatasan fitur, panduan penggunaan, atau penyesuaian kontrak dengan vendor. Ini menunjukkan model card berfungsi sebagai alat due diligence, bukan sekadar formalitas.

### Istilah kunci

- **Model card** — dokumen ringkas berisi detail model, maksud penggunaan, faktor, metrik, data, analisis kuantitatif, pertimbangan etis, dan catatan keterbatasan.
- **Intended use** — pernyataan use case yang masuk cakupan dan yang di luar cakupan, beserta pengguna yang dituju.
- **Factors** — variabel yang memengaruhi performa/dampak model (demografis, lingkungan, perangkat).
- **Quantitative analyses** — pelaporan performa per kelompok dan per irisan kelompok (mis. lansia-perempuan, muda-laki-laki).
- **FactSheet** — format alternatif pendokumentasian informasi model ML.

### Drill

1. Fungsi utama model card menurut tutorial adalah…
   a. Meningkatkan akurasi model
   b. Meningkatkan transparansi dengan mengomunikasikan informasi kunci model ke audiens luas
   c. Menggantikan proses audit
   d. Menyimpan bobot model

2. Sebuah model card untuk alat bantu diagnosis rontgen sebaiknya…
   a. Hanya berisi angka akurasi global
   b. Ditulis untuk audiens yang beragam (klinisi, peneliti, pasien, pembuat kebijakan) dan boleh mengasumsikan sebagian pengetahuan kesehatan/AI
   c. Ditulis hanya untuk tim internal
   d. Tidak menyebut keterbatasan agar tidak menimbulkan keraguan

3. Bagian mana yang secara eksplisit menyatakan use case yang **tidak** cocok untuk model?
   a. Model Details
   b. Factors
   c. Intended Use
   d. Caveats and Recommendations

4. Pelaporan performa per irisan faktor (mis. lansia-perempuan) termasuk bagian…
   a. Quantitative Analyses
   b. Training Data
   c. Ethical Considerations
   d. Metrics

5. Hambatan umum penerapan model card di organisasi dan jalan tengah yang disebut tutorial adalah…
   a. Tidak ada hambatan; semua organisasi wajib membuka data
   b. Kekhawatiran membuka proses/data proprietary/rahasia dagang → susun model card yang berguna tanpa informasi sensitif, atau pakai format lain seperti FactSheets
   c. Model card hanya untuk model open source
   d. Model card harus memuat seluruh kode pelatihan

---

## Kunci Jawaban

### Lesson 1 — Introduction to AI Ethics

1. **b** — Setelah menilai bahaya, tim bisa mendesain ulang untuk mengurangi bahaya, atau memutuskan tidak membangun sistem.
2. **c** — Kutipan tutorial: *"Ethics is a conversation"*; kursus membantu memulai, bukan menyelesaikan.
3. **c** — Dampak pada lapangan kerja disebut sebagai topik yang tidak dicakup, bersama kesenjangan digital dan kapabilitas pemerintah otoriter.
4. **b** — HCD → Bias → Fairness → Model cards.
5. **c** — Halaman kursus: "This course has no prerequisites".

### Lesson 2 — Human-Centered Design for AI

1. **b** — Langkah 1: observasi, wawancara, dan focus group dengan orang yang sudah menangani masalah.
2. **b** — Tiga dari empat pertanyaan penyaring langkah 2 terpenuhi: hasil disepakati baik, tugas repetitif dan melelahkan, dan rule-based terbukti tidak menambah nilai.
3. **c** — Langkah 5: sediakan jalur tantangan/opt-out dan pemantauan; bukan menurunkan sensitivitas sampai nol (itu mengorbankan tujuan deteksi).
4. **b** — Langkah 4: prototipe non-AI dulu, uji dengan kelompok beragam, iterasi cepat dan murah.
5. **d** — Justru sebaliknya: UI harus memudahkan orang memahami cara kerja sistem, mengubah pengaturan, dan memberi umpan balik.

### Lesson 3 — Identifying Bias in AI

1. **b** — Kondisi dunia saat data dihasilkan (komunitas islamofobik) sudah cacat → historical bias.
2. **a** — Kualitas "pengukuran" teks berbeda antar kelompok bahasa (measurement bias), dan memperlakukan semua bahasa seragam berpotensi aggregation bias.
3. **a** — Benchmark hanya mewakili populasi UK (evaluation bias) dan dipakai pada populasi Australia (deployment bias).
4. **b** — Biaya adalah proxy yang kualitasnya berbeda antar kelompok → measurement bias.
5. **c** — Bias bisa muncul di banyak tahap pipeline dan tidak saling eksklusif; menghapus atribut sensitif tidak otomatis menghilangkan bias (proxy tetap ada).

### Lesson 4 — AI Fairness

1. **b** — Komposisi hasil seleksi dibuat mencerminkan komposisi pelamar → demographic/statistical parity.
2. **a** — Kesamaan proporsi positif yang tertangkap = TPR sama = equal opportunity.
3. **b** — Proxy (kode pos untuk ras) harus ikut dihapus, dan group unaware umumnya bukan solusi untuk historical bias.
4. **d** — Impossibility Theorem: tidak mungkin mengoptimalkan lebih dari satu kriteria; pilih satu lewat diskusi tim.
5. **a** — Akurasi sama (memenuhi equal accuracy) tetapi TPR berbeda jauh (melanggar equal opportunity). Ini juga contoh bahwa memenuhi satu kriteria tidak berarti memenuhi yang lain.

### Lesson 5 — Model Cards

1. **b** — Definisi tutorial: dokumen singkat untuk meningkatkan transparansi ke audiens luas.
2. **b** — Audiens beragam, dengan asumsi pengetahuan domain tertentu; tetap memuat informasi teknis penting.
3. **c** — *Intended Use* memuat use case in-scope, pengguna yang dituju, dan use case out-of-scope.
4. **a** — *Quantitative Analyses* melaporkan performa per faktor dan irisan faktor.
5. **b** — Kekhawatiran proprietary/trade secret diselesaikan dengan model card yang berguna tanpa informasi sensitif, atau format alternatif seperti FactSheets.

---

## Ringkasan 1 paragraf

Kaggle *Intro to AI Ethics* membingkai etika AI sebagai percakapan yang berjalan, bukan sertifikasi yang selesai: ia dimulai dari **human-centered design** yang menuntut tim memahami kebutuhan orang sebelum menulis kode, bertanya apakah AI benar-benar menambah nilai dibanding solusi rule-based, menimbang bahaya di sepanjang pipeline (dan berani tidak membangun), memprototipe non-AI lebih dulu, membuka jalan bagi pengguna untuk menantang sistem, serta menanamkan pengawasan manusia dan red team sebagai ukuran keselamatan. Karena bahaya bisa muncul di mana saja, lesson berikutnya memecah **bias** menjadi enam tipe — historical, representation, measurement, aggregation, evaluation, deployment — dengan contoh nyata seperti Gender Shades dan model toksik Jigsaw yang menandai "muslim" dan "black" sebagai toksik. Untuk mengukur bias, lesson **fairness** menawarkan empat kriteria (demographic parity, equal opportunity, equal accuracy, group unaware) sekaligus menunjukkan lewat confusion matrix bahwa kriteria-kriteria itu saling bertentangan, sehingga memilihnya adalah keputusan etis, bukan teknis semata. Terakhir, **model cards** — sembilan bagian, dianalogikan sebagai label gizi — menjadi mekanisme transparansi dan akuntabilitas publik yang memaksa tim mendokumentasikan maksud penggunaan, faktor, metrik, dan keterbatasan modelnya. Benang merah seluruh kursus: keputusan teknis selalu keputusan tentang orang, dan dampaknya harus diukur per kelompok, bukan dirata-ratakan.

## Sumber

- Kaggle Learn — *Intro to AI Ethics* (halaman kursus, daftar 5 lesson, estimasi 4 jam, tanpa prasyarat): https://www.kaggle.com/learn/intro-to-ai-ethics
- Lesson 1 — *Introduction to AI Ethics* (Var Shankar): https://www.kaggle.com/code/var0101/introduction-to-ai-ethics
- Lesson 2 — *Human-Centered Design for AI* (Var Shankar): https://www.kaggle.com/code/var0101/human-centered-design-for-ai
- Lesson 3 — *Identifying Bias in AI* (Alexis Cook): https://www.kaggle.com/code/alexisbcook/identifying-bias-in-ai
- Lesson 4 — *AI Fairness* (Alexis Cook): https://www.kaggle.com/code/alexisbcook/ai-fairness
- Lesson 5 — *Model Cards* (Var Shankar): https://www.kaggle.com/code/var0101/model-cards
- Exercise notebooks (skenario & angka): Human-Centered Design `.../code/fork/15577437`, Identifying Bias `.../code/fork/15622654`, AI Fairness `.../code/fork/15622664`, Model Cards `.../code/fork/15577440`
- Rujukan yang disebut di dalam tutorial: Suresh & Guttag, *A Framework for Understanding Unintended Consequences of Machine Learning* (arXiv:1901.10002) · Mitchell et al., *Model Cards for Model Reporting* (arXiv:1810.03993) · *Impossibility Theorem of Machine Fairness* (arXiv:2007.06024) · Buolamwini & Gebru, *Gender Shades* · Jigsaw Unintended Bias in Toxicity Classification (Kaggle, 2019) · Inclusive Images Challenge (Kaggle)
