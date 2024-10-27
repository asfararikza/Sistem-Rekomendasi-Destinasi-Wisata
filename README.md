# Sistem-Rekomendasi-Destinasi-Wisata

---

## Project Overview

---

Industri pariwisata di Indonesia, terutama setelah pandemi Covid-19, mengalami tantangan signifikan akibat minimnya informasi tentang destinasi wisata, terutama di daerah seperti Kabupaten Malang (Putri & Suliadi, 2023). Banyak wisatawan kesulitan dalam menentukan pilihan tempat wisata yang tepat karena informasi yang tersedia di internet tidak mencukupi, sehingga mereka tidak dapat memanfaatkan potensi wisata yang ada (Islamiyah et. al, 2019). Hal ini dapat mengakibatkan ketidakpuasan dan keputusan yang kurang tepat saat memilih destinasi wisata.

Dengan kondisi ini, penting untuk mengembangkan sistem rekomendasi yang menggunakan metode content-based filtering dan collaborative filtering. Sistem ini diharapkan dapat membantu wisatawan menemukan destinasi yang sesuai dengan preferensi mereka dan meningkatkan pengalaman berwisata secara keseluruhan.

Referensi:
* [Rekomendasi Destinasi Wisata di Indonesia Menggunakan Metode Item2Vec](https://journals.unisba.ac.id/index.php/JRS/article/view/1770/1147)

* [Pemanfaatan MetodeItemBased Collaborative Filtering Untuk Rekomendasi Wisata Di Kabupaten Malang](https://jurnal.stmikasia.ac.id/index.php/jitika/article/view/70/249)
---

## Business Understanding

---

### Problem Statements
Industri pariwisata di Indonesia, terutama di Kabupaten Malang, menghadapi tantangan serius akibat minimnya informasi yang tersedia tentang destinasi wisata. Banyak wisatawan kesulitan untuk menemukan tempat wisata yang sesuai dengan preferensi mereka, sehingga mengurangi kepuasan dan pengalaman berwisata. Selain itu, dengan banyaknya pilihan yang ada, wisatawan sering merasa bingung dan tidak yakin dalam memilih destinasi yang tepat, yang dapat berujung pada keputusan pembelian yang salah.

### Goals
Tujuan dari pengembangan sistem rekomendasi ini adalah untuk:
1. Meningkatkan akses informasi mengenai destinasi wisata yang tersedia di Kabupaten Malang.
2. Membantu wisatawan dalam menemukan tempat wisata yang sesuai dengan preferensi mereka.

### Solution Approach
Untuk mencapai tujuan tersebut, dua pendekatan solusi dapat diimplementasikan:
1. **Content-Based Filtering**: menganalisis karakteristik dan fitur dari destinasi wisata seperti deskripsi dan kategori wisata tersebut.
2. **Collaborative Filtering**: memanfaatkan data dari pengguna lain untuk memberikan rekomendasi. Sistem akan menganalisis pola preferensi dan penilaian dari berbagai pengguna untuk menemukan kesamaan antara mereka.


## Data Understanding

---

Dataset yang digunakan dalam proyek ini adalah Indonesia Tourism Destination yang tersedia di [Kaggle](https://www.kaggle.com/datasets/aprabowo/indonesia-tourism-destination/data?select=tourism_rating.csv). Dataset ini terdiri dari beberapa file:

1. **tourism_with_id.csv** (437 records):
   - Place_Id: ID unik untuk setiap destinasi
   - Place_Name: Nama destinasi wisata
   - Description: Deskripsi tempat wisata
   - Category: Kategori tempat wisata
   - City: Kota lokasi wisata
   - Price: Harga tiket masuk
   - Rating: Rating rata-rata destinasi
   - Location: Koordinat lokasi

2. **tourism_rating.csv** (10000 records):
   - User_Id: ID unik pengguna
   - Place_Id: ID destinasi wisata
   - Place_Ratings: Rating yang diberikan (1-5)

## Exploratory Data Analysis (EDA)
---
### 1. Analisis Destinasi Wisata

#### 1.1 Statistik Deskriptif Harga dan Rating
```python
df_destinasi[['Price', 'Rating']].describe()
```
Analisis statistik menunjukkan variasi harga destinasi yang signifikan, mulai dari gratis hingga Rp 900,000 dengan rata-rata Rp 24,652. Rating destinasi secara keseluruhan sangat baik dengan rata-rata 4.44 dari 5, dengan rentang rating 3.40 hingga 5.00, mengindikasikan kepuasan pengunjung yang tinggi terhadap destinasi-destinasi wisata tersebut.
#### 1.2 Kota dengan Rating Tertinggi
![rating kota](https://raw.githubusercontent.com/asfararikza/Sistem-Rekomendasi-Destinasi-Wisata/refs/heads/main/images/Kota%20rating%20tertinggi.png)

Pada dataset ini menunjukkan bahwa kota yang memiliki rata-rata rating destinasi wisata tertinggi adalah Jakarta. Kemudian diikuti dengan kota Yogyakarta yang terkenal dengan destinasi wisata bersejarahnya.

### 1.3 Distribusi Kategori Destinasi
![jumlah wisata](https://raw.githubusercontent.com/asfararikza/Sistem-Rekomendasi-Destinasi-Wisata/refs/heads/main/images/jumlah%20wisata.png)

Berdasarkan pie chart yang dihasilkan:
* Taman Hiburan memiliki persentase terbesar dari total destinasi wisata
* Diikuti oleh destinasi Budaya dan Cagar Alam
* Distribusi ini mencerminkan keberagaman jenis destinasi wisata yang tersedia

Insight:
* Taman Hiburan mendominasi, menunjukkan minat besar wisatawan terhadap hiburan keluarga
* Kategori Budaya yang signifikan menunjukkan kekayaan budaya Indonesia melalui museum dan situs sejarah
* Cagar Alam yang juga memiliki proporsi besar mencerminkan kesadaran akan pelestarian lingkungan dan minat wisatawan untuk menikmati keindahan alam
---
## 2. Analisis Data Rating
### 2.1 Statistik Rating
```python
df_rating['Place_Ratings'].describe()
```
Hasil deskripsi dari kolom Place_Ratings menunjukkan bahwa dataset memiliki total 10.000 entri dengan nilai rating berkisar antara 1 hingga 5. Rata-rata rating yang diberikan adalah 3,07, yang mengindikasikan bahwa secara umum, pengguna memberikan rating yang cenderung positif tetapi tidak terlalu tinggi.

### 2.2 Distribusi Pengguna dan Destinasi
* Terdapat 300 pengguna unik yang memberikan rating
* Ada 437 tempat wisata yang berbeda yang mendapat rating
---
## Data Preparation

### A. Preparation untuk Content-Based Filtering

Pada tahap ini, dilakukan beberapa langkah untuk menyiapkan data yang akan digunakan dalam model content-based filtering. Berikut langkah-langkah data preparation yang dilakukan:

1. **Membuat Dataframe Tempat Wisata**  
   - Data tempat wisata diambil dari `df_destinasi` yang berisi informasi tentang `Place_Id`, `Place_Name`, `Description`, `Category`, dan `City`. Hanya kolom `Place_Id` dan `Place_Name` yang diambil sebagai data utama untuk daftar destinasi (`all_destinasi`), sementara kolom lainnya digunakan untuk proses selanjutnya.

2. **Menggabungkan Informasi Deskripsi, Kategori, dan Kota Menjadi Kolom `Tags`**  
   - Kolom `Tags` dibuat dengan menggabungkan kolom `Description`, `Category`, dan `City`. Kolom ini akan menjadi dasar untuk menghitung kesamaan konten dalam model. Langkah ini dilakukan agar setiap destinasi memiliki representasi teks yang komprehensif, sehingga model dapat memahami preferensi pengguna berdasarkan kata-kata yang muncul di deskripsi, kategori, dan kota.

3. **Pemeriksaan Missing Value**  
   - Dilakukan pemeriksaan data kosong (missing value) pada kolom `Tags` menggunakan `all_destinasi.isnull().sum()`. Langkah ini penting untuk memastikan bahwa setiap tempat wisata memiliki informasi yang lengkap. Data yang kosong dapat menyebabkan masalah saat model mencoba menghitung kesamaan antar destinasi.

4. **Pemeriksaan Duplikat**  
   - Jumlah data duplikat dicek menggunakan `all_destinasi.duplicated().sum()`. Duplikasi perlu dihindari karena dapat menyebabkan bias dalam rekomendasi, di mana destinasi yang sama bisa lebih sering muncul dalam rekomendasi.

### B. Preparation untuk Collaborative Filtering

Pada tahap data preparation untuk model collaborative filtering, dilakukan beberapa langkah penting agar data siap digunakan dalam proses pelatihan model. Berikut adalah penjelasan langkah-langkah tersebut:

1. **Menghapus Duplikasi pada User ID dan Place ID**  
   - Setiap `User_Id` dan `Place_Id` diubah menjadi list tanpa nilai yang sama untuk menghindari duplikasi, memastikan bahwa setiap pengguna dan destinasi memiliki identitas yang unik. Langkah ini penting agar encoding dapat dilakukan secara konsisten dan mencegah bias dalam model.

2. **Melakukan Encoding pada User ID dan Place ID**  
   - Encoding dilakukan dengan mengonversi `User_Id` dan `Place_Id` menjadi indeks numerik. Hal ini dilakukan untuk memungkinkan model mengolah data berbentuk numerik dan memetakan setiap pengguna dan destinasi dengan cara yang mudah dipahami oleh algoritma. Tabel mapping dua arah dibuat agar model dapat dengan mudah menghubungkan ID numerik dengan ID asli saat memberikan rekomendasi.

3. **Mapping User ID dan Place ID ke DataFrame Rating**  
   - Kolom baru `user` dan `destinasi` ditambahkan ke `df_rating` yang masing-masing menyimpan hasil encoding dari `User_Id` dan `Place_Id`. Langkah ini memudahkan model untuk mengakses data rating dalam bentuk numerik yang sudah di-encode.

4. **Konversi Tipe Data Rating ke Float**  
   - Data rating dikonversi menjadi tipe `float32` untuk memastikan tipe data konsisten selama proses pelatihan dan memudahkan perhitungan. Selain itu, tipe data float dipilih karena lebih akurat dalam perhitungan numerik dibandingkan tipe integer.

5. **Normalisasi Nilai Rating**  
   - Untuk mempermudah model memahami skala data, rating dinormalisasi ke rentang 0–1 dengan rumus:  
     
     rating =( rating asli - min rating)/(max rating - min rating)

   - Normalisasi membantu model untuk mengenali perbedaan tingkat kepuasan pengguna tanpa bias nilai rating yang besar atau kecil.

6. **Pengacakan Dataset**  
   - Dataset diacak (`shuffled`) dengan parameter `random_state=42` agar data yang dilatih tidak memiliki pola urutan yang mungkin menyebabkan model overfitting. Hal ini bertujuan agar model lebih fleksibel dan tidak bergantung pada pola tertentu.

7. **Membagi Dataset Menjadi Data Train dan Validasi**  
   - Dataset dibagi menjadi 80% data pelatihan (`train`) dan 20% data validasi (`validation`). Pembagian ini penting untuk mengukur performa model pada data yang belum dilihat selama pelatihan, sehingga model dapat dievaluasi dengan lebih akurat.


## Modeling and Result
---
### 1. Content-Based Filtering
Model ini menggunakan cosine similarity untuk menghitung kemiripan antar destinasi wisata berdasarkan deskripsi dan kategori.

Kelebihan:
- Tidak memerlukan data dari pengguna lain
- Dapat memberikan rekomendasi untuk item baru

Kekurangan:
- Terbatas pada fitur yang tersedia
- Tidak dapat mempelajari preferensi pengguna

**Proses Modeling Content-Based Filtering**

1. **Menggunakan TF-IDF Vectorization**  
   - Untuk menangkap fitur dari teks yang ada di kolom `Tags`, `TfidfVectorizer` dari `sklearn` digunakan. Proses ini mengubah teks menjadi matriks vektor TF-IDF, yang mengukur seberapa penting kata dalam sebuah dokumen relatif terhadap koleksi dokumen (corpus).

2. **Membentuk Matriks TF-IDF**  
   - Matriks TF-IDF yang dihasilkan kemudian diubah menjadi bentuk matriks padat dengan `todense()`, dan diorganisasi ke dalam DataFrame yang menunjukkan kemunculan kata-kata dalam setiap destinasi.

3. **Menghitung Cosine Similarity**  
   - Menggunakan `cosine_similarity` untuk menghitung kemiripan antara semua destinasi berdasarkan matriks TF-IDF. Matriks ini berisi nilai kemiripan antara setiap pasangan destinasi.

4. **Membuat DataFrame Kemiripan**  
   - Hasil perhitungan cosine similarity disimpan dalam DataFrame `cosine_sim_df`, yang memiliki nama destinasi sebagai indeks dan kolom, sehingga memudahkan dalam menemukan destinasi serupa.

5. **Fungsi Rekomendasi Destinasi**  
   - Sebuah fungsi `destination_recommendations` didefinisikan untuk memberikan rekomendasi berdasarkan kemiripan. Fungsi ini mengambil nama destinasi sebagai input dan mengembalikan sejumlah destinasi serupa (k) berdasarkan nilai kemiripan tertinggi.

**Output Model**


![output](https://raw.githubusercontent.com/asfararikza/Sistem-Rekomendasi-Destinasi-Wisata/refs/heads/main/images/Output%20content%20based%20filtering.png)

---
### 2. Collaborative Filtering
Model ini menggunakan matrix factorization dengan algoritma SVD untuk memberikan rekomendasi berdasarkan pola rating pengguna.

Kelebihan:
- Dapat menemukan pola yang kompleks
- Memberikan rekomendasi personal

Kekurangan:
- Membutuhkan data rating yang cukup
- Cold-start problem untuk pengguna baru

**Proses Modeling Collaborative Filtering**

1. **Mendefinisikan Model**  
   - Model `RecommenderNet` dibuat sebagai subclass dari `tf.keras.Model`. Model ini mengandung beberapa layer embedding untuk pengguna dan destinasi, serta bias untuk masing-masing. Embedding memungkinkan representasi numerik dari pengguna dan destinasi yang dapat belajar dari interaksi mereka.

2. **Inisialisasi Layer**  
   - Layer embedding untuk pengguna dan destinasi diinisialisasi dengan ukuran yang ditentukan (`embedding_size`). Regularisasi L2 diterapkan untuk mencegah overfitting.

3. **Mendefinisikan Fungsi Panggilan**  
   - Fungsi `call` menghitung prediksi dengan mengalikan vektor pengguna dan destinasi, kemudian menambahkan bias masing-masing. Hasilnya diproses dengan fungsi aktivasi sigmoid untuk mendapatkan nilai prediksi antara 0 dan 1.

4. **Kompilasi Model**  
   - Model dikompilasi menggunakan fungsi loss Binary Crossentropy dan optimizer Adam. Metric RMSE (Root Mean Squared Error) ditambahkan untuk mengevaluasi kinerja model.

5. **Pelatihan Model**  
   - Model dilatih pada data training (`x_train`, `y_train`) selama 100 epoch, dengan validasi dilakukan menggunakan data validation (`x_val`, `y_val`). Hasil pelatihan mencakup nilai loss dan RMSE pada data training dan validasi.

6. **Rekomendasi untuk Pengguna**  
   - Setelah model dilatih, dilakukan prediksi untuk rekomendasi destinasi bagi pengguna yang dipilih. Destinasi yang telah dikunjungi oleh pengguna diambil, dan destinasi yang belum dikunjungi diidentifikasi. Rekomendasi diberikan berdasarkan rating tertinggi yang diprediksi oleh model untuk destinasi yang belum dikunjungi.
  
**Output Model**


![output colab](https://raw.githubusercontent.com/asfararikza/Sistem-Rekomendasi-Destinasi-Wisata/refs/heads/main/images/output%20collaborative%20filtering.png)



## Evaluation
---
### 1. Content-Based Filtering
Dalam proyek ini, metrik evaluasi yang digunakan untuk model content-based filtering adalah metrik precision. Dalam sistem rekomendasi, precision adalah jumlah item rekomendasi yang relevan. Metrik ini tidak bisa dipanggil melalui library scikit learn karena tidak ada data target/label seperti pada supervised learning.

![metrik](https://camo.githubusercontent.com/cfd2d91b4d2604fc74af4f2422a6e91dc726fa7ef3d4d3c77850ad85c84c879c/68747470733a2f2f6d69726f2e6d656469756d2e636f6d2f76322f726573697a653a6669743a313430302f302a3636765f43627545453067746a75434b)

Hasil dari output model content based filtering dengan input model Pantai Indrayanti menghasilkan lima rekomendasi berikut.
![output](https://raw.githubusercontent.com/asfararikza/Sistem-Rekomendasi-Destinasi-Wisata/refs/heads/main/images/Output%20content%20based%20filtering.png)

Pada hasil tersebut menunjukkan bahwa seluruh rekomendasi yang model berikan sesuai yakni 5 rekomendasi destinasi wisata pantai. Berikut perhitungan evaluasi model ini:
* Precision = 5/5. Jadi presisinya = 100%

Ini berarti sistem rekomendasi memiliki Precision 100% atau 1.0, yang menunjukkan bahwa semua rekomendasi sesuai dengan tags yang mirip dengan Pantai Indrayanti.

---
### 2. Collaborative Filtering
Dalam proyek ini, metrik evaluasi yang digunakan adalah Root Mean Squared Error (RMSE). RMSE adalah ukuran yang biasa digunakan untuk mengevaluasi kualitas prediksi dalam model rekomendasi. RMSE mengukur perbedaan antara nilai yang diprediksi oleh model dan nilai aktual dalam dataset. Nilai RMSE yang lebih rendah menunjukkan bahwa model memberikan prediksi yang lebih akurat. Metrik ini cocok untuk mengevaluasi rekomendasi berbasis rating seperti pada Collaborative Filtering.

![metrik rmse](https://camo.githubusercontent.com/e3b9f25a2824bdf256a7ef76d05d734a1058be4d860ff5173a030609ac42d40a/68747470733a2f2f6d656469612e6765656b73666f726765656b732e6f72672f77702d636f6e74656e742f75706c6f6164732f32303230303632323137313734312f524d5345312e6a7067)

Model collaborative filtering ini menghasilkan nilai **loss** sebesar **0.6464** pada data pelatihan dan **0.7173** pada data validasi. Nilai **loss** yang lebih rendah pada data pelatihan dibandingkan dengan data validasi menunjukkan bahwa model telah belajar dengan baik dari data latih, tetapi perbedaan ini juga menunjukkan adanya potensi **overfitting**. 

Secara keseluruhan, nilai **loss** yang relatif rendah, bersama dengan **RMSE** yang juga rendah (**0.3110** untuk pelatihan dan **0.3598** untuk validasi), menunjukkan bahwa model mampu memprediksi rating dengan akurat, meskipun ada sedikit kekhawatiran mengenai kemampuan model dalam menggeneralisasi pada data baru.



## Kesimpulan
Sistem rekomendasi yang dibangun berhasil memberikan rekomendasi destinasi wisata yang relevan dengan akurasi yang baik. Content-based filtering efektif untuk merekomendasikan destinasi serupa, sementara collaborative filtering baik dalam memberikan rekomendasi personal berdasarkan preferensi pengguna.

