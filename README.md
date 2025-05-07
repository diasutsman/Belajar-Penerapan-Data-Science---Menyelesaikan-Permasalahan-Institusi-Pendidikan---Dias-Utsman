# Proyek Akhir: Menyelesaikan Permasalahan Perusahaan Edutech

## Business Understanding

Jaya Jaya Institut adalah sebuah institusi pendidikan perguruan tinggi yang telah berdiri sejak tahun 2000. Institusi ini telah berhasil mencetak banyak lulusan dengan reputasi yang sangat baik di bidangnya masing-masing. Namun, Jaya Jaya Institut menghadapi tantangan serius terkait dengan tingginya tingkat dropout mahasiswa yang tidak menyelesaikan pendidikannya.

### Permasalahan Bisnis
Tuliskan seluruh permasalahan bisnis yang akan diselesaikan.

1. Tingginya tingkat dropout mahasiswa (32.1% dari total mahasiswa terdaftar)
2. Kesulitan mengidentifikasi faktor-faktor utama yang mempengaruhi dropout
3. Tidak adanya sistem prediksi dini untuk mahasiswa berisiko dropout
4. Kesulitan monitoring performa akademik mahasiswa secara efektif

### Cakupan Proyek
Tuliskan cakupan proyek yang akan dikerjakan.

1. Analisis data mahasiswa untuk memahami faktor-faktor yang mempengaruhi dropout
2. Pengembangan model machine learning untuk memprediksi mahasiswa berisiko dropout
3. Pembuatan dashboard untuk monitoring performa akademik mahasiswa
4. Implementasi sistem prediksi dalam bentuk aplikasi web
5. Penyusunan rekomendasi action items untuk mengurangi tingkat dropout

### Persiapan

Sumber data: Dataset mahasiswa Jaya Jaya Institut yang berisi informasi demografis, akademis, dan sosio-ekonomi dari 4424 mahasiswa dengan 37 atribut.

Setup environment:
```
pip install -r requirements.txt
```

## Business Dashboard
Jelaskan tentang business dashboard yang telah dibuat. Jika ada, sertakan juga link untuk mengakses dashboard tersebut.

Untuk membantu Jaya Jaya Institut dalam memonitor performa mahasiswa, telah dibuat dashboard interaktif menggunakan Metabase. Dashboard ini menyediakan visualisasi komprehensif mengenai distribusi status mahasiswa, faktor-faktor penyebab dropout, dan performa akademik mahasiswa.

Dashboard Metabase dapat diakses dengan kredensial berikut:
* **Email**: root@mail.com
* **Password**: root123

Screenshot dari dashboard dapat dilihat pada file `dias_utsman-dashboard`.

## Menjalankan Sistem Machine Learning
Jelaskan cara menjalankan protoype sistem machine learning yang telah dibuat. Selain itu, sertakan juga link untuk mengakses prototype tersebut.

System machine learning yang telah dikembangkan untuk memprediksi mahasiswa yang berisiko dropout terdiri dari model Random Forest dengan akurasi 71.52% dan terintegrasi dengan aplikasi web Streamlit.

```
# 1. Clone repository (jika belum dilakukan)
# git clone <repo-url>

# 2. Pindah ke direktori proyek
cd "Belajar Penerapan Data Science - Menyelesaikan Permasalahan Institusi Pendidikan"

# 3. Install dependencies
pip install -r requirements.txt

# 4. Jalankan aplikasi Streamlit
streamlit run app.py
```

Setelah menjalankan perintah di atas, aplikasi akan berjalan di browser pada alamat http://localhost:8501

## Conclusion
Jelaskan konklusi dari proyek yang dikerjakan.

Berdasarkan analisis data dan model machine learning yang telah dikembangkan, dapat disimpulkan bahwa:

1. Faktor akademik, khususnya performa pada semester pertama dan kedua, menjadi prediktor terkuat untuk risiko dropout mahasiswa.

2. Faktor ekonomi seperti status beasiswa dan keterlambatan pembayaran biaya kuliah memiliki pengaruh signifikan terhadap keputusan dropout.

3. Model Random Forest yang dikembangkan mampu memprediksi mahasiswa berisiko dropout dengan akurasi 71.52%, yang cukup baik untuk implementasi sistem early warning.

4. Identifikasi dini mahasiswa berisiko dropout memungkinkan institusi untuk melakukan intervensi tepat waktu, yang berpotensi mengurangi tingkat dropout secara signifikan.

### Rekomendasi Action Items
Berikan beberapa rekomendasi action items yang harus dilakukan perusahaan guna menyelesaikan permasalahan atau mencapai target mereka.
- Implementasi Sistem Early Warning: Mengintegrasikan model prediksi dropout ke dalam sistem akademik untuk mendeteksi mahasiswa berisiko secara otomatis di minggu-minggu awal semester.
- Program Mentoring Akademik: Membentuk program mentoring akademik khusus bagi mahasiswa yang teridentifikasi berisiko tinggi dropout, dengan fokus pada mata kuliah yang sering menjadi hambatan.
- Dukungan Finansial Terstruktur: Menyediakan program bantuan keuangan dan beasiswa yang lebih terstruktur untuk mahasiswa dengan masalah ekonomi.
- Workshop Keterampilan Belajar: Menyelenggarakan workshop reguler tentang keterampilan belajar, manajemen waktu, dan strategi menghadapi ujian, terutama bagi mahasiswa baru.
- Monitoring Berkelanjutan: Melakukan monitoring berkelanjutan terhadap performa mahasiswa dan mengukur efektivitas program intervensi yang diterapkan.
