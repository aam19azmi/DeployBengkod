# 🚀 Obesity Prediction API (Backend)

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-Latest-009688?style=for-the-badge&logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED?style=for-the-badge&logo=docker)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn)

Repository ini berisi **Backend REST API** untuk sistem Prediksi Tingkat Obesitas. Dibangun menggunakan **FastAPI** dan dikemas menggunakan **Docker**, API ini melayani permintaan prediksi dari frontend dengan memproses data input, menjalankannya pada model Machine Learning, dan mengembalikan hasil diagnosis dalam Bahasa Indonesia.

👉 **Frontend Repository:** [DeployBengkodWebsite](https://github.com/aam19azmi/DeployBengkodWebsite)
👉 **Live Demo:** [Lihat Website][(.sourcecodejournal.dev)](https://weightdetection.sourcecodejournal.dev/)

---

## 🌟 Fitur Utama

* **⚡ High Performance:** Menggunakan FastAPI dan Uvicorn untuk respons super cepat.
* **🐳 Dockerized:** Siap deploy di mana saja (Railway, AWS, GCP) menggunakan container.
* **🛡️ Secure CORS:** Dikonfigurasi hanya menerima request dari domain frontend resmi (`aam19azmi.github.io`).
* **🧠 Smart Preprocessing:** Otomatis menangani normalisasi teks (case-insensitive), One-Hot Encoding, dan penskalaan data sebelum masuk ke model.
* **🇮🇩 Localized Output:** Hasil prediksi langsung diterjemahkan ke Bahasa Indonesia (contoh: *Obesity_Type_I* $\rightarrow$ *Obesitas Tipe I*).

---

## 🛠️ Tech Stack

* **Framework:** FastAPI
* **ML Library:** Scikit-learn, Pandas, Joblib
* **Containerization:** Docker
* **Server:** Uvicorn

---

## 📂 Struktur File

| File | Deskripsi |
| :--- | :--- |
| `Dockerfile` | Konfigurasi image Docker (Python 3.10-slim) |
| `app.py` | Logika utama API, preprocessing, dan endpoint |
| `requirements.txt` | Daftar dependensi Python |
| `*.pkl` | File artifak model ML (Model, Scaler, Encoders) |

---

## 🚀 Instalasi & Menjalankan (Local)

Pastikan Anda memiliki Python 3.10+ terinstall.

### 1. Clone Repository
```bash
git clone [https://github.com/aam19azmi/DeployBengkod.git](https://github.com/aam19azmi/DeployBengkod.git)
cd DeployBengkod
2. Install Dependencies
Bash

pip install -r requirements.txt
3. Jalankan Server
Bash

uvicorn app:app --reload
Server akan berjalan di http://127.0.0.1:8000.

🐳 Menjalankan dengan Docker
Jika Anda tidak ingin menginstall Python secara manual, gunakan Docker:

1. Build Image
Bash

docker build -t obesity-api .
2. Run Container
Bash

docker run -p 8080:8080 obesity-api
Server akan berjalan di http://127.0.0.1:8080.

📖 Dokumentasi API
Endpoint: /predict
Method: POST

Description: Menerima data kesehatan dan mengembalikan status berat badan.

Contoh Request Body (JSON)
JSON

{
  "Gender": "Male",
  "Age": 24,
  "Height": 1.75,
  "Weight": 80,
  "family_history_with_overweight": "yes",
  "FAVC": "yes",
  "FCVC": 2.0,
  "NCP": 3.0,
  "CAEC": "Sometimes",
  "SMOKE": "no",
  "CH2O": 2.0,
  "SCC": "no",
  "FAF": 1.0,
  "TUE": 1.0,
  "CALC": "Sometimes",
  "MTRANS": "Public_Transportation"
}
Contoh Response (JSON)
JSON

{
  "prediction": "Kelebihan Berat Badan Tingkat I"
}
☁️ Deployment (Railway)
Repository ini dirancang untuk deployment otomatis di Railway.

Connect GitHub Repository ke Railway.

Railway akan otomatis mendeteksi Dockerfile.

Set variabel port (jika diperlukan), namun Dockerfile sudah mengatur expose port 8080.

Developed by Azmi Jalaluddin Amron


### Apa yang saya sesuaikan dari kode kamu?

1.  **Docker Port:** Di `Dockerfile`, kamu menggunakan `CMD` dengan port **8080**. Jadi di panduan Docker saya tulis port 8080. Sedangkan untuk run manual biasa (`uvicorn` tanpa docker), defaultnya 8000.
2.  **CORS Warning:** Saya menyebutkan bahwa CORS di-set ke `aam19azmi.github.io`. Ini penting agar orang yang mencoba run lokal tahu kenapa mereka mungkin tidak bisa akses dari frontend lokal (`localhost`) kecuali mereka mengubah `app.py` dulu.
3.  **Input Schema:** Saya menyesuaikan contoh JSON dengan class `InputData` di `app.py`. Perhatikan bahwa `Height` dan `Weight` adalah `float`, dan input teks seperti `yes`/`no` sudah di-handle oleh fungsi `preprocess` kamu (jadi user tidak harus mengetik huruf besar/kecil dengan presisi).
4.  **Output:** Saya menampilkan contoh output dalam Bahasa Indonesia sesuai fungsi `translate_label`.
