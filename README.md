# 🍎 Obesity Prediction API

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-009688?style=for-the-badge&logo=fastapi)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn)
![Railway](https://img.shields.io/badge/Railway-Deployment-0B0D0E?style=for-the-badge&logo=railway)

**Obesity Prediction API** adalah layanan backend berbasis Machine Learning yang bertugas memprediksi tingkat risiko obesitas seseorang. API ini menerima input berupa data fisik dan kebiasaan gaya hidup, lalu mengembalikan hasil klasifikasi kesehatan.

Proyek ini merupakan bagian **Backend** dari sistem kalkulator kesehatan.
👉 **Lihat Frontend Project di sini:** [Obesity Calculator Frontend](https://github.com/aam19azmi/DeployBengkodWebsite)

---

## 📂 Struktur Proyek

Berikut adalah file-file utama yang digunakan dalam sistem ini:

| File | Deskripsi |
| :--- | :--- |
| `app.py` | Entry point utama aplikasi (FastAPI) |
| `model.pkl` | Model Machine Learning yang sudah dilatih (Pickle format) |
| `scaler.pkl` | Objek StandardScaler untuk normalisasi data numerik |
| `label_encoders.pkl` | Encoder untuk mengubah data kategori menjadi angka |
| `selected_features.pkl` | Daftar 10 fitur prioritas hasil seleksi fitur |
| `requirements.txt` | Daftar pustaka (library) Python yang dibutuhkan |

---

## 🧠 Dataset & Fitur (Data Dictionary)

Model ini dilatih menggunakan dataset obesitas dari negara Meksiko, Peru, dan Kolombia. Berikut adalah penjelasan parameter input yang dibutuhkan API:

### Data Fisik
* **Gender**: `Male` / `Female`
* **Age**: Usia (Tahun)
* **Height**: Tinggi Badan (cm)
* **Weight**: Berat Badan (kg)

### Kebiasaan Makan & Gaya Hidup
* **family_history_with_overweight**: Riwayat keluarga obesitas (`1` = Ya, `0` = Tidak)
* **FAVC**: Sering makan makanan tinggi kalori (`1` = Ya, `0` = Tidak)
* **FCVC**: Konsumsi sayuran (Skala `1-3`)
* **NCP**: Jumlah makan utama per hari (Skala `1-4`)
* **CAEC**: Makan di antara waktu makan (*Snacking*)
    * `no`, `Sometimes`, `Frequently`, `Always`
* **SMOKE**: Perokok (`1` = Ya, `0` = Tidak)
* **CH2O**: Konsumsi air harian (Liter, Skala `1-3`)
* **SCC**: Memantau asupan kalori (`1` = Ya, `0` = Tidak)
* **FAF**: Frekuensi aktivitas fisik (Skala `0-3`)
* **TUE**: Waktu penggunaan perangkat teknologi (Skala `0-2`)
* **CALC**: Konsumsi alkohol (`no`, `Sometimes`, `Frequently`, `Always`)
* **MTRANS**: Transportasi utama (`Public_Transportation`, `Walking`, `Automobile`, `Motorbike`, `Bike`)

---

## 🚀 Instalasi & Menjalankan Lokal

Ikuti langkah ini untuk menjalankan API di komputer Anda:

### 1. Clone Repository
```bash
git clone [https://github.com/aam19azmi/DeployBengkod.git](https://github.com/aam19azmi/DeployBengkod.git)
cd DeployBengkod
2. Buat Virtual Environment (Disarankan)
Bash

# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
3. Install Dependencies
Bash

pip install -r requirements.txt
4. Jalankan Server
Bash

uvicorn app:app --reload
API akan berjalan di http://127.0.0.1:8000.

📖 Dokumentasi API
FastAPI menyediakan dokumentasi interaktif secara otomatis. Setelah server berjalan, buka browser dan akses:

Swagger UI: http://127.0.0.1:8000/docs (Untuk mencoba API langsung)

ReDoc: http://127.0.0.1:8000/redoc

⚡ Cara Menggunakan (Endpoint)
Prediksi (POST)
URL: /predict

Contoh Body (JSON):

JSON

{
  "Gender": "Female",
  "Age": 25,
  "Height": 160,
  "Weight": 60,
  "family_history_with_overweight": 1,
  "FAVC": 1,
  "FCVC": 2,
  "NCP": 3,
  "CAEC": "Sometimes",
  "SMOKE": 0,
  "CH2O": 2,
  "SCC": 0,
  "FAF": 1,
  "TUE": 2,
  "CALC": "Sometimes",
  "MTRANS": "Public_Transportation"
}
Contoh Request via cURL:

Bash

curl -X POST "[http://127.0.0.1:8000/predict](http://127.0.0.1:8000/predict)" \
-H "Content-Type: application/json" \
-d '{
  "Gender": "Female", "Age": 25, "Height": 160, "Weight": 60,
  "family_history_with_overweight": 1, "FAVC": 1, "FCVC": 2,
  "NCP": 3, "CAEC": "Sometimes", "SMOKE": 0, "CH2O": 2,
  "SCC": 0, "FAF": 1, "TUE": 2, "CALC": "Sometimes",
  "MTRANS": "Public_Transportation"
}'
Contoh Response:

JSON

{
  "prediction": "Normal Weight"
}
☁️ Deployment
Proyek ini dikonfigurasi untuk siap di-deploy ke Railway. Pastikan file requirements.txt dan Procfile (jika diperlukan) sudah tersedia.

Developed by Azmi Jalaluddin Amron Fullstack Developer & AI Researcher
