# Obesity Prediction API

API ini menyediakan layanan prediksi tingkat obesitas berdasarkan data kebiasaan makan dan kondisi fisik individu. Model sudah dilatih menggunakan dataset obesitas dari Meksiko, Peru, dan Kolombia.

---

## Isi Proyek

| File                        | Keterangan                          |
| --------------------------- | ---------------------------------- |
| `label_encoders.pkl`        | Encoder untuk fitur kategorikal    |
| `final_feature_columns.pkl` | Fitur hasil one-hot encoding       |
| `selected_features.pkl`     | 10 fitur terbaik hasil seleksi     |
| `scaler.pkl`                | Objek StandardScaler untuk scaling |
| `model.pkl`                 | Model ML terbaik untuk prediksi    |
| `app.py`                    | API FastAPI untuk deployment model |
| `requirements.txt`          | Daftar dependencies Python         |

---

## Dataset

Dataset berasal dari penelitian obesitas pada individu dari Meksiko, Peru, dan Kolombia dengan 17 fitur input dan target klasifikasi obesitas dengan 7 kategori. Dataset sebagian besar disintesis menggunakan SMOTE.

---

## Instalasi dan Jalankan Lokal

1. Clone repository ini:

```bash
git clone https://github.com/username/obesity-prediction-api.git
cd obesity-prediction-api

2. Buat virtual environment (opsional):

python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows

3. Install dependencies:
pip install -r requirements.txt

4. Jalankan API:
uvicorn app:app --reload
API akan berjalan di http://127.0.0.1:8000


## Cara Menggunakan API
## Endpoint utama untuk prediksi:

POST /predict
Format Input (JSON)
## Contoh JSON input sesuai dengan atribut:

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
## Contoh Request dengan curl

curl -X POST "http://127.0.0.1:8000/predict" \
-H "Content-Type: application/json" \
-d '{
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
}'
## Contoh Response:

{
  "prediction": "Normal Weight"
}
