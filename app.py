import os
import pandas as pd
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from sklearn.preprocessing import LabelEncoder
import socket
import uvicorn
import joblib  # <-- Ganti pickle dengan joblib

app = FastAPI(title="Obesity Prediction API")

# Tambahkan konfigurasi CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://aam19azmi.github.io"],  # Ganti * jika perlu
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load artifacts menggunakan joblib
label_encoders = joblib.load('label_encoders.pkl')
final_feature_columns = joblib.load('final_feature_columns.pkl')
selected_features = joblib.load('selected_features.pkl')
scaler = joblib.load('scaler.pkl')
model = joblib.load('model.pkl')

# Definisikan input data schema
class InputData(BaseModel):
    Gender: str  # {Female,Male}
    Age: float
    Height: float
    Weight: float
    family_history_with_overweight: str  # {yes,no}
    FAVC: str  # {yes,no}
    FCVC: float
    NCP: float
    CAEC: str  # {no,Sometimes,Frequently,Always}
    SMOKE: str  # {yes,no}
    CH2O: float
    SCC: str  # {yes,no}
    FAF: float
    TUE: float
    CALC: str  # {no,Sometimes,Frequently,Always}
    MTRANS: str  # {Automobile,Motorbike,Bike,Public_Transportation,Walking}

def preprocess(input_data: InputData):
    data_dict = input_data.dict()
    df = pd.DataFrame([data_dict])

    # 1️⃣ Normalisasi format teks
    for col in ['Gender', 'MTRANS']:
        df[col] = df[col].str.strip().str.title()  # Capitalize

    for col in ['family_history_with_overweight', 'FAVC', 'SMOKE', 'SCC']:
        df[col] = df[col].str.strip().str.lower()  # Lowercase

    for col in ['CAEC', 'CALC']:
        df[col] = df[col].str.strip().apply(lambda x: 'no' if x.lower() == 'no' else x.title())

    # 2️⃣ Label Encoding untuk kolom yang memang dilatih dengan encoder
    for col, le in label_encoders.items():
        if col in df.columns:
            unseen_labels = set(df[col].unique()) - set(le.classes_)
            if unseen_labels:
                raise HTTPException(
                    status_code=400,
                    detail=f"Kolom '{col}' mengandung label tidak dikenal: {unseen_labels}"
                )
            df[col] = le.transform(df[col].astype(str))

    # 3️⃣ One-Hot Encoding untuk fitur nominal (misalnya Gender, MTRANS)
    df = pd.get_dummies(df)

    # 4️⃣ Pastikan semua kolom dari training tersedia
    for col in final_feature_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[final_feature_columns]  # Susun sesuai urutan yang benar

    # 5️⃣ Scaling seluruh fitur sesuai training
    X_scaled = scaler.transform(df)
    df_scaled = pd.DataFrame(X_scaled, columns=final_feature_columns)

    # 6️⃣ Ambil hanya fitur yang dipilih saat training model akhir
    df_selected = df_scaled[selected_features]

    return df_selected

def translate_label(label):
    mapping = {
        "Insufficient_Weight": "Berat Badan Kurang",
        "Normal_Weight": "Berat Badan Normal",
        "Overweight_Level_I": "Kelebihan Berat Badan Tingkat I",
        "Overweight_Level_II": "Kelebihan Berat Badan Tingkat II",
        "Obesity_Type_I": "Obesitas Tipe I",
        "Obesity_Type_II": "Obesitas Tipe II",
        "Obesity_Type_III": "Obesitas Tipe III"
    }
    return mapping.get(label, label).replace("_", " ").title()

@app.post("/predict")
def predict(input_data: InputData):
    try:
        X = preprocess(input_data)
        print("Input ke model (X):", X)
        pred = model.predict(X)
        raw_label = label_encoders['NObeyesdad'].inverse_transform([pred[0]])[0]
        translated_label = translate_label(raw_label)
        return {"prediction": translated_label}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # 8000 sebagai fallback jika dijalankan lokal
    uvicorn.run(app, host="0.0.0.0", port=port)
