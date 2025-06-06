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
    print("DEBUG: Kolom input awal:", df.columns.tolist())
    print("DEBUG: Data awal:\n", df)

    # 1. Normalisasi teks pada kolom kategori
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].str.strip().str.title()
    if 'family_history_with_overweight' in df.columns:
        df['family_history_with_overweight'] = df['family_history_with_overweight'].str.strip().str.lower()
    if 'FAVC' in df.columns:
        df['FAVC'] = df['FAVC'].str.strip().str.lower()
    if 'CAEC' in df.columns:
        df['CAEC'] = df['CAEC'].str.strip().apply(lambda x: 'no' if x.lower() == 'no' else x.title())
    if 'CALC' in df.columns:
        df['CALC'] = df['CALC'].str.strip().apply(lambda x: 'no' if x.lower() == 'no' else x.title())
    print("DEBUG: Data setelah normalisasi teks:\n", df)

    # 2. Drop kolom yang tidak termasuk base_features
    base_features = set(col.split('_')[0] for col in final_feature_columns)
    print("DEBUG: base_features yang diperbolehkan:", base_features)
    df = df[[col for col in df.columns if col in base_features]]
    print("DEBUG: Kolom setelah filter base_features:", df.columns.tolist())

    # 3. Label encode fitur kategori yang perlu (misal Gender, MTRANS)
    for col, le in label_encoders.items():
        if col in df.columns:
            unseen = set(df[col].astype(str)) - set(le.classes_)
            if unseen:
                raise HTTPException(status_code=400, detail=f"Unseen label in '{col}': {unseen}")
            df[col] = le.transform(df[col].astype(str))
    print("DEBUG: Kolom setelah label encoding:", df.columns.tolist())
    print("DEBUG: Data setelah label encoding:\n", df)

    # 4. One-hot encoding hanya untuk fitur kategorikal yang **tidak di-label encode** 
    dummy_cols = [col for col in df.select_dtypes(include=['object']).columns if col not in label_encoders.keys()]
    print("DEBUG: Kolom untuk one-hot encoding:", dummy_cols)
    if dummy_cols:
        df = pd.get_dummies(df, columns=dummy_cols)
    print("DEBUG: Kolom setelah one-hot encoding:", df.columns.tolist())
    print("DEBUG: Data setelah one-hot encoding:\n", df)

    # 5. Reindex agar kolom dan urutan sesuai final_feature_columns
    missing_cols = set(final_feature_columns) - set(df.columns)
    extra_cols = set(df.columns) - set(final_feature_columns)
    print("DEBUG: Kolom hilang sebelum reindex:", missing_cols)
    print("DEBUG: Kolom ekstra sebelum reindex:", extra_cols)
    df = df.reindex(columns=final_feature_columns, fill_value=0)
    print("DEBUG: Kolom setelah reindex:", df.columns.tolist())

    # 6. Ambil subset fitur yang dipakai model akhir (jika ada seleksi fitur)
    df_selected = df[selected_features]
    print("DEBUG: Data setelah seleksi fitur:\n", df_selected)

    # 7. Scaling hanya pada fitur yang dipilih
    X_scaled = scaler.transform(df_selected)
    df_scaled = pd.DataFrame(X_scaled, columns=selected_features)
    print("DEBUG: Data setelah scaling:\n", df_scaled)

    return df_scaled

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
