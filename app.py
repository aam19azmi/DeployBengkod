import pandas as pd
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from sklearn.preprocessing import LabelEncoder
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
    Gender: str
    Age: float
    Height: float
    Weight: float
    family_history_with_overweight: int
    FAVC: int
    FCVC: int
    NCP: float
    CAEC: str
    SMOKE: int
    CH2O: float
    SCC: int
    FAF: float
    TUE: int
    CALC: str
    MTRANS: str

def preprocess(input_data: InputData):
    data_dict = input_data.dict()
    df = pd.DataFrame([data_dict])

    # Ubah 0/1 ke 'no'/'yes' untuk kolom tertentu
    yes_no_columns = ['family_history_with_overweight', 'FAVC', 'SMOKE', 'SCC']
    for col in yes_no_columns:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: 'yes' if x in [1, '1', True] else 'no' if x in [0, '0', False] else x)

    # Replace "No" dengan "no" khusus untuk kolom CAEC dan CALC
    for col in ['CAEC', 'CALC']:
        if col in df.columns:
            df[col] = df[col].replace("No", "no")

    # Normalisasi kapitalisasi string input untuk kolom lain
    for col in ['Gender', 'CAEC', 'CALC', 'MTRANS']:
        if col in df.columns:
            df[col] = df[col].str.strip().str.title()

    # Label encoding seperti biasa
    for col, le in label_encoders.items():
        if col in df.columns:
            # Pastikan tipe string jika kelasnya string
            if le.classes_.dtype.kind in {'U', 'S', 'O'}:
                df[col] = df[col].astype(str)
            else:
                df[col] = df[col].astype(int)

            unseen_labels = set(df[col].unique()) - set(le.classes_)
            if unseen_labels:
                raise HTTPException(
                    status_code=400,
                    detail=f"Kolom '{col}' mengandung label tidak dikenal: {unseen_labels}"
                )
            df[col] = le.transform(df[col])

    # One-hot encoding dan lain-lain
    df = pd.get_dummies(df)
    for col in final_feature_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[final_feature_columns]
    df = df[selected_features]

    X_scaled = scaler.transform(df)
    return X_scaled

@app.post("/predict")
def predict(input_data: InputData):
    try:
        X = preprocess(input_data)
        pred = model.predict(X)
        return {"prediction": pred[0]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
