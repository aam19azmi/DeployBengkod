import pickle
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware  # ⬅️ Tambahkan ini
from pydantic import BaseModel
from typing import Optional
import uvicorn

app = FastAPI(title="Obesity Prediction API")

# Tambahkan konfigurasi CORS ⬇️
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://aam19azmi.github.io"],  # Ganti * jika perlu
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load artifacts
with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)
with open('final_feature_columns.pkl', 'rb') as f:
    final_feature_columns = pickle.load(f)
with open('selected_features.pkl', 'rb') as f:
    selected_features = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

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

    for col, le in label_encoders.items():
        if col in df.columns:
            df[col] = le.transform(df[col])

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
    X = preprocess(input_data)
    pred = model.predict(X)
    return {"prediction": pred[0]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
