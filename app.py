import os
import pandas as pd
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.preprocessing import LabelEncoder
import uvicorn
import joblib

# Inisialisasi logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Obesity Prediction API")

# Konfigurasi CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://aam19azmi.github.io"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load artifacts
label_encoders = joblib.load('label_encoders.pkl')
features = joblib.load('features.pkl')
selected_features = joblib.load('selected_features.pkl')
scaler = joblib.load('scaler.pkl')
model = joblib.load('model.pkl')

# Input schema
class InputData(BaseModel):
    Gender: str
    Age: float
    Height: float
    Weight: float
    family_history_with_overweight: str
    FAVC: str
    FCVC: float
    NCP: float
    CAEC: str
    SMOKE: str
    CH2O: float
    SCC: str
    FAF: float
    TUE: float
    CALC: str
    MTRANS: str

def preprocess(input_data: InputData):
    data_dict = input_data.dict()
    df = pd.DataFrame([data_dict])
    logger.info("DEBUG: Kolom input awal: %s", df.columns.tolist())
    logger.info("DEBUG: Data awal:\n%s", df)

    # Normalisasi teks
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
    logger.info("DEBUG: Data setelah normalisasi teks:\n%s", df)

    # Drop kolom tidak relevan
    base_features = set(col.split('_')[0] for col in features)
    logger.info("DEBUG: base_features yang diperbolehkan: %s", base_features)
    df = df[[col for col in df.columns if col in base_features]]
    logger.info("DEBUG: Kolom setelah filter base_features: %s", df.columns.tolist())

    # Label encoding
    for col, le in label_encoders.items():
        if col in df.columns:
            unseen = set(df[col].astype(str)) - set(le.classes_)
            if unseen:
                raise HTTPException(status_code=400, detail=f"Unseen label in '{col}': {unseen}")
            df[col] = le.transform(df[col].astype(str))
    logger.info("DEBUG: Kolom setelah label encoding: %s", df.columns.tolist())
    logger.info("DEBUG: Data setelah label encoding:\n%s", df)

    # One-hot encoding
    dummy_cols = [col for col in df.select_dtypes(include=['object']).columns if col not in label_encoders.keys()]
    logger.info("DEBUG: Kolom untuk one-hot encoding: %s", dummy_cols)
    if dummy_cols:
        df = pd.get_dummies(df, columns=dummy_cols)
    logger.info("DEBUG: Kolom setelah one-hot encoding: %s", df.columns.tolist())
    logger.info("DEBUG: Data setelah one-hot encoding:\n%s", df)

    # Reindex kolom
    if 'NObeyesdad' in df.columns:
        df.drop(columns=['NObeyesdad'], inplace=True)
    missing_cols = set(features) - set(df.columns)
    extra_cols = set(df.columns) - set(features)
    logger.info("DEBUG: Kolom hilang sebelum reindex: %s", missing_cols)
    logger.info("DEBUG: Kolom ekstra sebelum reindex: %s", extra_cols)
    df = df.reindex(columns=features, fill_value=0)
    logger.info("DEBUG: Kolom setelah reindex: %s", df.columns.tolist())

    # Seleksi fitur
    df_selected = df[selected_features]
    logger.info("DEBUG: Data setelah seleksi fitur:\n%s", df_selected)

    # Scaling
    X_scaled = scaler.transform(df_selected)
    df_scaled = pd.DataFrame(X_scaled, columns=selected_features)
    logger.info("DEBUG: Data setelah scaling:\n%s", df_scaled)

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
    return mapping.get(label, label).title()

@app.post("/predict")
def predict(input_data: InputData):
    try:
        X = preprocess(input_data)
        logger.info("DEBUG: Input ke model (X):\n%s", X)
        pred = model.predict(X)
        raw_label = label_encoders['NObeyesdad'].inverse_transform([pred[0]])[0]
        translated_label = translate_label(raw_label)
        logger.info("DEBUG: Hasil Prediksi: %s", pred)
        logger.info("DEBUG: Hasil Prediksi: %s", translated_label)
        return {"prediction": translated_label}
    except Exception as e:
        logger.error("ERROR saat prediksi: %s", str(e))
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
