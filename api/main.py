from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from typing import List

app = FastAPI(title="Wine Quality Prediction API",
              description="API для предсказания качества вина с использованием логистической регрессии",
              version="1.0")

# Конфигурация (аналогично вашей)
config = {
    "random_state": 42, 
    "data": {
        "test_size": 0.2,
        "binary_classification": True, 
        "quality_threshold": 6,
    },
    "logistic_regression": {
        "max_iter": 1000, 
        "penalty": "l2",  
        "C": 1.0,  
        "class_weight": "balanced" 
    }
}

# Загрузка и подготовка данных
df = pd.read_csv('winequality-red.csv')
X = df.drop('quality', axis=1).values
y = df['quality'].values

if config["data"]["binary_classification"]:
    y = np.where(y >= config["data"]["quality_threshold"], 1, 0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Обучение модели
model = LogisticRegression(
    max_iter=config["logistic_regression"]["max_iter"],
    penalty=config["logistic_regression"]["penalty"],
    C=config["logistic_regression"]["C"],
    random_state=config["random_state"],
    class_weight=config["logistic_regression"]["class_weight"]
)
model.fit(X_scaled, y)

# Сохранение модели и scaler (для демонстрации)
joblib.dump(model, 'wine_quality_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Pydantic модель для валидации входных данных
class WineFeatures(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float

# Эндпоинты API
@app.get("/healthcheck")
def healthcheck():
    return {
        "status": "OK",
        "model_loaded": True,
        "model_type": "Logistic Regression",
        "binary_classification": config["data"]["binary_classification"]
    }

@app.get("/model-info")
def model_info():
    return {
        "model_name": "Wine Quality Classifier",
        "algorithm": "Logistic Regression",
        "classes": ["bad", "good"] if config["data"]["binary_classification"] else list(range(3, 9)),
        "input_features": [
            "fixed_acidity", "volatile_acidity", "citric_acid",
            "residual_sugar", "chlorides", "free_sulfur_dioxide",
            "total_sulfur_dioxide", "density", "pH",
            "sulphates", "alcohol"
        ],
        "config": config
    }

@app.post("/predict")
def predict(features: WineFeatures):
    try:
        # Преобразуем входные данные в массив numpy
        input_data = np.array([
            features.fixed_acidity,
            features.volatile_acidity,
            features.citric_acid,
            features.residual_sugar,
            features.chlorides,
            features.free_sulfur_dioxide,
            features.total_sulfur_dioxide,
            features.density,
            features.pH,
            features.sulphates,
            features.alcohol
        ]).reshape(1, -1)
        
        # Масштабируем данные
        input_scaled = scaler.transform(input_data)
        
        # Делаем предсказание
        prediction = model.predict(input_scaled)
        probabilities = model.predict_proba(input_scaled)
        
        # Формируем ответ
        if config["data"]["binary_classification"]:
            result = {
                "prediction": "good" if prediction[0] == 1 else "bad",
                "probability": float(probabilities[0][1] if prediction[0] == 1 else probabilities[0][0]),
                "threshold": config["data"]["quality_threshold"]
            }
        else:
            result = {
                "prediction": int(prediction[0]),
                "probabilities": {str(i): float(prob) for i, prob in enumerate(probabilities[0])}
            }
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/batch-predict")
def batch_predict(features_list: List[WineFeatures]):
    try:
        # Преобразуем список объектов в DataFrame
        input_data = pd.DataFrame([f.dict() for f in features_list])
        
        # Масштабируем данные
        input_scaled = scaler.transform(input_data)
        
        # Делаем предсказания
        predictions = model.predict(input_scaled)
        probabilities = model.predict_proba(input_scaled)
        
        # Формируем ответ
        results = []
        for i in range(len(predictions)):
            if config["data"]["binary_classification"]:
                results.append({
                    "prediction": "good" if predictions[i] == 1 else "bad",
                    "probability": float(probabilities[i][1] if predictions[i] == 1 else probabilities[i][0])
                })
            else:
                results.append({
                    "prediction": int(predictions[i]),
                    "probabilities": {str(j): float(prob) for j, prob in enumerate(probabilities[i])}
                })
        
        return {"predictions": results}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))