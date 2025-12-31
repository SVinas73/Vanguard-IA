from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
import os
import json

# ML imports
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from statsmodels.tsa.holtwinters import ExponentialSmoothing

import sys
sys.path.append('..')
from database import get_products, get_movements

router = APIRouter()

# Directorio para guardar modelos
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# ============================================
# SCHEMAS
# ============================================

class TrainingStatus(BaseModel):
    status: str
    last_training: Optional[str]
    models_trained: int
    accuracy_metrics: Dict[str, Any]

class TrainingResult(BaseModel):
    success: bool
    models_trained: list
    duration_seconds: float
    metrics: Dict[str, Any]

# ============================================
# FUNCIONES DE ENTRENAMIENTO
# ============================================

def train_demand_model(products_df: pd.DataFrame, movements_df: pd.DataFrame) -> Dict[str, Any]:
    """Entrena modelo XGBoost para predicción de demanda por producto"""
    
    if movements_df.empty:
        return {"error": "No hay movimientos para entrenar"}
    
    sales = movements_df[movements_df['tipo'] == 'salida'].copy()
    if sales.empty:
        return {"error": "No hay datos de salidas"}
    
    models_saved = []
    metrics = {}
    
    for _, product in products_df.iterrows():
        codigo = product['codigo']
        product_sales = sales[sales['codigo'] == codigo].copy()
        
        if len(product_sales) < 14:  # Mínimo 2 semanas de datos
            continue
        
        # Preparar features
        product_sales['dayofweek'] = product_sales['created_at'].dt.dayofweek
        product_sales['month'] = product_sales['created_at'].dt.month
        product_sales['day'] = product_sales['created_at'].dt.day
        product_sales['week'] = product_sales['created_at'].dt.isocalendar().week
        
        # Agrupar por día
        daily_sales = product_sales.groupby(
            product_sales['created_at'].dt.date
        ).agg({
            'cantidad': 'sum',
            'dayofweek': 'first',
            'month': 'first',
            'week': 'first'
        }).reset_index()
        
        if len(daily_sales) < 14:
            continue
        
        # Features y target
        X = daily_sales[['dayofweek', 'month', 'week']].values
        y = daily_sales['cantidad'].values
        
        # Split train/test (80/20)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Entrenar XGBoost
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            objective='reg:squarederror',
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Evaluar
        if len(X_test) > 0:
            y_pred = model.predict(X_test)
            mse = np.mean((y_test - y_pred) ** 2)
            mae = np.mean(np.abs(y_test - y_pred))
            metrics[codigo] = {"mse": float(mse), "mae": float(mae)}
        
        # Guardar modelo
        model_path = os.path.join(MODELS_DIR, f"demand_{codigo}.joblib")
        joblib.dump(model, model_path)
        models_saved.append(codigo)
    
    return {
        "models_saved": models_saved,
        "total": len(models_saved),
        "metrics": metrics
    }


def train_anomaly_model(movements_df: pd.DataFrame) -> Dict[str, Any]:
    """Entrena modelo Isolation Forest para detección de anomalías"""
    
    if movements_df.empty or len(movements_df) < 50:
        return {"error": "Datos insuficientes para entrenar modelo de anomalías"}
    
    # Preparar features
    movements_df = movements_df.copy()
    movements_df['hour'] = movements_df['created_at'].dt.hour
    movements_df['dayofweek'] = movements_df['created_at'].dt.dayofweek
    movements_df['tipo_num'] = (movements_df['tipo'] == 'salida').astype(int)
    
    # Calcular estadísticas por producto
    product_stats = movements_df.groupby('codigo').agg({
        'cantidad': ['mean', 'std']
    }).reset_index()
    product_stats.columns = ['codigo', 'qty_mean', 'qty_std']
    product_stats['qty_std'] = product_stats['qty_std'].fillna(1)
    
    movements_df = movements_df.merge(product_stats, on='codigo', how='left')
    movements_df['qty_zscore'] = (
        (movements_df['cantidad'] - movements_df['qty_mean']) / 
        movements_df['qty_std'].replace(0, 1)
    )
    
    # Features para Isolation Forest
    features = movements_df[[
        'cantidad', 'hour', 'dayofweek', 'tipo_num', 'qty_zscore'
    ]].fillna(0).values
    
    # Normalizar
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Entrenar Isolation Forest
    model = IsolationForest(
        contamination=0.1,
        random_state=42,
        n_estimators=100
    )
    model.fit(features_scaled)
    
    # Guardar modelo y scaler
    joblib.dump(model, os.path.join(MODELS_DIR, "anomaly_model.joblib"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "anomaly_scaler.joblib"))
    joblib.dump(product_stats, os.path.join(MODELS_DIR, "product_stats.joblib"))
    
    return {
        "model_saved": True,
        "training_samples": len(features),
        "contamination": 0.1
    }


def train_time_series_models(products_df: pd.DataFrame, movements_df: pd.DataFrame) -> Dict[str, Any]:
    """Entrena modelos Holt-Winters para predicción de series temporales"""
    
    if movements_df.empty:
        return {"error": "No hay movimientos"}
    
    sales = movements_df[movements_df['tipo'] == 'salida'].copy()
    models_saved = []
    
    for _, product in products_df.iterrows():
        codigo = product['codigo']
        product_sales = sales[sales['codigo'] == codigo].copy()
        
        if len(product_sales) < 21:  # Mínimo 3 semanas
            continue
        
        # Crear serie temporal diaria
        product_sales['date'] = product_sales['created_at'].dt.date
        daily = product_sales.groupby('date')['cantidad'].sum().reset_index()
        
        # Rellenar días sin ventas
        date_range = pd.date_range(start=daily['date'].min(), end=daily['date'].max(), freq='D')
        full_series = pd.DataFrame({'date': date_range})
        full_series['date'] = full_series['date'].dt.date
        full_series = full_series.merge(daily, on='date', how='left')
        full_series['cantidad'] = full_series['cantidad'].fillna(0)
        
        if len(full_series) < 14:
            continue
        
        try:
            # Entrenar Holt-Winters
            model = ExponentialSmoothing(
                full_series['cantidad'].values,
                trend='add',
                seasonal=None,
                initialization_method='estimated'
            )
            fitted_model = model.fit()
            
            # Guardar parámetros del modelo
            model_params = {
                'smoothing_level': fitted_model.params.get('smoothing_level', 0),
                'smoothing_trend': fitted_model.params.get('smoothing_trend', 0),
                'last_values': full_series['cantidad'].values[-14:].tolist(),
                'fitted_at': datetime.now().isoformat()
            }
            
            params_path = os.path.join(MODELS_DIR, f"ts_{codigo}.json")
            with open(params_path, 'w') as f:
                json.dump(model_params, f)
            
            models_saved.append(codigo)
            
        except Exception as e:
            continue
    
    return {
        "models_saved": models_saved,
        "total": len(models_saved)
    }


# ============================================
# ENDPOINTS
# ============================================

@router.post("/train-all", response_model=TrainingResult)
async def train_all_models(background_tasks: BackgroundTasks):
    """
    Entrena todos los modelos de IA con los datos actuales.
    Puede ser llamado manualmente o por un cron job.
    """
    start_time = datetime.now()
    
    try:
        products = get_products()
        movements = get_movements()
        
        results = {
            "demand": {},
            "anomaly": {},
            "time_series": {}
        }
        
        # Entrenar modelo de demanda (XGBoost)
        results["demand"] = train_demand_model(products, movements)
        
        # Entrenar modelo de anomalías (Isolation Forest)
        results["anomaly"] = train_anomaly_model(movements)
        
        # Entrenar modelos de series temporales (Holt-Winters)
        results["time_series"] = train_time_series_models(products, movements)
        
        # Guardar timestamp del último entrenamiento
        training_log = {
            "last_training": datetime.now().isoformat(),
            "results": results
        }
        with open(os.path.join(MODELS_DIR, "training_log.json"), 'w') as f:
            json.dump(training_log, f, indent=2)
        
        duration = (datetime.now() - start_time).total_seconds()
        
        models_trained = []
        if results["demand"].get("total", 0) > 0:
            models_trained.append(f"Demanda ({results['demand']['total']} productos)")
        if results["anomaly"].get("model_saved"):
            models_trained.append("Anomalías")
        if results["time_series"].get("total", 0) > 0:
            models_trained.append(f"Series Temporales ({results['time_series']['total']} productos)")
        
        return TrainingResult(
            success=True,
            models_trained=models_trained,
            duration_seconds=duration,
            metrics=results
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_training_status():
    """Obtiene el estado del último entrenamiento"""
    
    log_path = os.path.join(MODELS_DIR, "training_log.json")
    
    if not os.path.exists(log_path):
        return {
            "status": "never_trained",
            "last_training": None,
            "models_available": []
        }
    
    with open(log_path, 'r') as f:
        log = json.load(f)
    
    # Listar modelos disponibles
    models_available = []
    for file in os.listdir(MODELS_DIR):
        if file.endswith('.joblib') or file.endswith('.json'):
            models_available.append(file)
    
    return {
        "status": "trained",
        "last_training": log.get("last_training"),
        "models_available": models_available,
        "results": log.get("results", {})
    }


@router.post("/schedule-nightly")
async def schedule_nightly_training():
    """
    Información sobre cómo configurar el entrenamiento nocturno.
    El cron job real se configura en Railway o con un servicio externo.
    """
    return {
        "mensaje": "Para entrenar automáticamente cada noche:",
        "opciones": [
            {
                "servicio": "Railway Cron",
                "configuracion": "Agregar RAILWAY_CRON_SCHEDULE='0 3 * * *' en variables de entorno"
            },
            {
                "servicio": "cron-job.org (gratis)",
                "configuracion": f"Crear cron job que llame POST a /api/training/train-all cada noche"
            },
            {
                "servicio": "GitHub Actions",
                "configuracion": "Crear workflow con schedule: cron: '0 3 * * *'"
            }
        ],
        "endpoint_entrenamiento": "/api/training/train-all",
        "metodo": "POST"
    }