from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ML imports
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from statsmodels.tsa.holtwinters import ExponentialSmoothing

import sys
sys.path.append('..')
from database import get_products, get_movements, get_product_movements

router = APIRouter()

# ============================================
# MODELOS Y SCHEMAS
# ============================================

class PredictionRequest(BaseModel):
    codigo: str
    dias_futuro: Optional[int] = 30

class DemandPredictionRequest(BaseModel):
    dias_historico: Optional[int] = 90
    dias_futuro: Optional[int] = 7

# ============================================
# HOLT-WINTERS - Predicción de agotamiento (reemplaza LSTM)
# ============================================

def prepare_time_series(movements_df):
    """Preparar serie temporal para Holt-Winters"""
    if movements_df.empty:
        return None
    
    # Agrupar por día y sumar salidas
    daily = movements_df[movements_df['tipo'] == 'salida'].copy()
    if daily.empty:
        return None
    
    daily['date'] = daily['created_at'].dt.date
    daily_consumption = daily.groupby('date')['cantidad'].sum().reset_index()
    daily_consumption.columns = ['date', 'consumption']
    
    if len(daily_consumption) < 7:
        return None
    
    # Crear serie temporal completa (rellenar días sin movimientos)
    date_range = pd.date_range(
        start=daily_consumption['date'].min(),
        end=daily_consumption['date'].max(),
        freq='D'
    )
    
    full_series = pd.DataFrame({'date': date_range})
    full_series['date'] = full_series['date'].dt.date
    full_series = full_series.merge(daily_consumption, on='date', how='left')
    full_series['consumption'] = full_series['consumption'].fillna(0)
    
    return full_series['consumption'].values

@router.post("/stock-depletion/{codigo}")
async def predict_stock_depletion(codigo: str, dias_futuro: int = 30):
    """
    Holt-Winters: Predice cuándo se va a agotar un producto
    """
    try:
        # Obtener datos
        products = get_products()
        product = products[products['codigo'] == codigo]
        
        if product.empty:
            raise HTTPException(status_code=404, detail="Producto no encontrado")
        
        current_stock = int(product['stock'].values[0])
        movements = get_product_movements(codigo)
        
        if movements.empty or len(movements[movements['tipo'] == 'salida']) < 7:
            # No hay suficientes datos, usar promedio simple
            salidas = movements[movements['tipo'] == 'salida']
            if not salidas.empty:
                days_span = (salidas['created_at'].max() - salidas['created_at'].min()).days or 1
                avg_daily = salidas['cantidad'].sum() / days_span
            else:
                avg_daily = 0
                
            if avg_daily > 0:
                days_until_depletion = int(current_stock / avg_daily)
            else:
                days_until_depletion = None
            
            return {
                "codigo": codigo,
                "stock_actual": current_stock,
                "dias_hasta_agotamiento": days_until_depletion,
                "fecha_estimada_agotamiento": (datetime.now() + timedelta(days=days_until_depletion)).isoformat() if days_until_depletion else None,
                "modelo": "promedio_simple",
                "confianza": 0.5,
                "mensaje": "Pocos datos históricos, usando promedio simple"
            }
        
        # Preparar serie temporal
        time_series = prepare_time_series(movements)
        
        if time_series is None or len(time_series) < 14:
            # Fallback a promedio
            salidas = movements[movements['tipo'] == 'salida']
            days_span = (salidas['created_at'].max() - salidas['created_at'].min()).days or 1
            avg_daily = salidas['cantidad'].sum() / days_span
            days_until_depletion = int(current_stock / avg_daily) if avg_daily > 0 else None
            
            return {
                "codigo": codigo,
                "stock_actual": current_stock,
                "dias_hasta_agotamiento": days_until_depletion,
                "fecha_estimada_agotamiento": (datetime.now() + timedelta(days=days_until_depletion)).isoformat() if days_until_depletion else None,
                "modelo": "promedio_simple",
                "confianza": 0.6
            }
        
        # Aplicar Holt-Winters
        try:
            model = ExponentialSmoothing(
                time_series,
                trend='add',
                seasonal=None,  # Sin estacionalidad si hay pocos datos
                initialization_method='estimated'
            )
            fitted_model = model.fit()
            
            # Predecir consumo futuro
            future_consumption = fitted_model.forecast(dias_futuro)
            future_consumption = np.maximum(future_consumption, 0)  # No puede ser negativo
            
        except Exception:
            # Si falla Holt-Winters, usar promedio móvil
            avg_daily = np.mean(time_series[-14:])
            future_consumption = np.full(dias_futuro, avg_daily)
        
        # Calcular días hasta agotamiento
        cumulative_consumption = np.cumsum(future_consumption)
        days_until_depletion = None
        
        for i, total in enumerate(cumulative_consumption):
            if total >= current_stock:
                days_until_depletion = i + 1
                break
        
        if days_until_depletion is None and len(cumulative_consumption) > 0:
            avg_daily = np.mean(future_consumption)
            if avg_daily > 0:
                days_until_depletion = int(current_stock / avg_daily)
        
        return {
            "codigo": codigo,
            "stock_actual": current_stock,
            "dias_hasta_agotamiento": days_until_depletion,
            "fecha_estimada_agotamiento": (datetime.now() + timedelta(days=days_until_depletion)).isoformat() if days_until_depletion else None,
            "consumo_diario_predicho": float(np.mean(future_consumption)),
            "prediccion_proximos_dias": [float(x) for x in future_consumption[:7]],
            "modelo": "Holt-Winters",
            "confianza": 0.80
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================
# XGBoost - Predicción de demanda
# ============================================

@router.post("/demand-forecast")
async def predict_demand(request: DemandPredictionRequest):
    """
    XGBoost: Predice la demanda de todos los productos para la próxima semana
    """
    try:
        products = get_products()
        movements = get_movements()
        
        if movements.empty:
            raise HTTPException(status_code=400, detail="No hay movimientos registrados")
        
        # Filtrar solo salidas (consumo)
        sales = movements[movements['tipo'] == 'salida'].copy()
        
        if sales.empty:
            raise HTTPException(status_code=400, detail="No hay datos de salidas")
        
        predictions = []
        
        for _, product in products.iterrows():
            codigo = product['codigo']
            product_sales = sales[sales['codigo'] == codigo].copy()
            
            if len(product_sales) < 5:
                # Pocos datos, predicción simple
                avg_weekly = product_sales['cantidad'].sum() / max(1, len(product_sales)) * 7
                predictions.append({
                    "codigo": codigo,
                    "descripcion": product['descripcion'],
                    "demanda_predicha_semana": float(avg_weekly),
                    "modelo": "promedio",
                    "confianza": 0.4
                })
                continue
            
            # Preparar features para XGBoost
            product_sales['dayofweek'] = product_sales['created_at'].dt.dayofweek
            product_sales['month'] = product_sales['created_at'].dt.month
            product_sales['day'] = product_sales['created_at'].dt.day
            
            # Agrupar por día
            daily_sales = product_sales.groupby(
                product_sales['created_at'].dt.date
            ).agg({
                'cantidad': 'sum',
                'dayofweek': 'first',
                'month': 'first'
            }).reset_index()
            
            if len(daily_sales) < 7:
                avg_weekly = daily_sales['cantidad'].sum()
                predictions.append({
                    "codigo": codigo,
                    "descripcion": product['descripcion'],
                    "demanda_predicha_semana": float(avg_weekly),
                    "modelo": "promedio",
                    "confianza": 0.5
                })
                continue
            
            # Features y target
            X = daily_sales[['dayofweek', 'month']].values
            y = daily_sales['cantidad'].values
            
            # Entrenar XGBoost
            model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                objective='reg:squarederror'
            )
            model.fit(X, y)
            
            # Predecir próximos 7 días
            future_dates = pd.date_range(start=datetime.now(), periods=7, freq='D')
            future_features = np.array([
                [d.dayofweek, d.month] for d in future_dates
            ])
            
            future_demand = model.predict(future_features)
            weekly_demand = float(np.sum(future_demand))
            
            predictions.append({
                "codigo": codigo,
                "descripcion": product['descripcion'],
                "demanda_predicha_semana": weekly_demand,
                "demanda_por_dia": [float(x) for x in future_demand],
                "modelo": "XGBoost",
                "confianza": 0.75
            })
        
        # Ordenar por demanda predicha
        predictions.sort(key=lambda x: x['demanda_predicha_semana'], reverse=True)
        
        return {
            "periodo": f"Próximos 7 días desde {datetime.now().strftime('%Y-%m-%d')}",
            "predicciones": predictions,
            "total_productos_analizados": len(predictions)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/summary")
async def get_predictions_summary():
    """Resumen de predicciones para el dashboard"""
    try:
        products = get_products()
        movements = get_movements()
        
        if products.empty:
            return {"productos_criticos": [], "total_analizado": 0}
        
        critical_products = []
        
        for _, product in products.iterrows():
            codigo = product['codigo']
            stock = product['stock']
            stock_minimo = product.get('stock_minimo', 5)
            
            product_movements = movements[movements['codigo'] == codigo]
            salidas = product_movements[product_movements['tipo'] == 'salida']
            
            if not salidas.empty:
                # Calcular consumo promedio diario
                days_span = (salidas['created_at'].max() - salidas['created_at'].min()).days or 1
                total_consumed = salidas['cantidad'].sum()
                daily_rate = total_consumed / days_span
                
                if daily_rate > 0:
                    days_remaining = stock / daily_rate
                    
                    if days_remaining < 14 or stock <= stock_minimo:
                        critical_products.append({
                            "codigo": codigo,
                            "descripcion": product['descripcion'],
                            "stock_actual": int(stock),
                            "stock_minimo": int(stock_minimo),
                            "consumo_diario": round(daily_rate, 2),
                            "dias_restantes": round(days_remaining, 1),
                            "urgencia": "critica" if days_remaining < 3 or stock == 0 else "media" if days_remaining < 7 else "baja"
                        })
        
        # Ordenar por días restantes
        critical_products.sort(key=lambda x: x['dias_restantes'])
        
        return {
            "productos_criticos": critical_products[:10],
            "total_analizado": len(products),
            "total_criticos": len(critical_products)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))