from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ML imports
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

import sys
sys.path.append('..')
from database import get_products, get_movements

router = APIRouter()

# ============================================
# SCHEMAS
# ============================================

class AnomalyResponse(BaseModel):
    id: str
    codigo: str
    descripcion: str
    tipo: str
    cantidad: int
    fecha: str
    usuario: Optional[str]
    anomaly_score: float
    razon: str
    severidad: str

# ============================================
# ISOLATION FOREST - Detección de anomalías
# ============================================

@router.get("/detect")
async def detect_anomalies(dias: int = 30, threshold: float = -0.3):
    """
    Isolation Forest: Detecta movimientos anómalos
    
    - Cantidades inusuales (muy altas o muy bajas)
    - Horarios inusuales
    - Patrones de usuario sospechosos
    """
    try:
        movements = get_movements()
        products = get_products()
        
        if movements.empty:
            return {"anomalias": [], "total_analizado": 0}
        
        # Filtrar por período
        cutoff_date = datetime.now() - timedelta(days=dias)
        recent_movements = movements[movements['created_at'] >= cutoff_date].copy()
        
        if len(recent_movements) < 10:
            return {
                "anomalias": [],
                "total_analizado": len(recent_movements),
                "mensaje": "Muy pocos datos para detectar anomalías"
            }
        
        # Preparar features
        recent_movements['hour'] = recent_movements['created_at'].dt.hour
        recent_movements['dayofweek'] = recent_movements['created_at'].dt.dayofweek
        recent_movements['tipo_num'] = (recent_movements['tipo'] == 'salida').astype(int)
        
        # Calcular estadísticas por producto para contextualizar
        product_stats = recent_movements.groupby('codigo').agg({
            'cantidad': ['mean', 'std', 'max']
        }).reset_index()
        product_stats.columns = ['codigo', 'qty_mean', 'qty_std', 'qty_max']
        product_stats['qty_std'] = product_stats['qty_std'].fillna(1)
        
        # Merge con movimientos
        recent_movements = recent_movements.merge(product_stats, on='codigo', how='left')
        
        # Calcular z-score de cantidad
        recent_movements['qty_zscore'] = (
            (recent_movements['cantidad'] - recent_movements['qty_mean']) / 
            recent_movements['qty_std'].replace(0, 1)
        )
        
        # Features para Isolation Forest
        features = recent_movements[[
            'cantidad', 
            'hour', 
            'dayofweek', 
            'tipo_num',
            'qty_zscore'
        ]].fillna(0)
        
        # Normalizar
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Entrenar Isolation Forest
        iso_forest = IsolationForest(
            contamination=0.1,  # Esperamos ~10% de anomalías
            random_state=42,
            n_estimators=100
        )
        
        # Predecir
        predictions = iso_forest.fit_predict(features_scaled)
        scores = iso_forest.decision_function(features_scaled)
        
        recent_movements['is_anomaly'] = predictions == -1
        recent_movements['anomaly_score'] = scores
        
        # Filtrar anomalías
        anomalies = recent_movements[recent_movements['anomaly_score'] < threshold].copy()
        
        # Preparar respuesta
        anomaly_list = []
        
        for _, row in anomalies.iterrows():
            # Determinar razón de la anomalía
            razones = []
            
            if abs(row['qty_zscore']) > 2:
                if row['qty_zscore'] > 0:
                    razones.append(f"Cantidad muy alta ({row['cantidad']} vs promedio {row['qty_mean']:.1f})")
                else:
                    razones.append(f"Cantidad muy baja ({row['cantidad']} vs promedio {row['qty_mean']:.1f})")
            
            if row['hour'] < 6 or row['hour'] > 22:
                razones.append(f"Horario inusual ({row['hour']}:00)")
            
            if row['dayofweek'] in [5, 6]:  # Fin de semana
                razones.append("Movimiento en fin de semana")
            
            if not razones:
                razones.append("Patrón general inusual")
            
            # Determinar severidad
            if row['anomaly_score'] < -0.5:
                severidad = "alta"
            elif row['anomaly_score'] < -0.3:
                severidad = "media"
            else:
                severidad = "baja"
            
            # Obtener descripción del producto
            product = products[products['codigo'] == row['codigo']]
            descripcion = product['descripcion'].values[0] if not product.empty else row['codigo']
            
            anomaly_list.append({
                "id": str(row.get('id', '')),
                "codigo": row['codigo'],
                "descripcion": descripcion,
                "tipo": row['tipo'],
                "cantidad": int(row['cantidad']),
                "fecha": row['created_at'].isoformat(),
                "usuario": row.get('usuario_email', 'Desconocido'),
                "anomaly_score": float(row['anomaly_score']),
                "razon": " | ".join(razones),
                "severidad": severidad
            })
        
        # Ordenar por score (más anómalo primero)
        anomaly_list.sort(key=lambda x: x['anomaly_score'])
        
        return {
            "anomalias": anomaly_list,
            "total_analizado": len(recent_movements),
            "total_anomalias": len(anomaly_list),
            "periodo_dias": dias,
            "modelo": "Isolation Forest",
            "threshold": threshold
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/user-patterns")
async def analyze_user_patterns(dias: int = 30):
    """
    Analiza patrones de uso por usuario para detectar comportamientos sospechosos
    """
    try:
        movements = get_movements()
        
        if movements.empty:
            return {"usuarios": [], "mensaje": "No hay datos"}
        
        # Filtrar por período
        cutoff_date = datetime.now() - timedelta(days=dias)
        recent = movements[movements['created_at'] >= cutoff_date].copy()
        
        if recent.empty:
            return {"usuarios": [], "mensaje": "No hay datos en el período"}
        
        # Agrupar por usuario
        user_stats = recent.groupby('usuario_email').agg({
            'id': 'count',
            'cantidad': ['sum', 'mean', 'max'],
            'created_at': ['min', 'max']
        }).reset_index()
        
        user_stats.columns = [
            'usuario', 'total_movimientos', 
            'cantidad_total', 'cantidad_promedio', 'cantidad_max',
            'primer_movimiento', 'ultimo_movimiento'
        ]
        
        # Calcular métricas adicionales
        results = []
        
        for _, user in user_stats.iterrows():
            user_movements = recent[recent['usuario_email'] == user['usuario']]
            
            # Horarios de actividad
            hours = user_movements['created_at'].dt.hour
            
            # Detectar patrones sospechosos
            alertas = []
            
            if user['cantidad_max'] > user['cantidad_promedio'] * 3:
                alertas.append("Movimientos con cantidades muy variables")
            
            if (hours < 6).any() or (hours > 22).any():
                alertas.append("Actividad en horarios inusuales")
            
            # Muchos movimientos en poco tiempo
            time_span = (user['ultimo_movimiento'] - user['primer_movimiento']).total_seconds() / 3600
            if time_span > 0 and user['total_movimientos'] / time_span > 10:
                alertas.append("Alta frecuencia de movimientos")
            
            results.append({
                "usuario": user['usuario'] or "Sin identificar",
                "total_movimientos": int(user['total_movimientos']),
                "cantidad_total": int(user['cantidad_total']),
                "cantidad_promedio": round(float(user['cantidad_promedio']), 2),
                "primer_movimiento": user['primer_movimiento'].isoformat(),
                "ultimo_movimiento": user['ultimo_movimiento'].isoformat(),
                "alertas": alertas,
                "riesgo": "alto" if len(alertas) >= 2 else "medio" if len(alertas) == 1 else "bajo"
            })
        
        # Ordenar por riesgo
        risk_order = {"alto": 0, "medio": 1, "bajo": 2}
        results.sort(key=lambda x: risk_order[x['riesgo']])
        
        return {
            "usuarios": results,
            "periodo_dias": dias,
            "total_usuarios": len(results)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/realtime-check")
async def realtime_anomaly_check(codigo: str, tipo: str, cantidad: int):
    """
    Verifica en tiempo real si un movimiento es anómalo antes de registrarlo
    """
    try:
        movements = get_movements()
        products = get_products()
        
        product = products[products['codigo'] == codigo]
        if product.empty:
            return {"is_anomaly": False, "mensaje": "Producto no encontrado"}
        
        # Obtener historial del producto
        product_movements = movements[movements['codigo'] == codigo]
        
        if len(product_movements) < 5:
            return {
                "is_anomaly": False,
                "mensaje": "Pocos datos históricos para evaluar",
                "cantidad_sugerida": None
            }
        
        # Calcular estadísticas
        same_type = product_movements[product_movements['tipo'] == tipo]
        
        if same_type.empty:
            return {
                "is_anomaly": False,
                "mensaje": "Sin historial de este tipo de movimiento"
            }
        
        mean_qty = same_type['cantidad'].mean()
        std_qty = same_type['cantidad'].std() or 1
        max_qty = same_type['cantidad'].max()
        
        # Calcular z-score
        z_score = (cantidad - mean_qty) / std_qty
        
        # Determinar si es anómalo
        is_anomaly = abs(z_score) > 2.5 or cantidad > max_qty * 1.5
        
        alertas = []
        if z_score > 2.5:
            alertas.append(f"Cantidad muy superior al promedio ({mean_qty:.1f})")
        elif z_score < -2.5:
            alertas.append(f"Cantidad muy inferior al promedio ({mean_qty:.1f})")
        
        if cantidad > max_qty * 1.5:
            alertas.append(f"Supera el máximo histórico ({max_qty})")
        
        # Verificar stock disponible para salidas
        if tipo == 'salida':
            stock_actual = product['stock'].values[0]
            if cantidad > stock_actual:
                is_anomaly = True
                alertas.append(f"Cantidad supera el stock disponible ({stock_actual})")
        
        return {
            "is_anomaly": is_anomaly,
            "z_score": round(float(z_score), 2),
            "cantidad_promedio": round(float(mean_qty), 2),
            "cantidad_maxima_historica": int(max_qty),
            "alertas": alertas,
            "severidad": "alta" if len(alertas) >= 2 else "media" if alertas else "ninguna"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))