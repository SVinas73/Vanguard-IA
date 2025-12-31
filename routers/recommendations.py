from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import joblib
import json

import sys
sys.path.append('..')
from database import get_products, get_movements

router = APIRouter()

MODELS_DIR = "models"

# ============================================
# SCHEMAS
# ============================================

class PurchaseRecommendation(BaseModel):
    codigo: str
    descripcion: str
    stock_actual: int
    stock_minimo: int
    dias_restantes: float
    consumo_diario: float
    cantidad_sugerida: int
    urgencia: str
    razon: str
    costo_estimado: Optional[float] = None

class RecommendationsResponse(BaseModel):
    fecha_generacion: str
    total_productos_analizar: int
    productos_a_pedir: List[PurchaseRecommendation]
    inversion_total_estimada: float
    resumen: dict

# ============================================
# FUNCIONES
# ============================================

def calcular_consumo_diario(movements_df: pd.DataFrame, codigo: str, dias: int = 30) -> float:
    """Calcula el consumo diario promedio de un producto"""
    cutoff = datetime.now() - timedelta(days=dias)
    
    product_movements = movements_df[
        (movements_df['codigo'] == codigo) & 
        (movements_df['tipo'] == 'salida') &
        (movements_df['created_at'] >= cutoff)
    ]
    
    if product_movements.empty:
        return 0.0
    
    total = product_movements['cantidad'].sum()
    dias_reales = (product_movements['created_at'].max() - product_movements['created_at'].min()).days or 1
    
    return total / dias_reales


def calcular_cantidad_optima(
    consumo_diario: float,
    stock_actual: int,
    stock_minimo: int,
    dias_cobertura: int = 30,
    lead_time: int = 7
) -> int:
    """
    Calcula la cantidad óptima a pedir.
    
    - dias_cobertura: cuántos días de stock queremos tener
    - lead_time: días que tarda en llegar el pedido
    """
    if consumo_diario <= 0:
        return 0
    
    # Stock necesario para cubrir el período + tiempo de entrega
    stock_necesario = consumo_diario * (dias_cobertura + lead_time)
    
    # Punto de reorden (cuándo pedir)
    punto_reorden = consumo_diario * lead_time + stock_minimo
    
    # Si estamos por debajo del punto de reorden, calcular cuánto pedir
    if stock_actual <= punto_reorden:
        cantidad = int(np.ceil(stock_necesario - stock_actual))
        return max(cantidad, 0)
    
    return 0


def predecir_demanda_futura(codigo: str, dias: int = 30) -> Optional[float]:
    """Usa el modelo entrenado para predecir demanda futura"""
    model_path = os.path.join(MODELS_DIR, f"demand_{codigo}.joblib")
    
    if not os.path.exists(model_path):
        return None
    
    try:
        model = joblib.load(model_path)
        
        # Generar features para los próximos días
        future_dates = pd.date_range(start=datetime.now(), periods=dias, freq='D')
        features = np.array([
            [d.dayofweek, d.month, d.isocalendar()[1]] for d in future_dates
        ])
        
        predictions = model.predict(features)
        return float(np.sum(predictions))
        
    except Exception:
        return None


# ============================================
# ENDPOINTS
# ============================================

@router.get("/purchase-list", response_model=RecommendationsResponse)
async def get_purchase_recommendations(
    dias_cobertura: int = 30,
    lead_time: int = 7,
    incluir_no_urgentes: bool = False
):
    """
    Genera una lista de compras recomendada basada en:
    - Consumo histórico
    - Predicciones de IA
    - Stock actual y mínimo
    - Tiempo de entrega del proveedor
    """
    try:
        products = get_products()
        movements = get_movements()
        
        if products.empty:
            raise HTTPException(status_code=400, detail="No hay productos")
        
        recommendations = []
        
        for _, product in products.iterrows():
            codigo = product['codigo']
            stock_actual = int(product['stock'])
            stock_minimo = int(product['stock_minimo'])
            precio = float(product['precio']) if product['precio'] else 0
            
            # Calcular consumo diario
            consumo_diario = calcular_consumo_diario(movements, codigo)
            
            # Intentar usar predicción de IA
            demanda_predicha = predecir_demanda_futura(codigo, dias_cobertura)
            if demanda_predicha and demanda_predicha > 0:
                consumo_diario_predicho = demanda_predicha / dias_cobertura
                # Usar el mayor entre histórico y predicho (conservador)
                consumo_diario = max(consumo_diario, consumo_diario_predicho)
            
            # Calcular días restantes
            if consumo_diario > 0:
                dias_restantes = stock_actual / consumo_diario
            else:
                dias_restantes = float('inf')
            
            # Calcular cantidad a pedir
            cantidad_sugerida = calcular_cantidad_optima(
                consumo_diario, stock_actual, stock_minimo, dias_cobertura, lead_time
            )
            
            # Determinar urgencia
            if stock_actual == 0:
                urgencia = "CRITICA"
                razon = "Sin stock - PEDIR INMEDIATAMENTE"
            elif dias_restantes < lead_time:
                urgencia = "ALTA"
                razon = f"Stock se agota antes de que llegue el pedido ({dias_restantes:.1f} días)"
            elif dias_restantes < lead_time * 2:
                urgencia = "MEDIA"
                razon = f"Stock bajo, pedir pronto ({dias_restantes:.1f} días restantes)"
            elif stock_actual <= stock_minimo:
                urgencia = "MEDIA"
                razon = f"Por debajo del stock mínimo ({stock_minimo})"
            else:
                urgencia = "BAJA"
                razon = "Stock suficiente por ahora"
            
            # Solo incluir si hay que pedir o si se piden todos
            if cantidad_sugerida > 0 or (incluir_no_urgentes and urgencia != "BAJA"):
                recommendations.append(PurchaseRecommendation(
                    codigo=codigo,
                    descripcion=product['descripcion'],
                    stock_actual=stock_actual,
                    stock_minimo=stock_minimo,
                    dias_restantes=dias_restantes if dias_restantes != float('inf') else 999,
                    consumo_diario=round(consumo_diario, 2),
                    cantidad_sugerida=cantidad_sugerida,
                    urgencia=urgencia,
                    razon=razon,
                    costo_estimado=round(precio * cantidad_sugerida, 2) if precio > 0 else None
                ))
        
        # Ordenar por urgencia
        urgencia_orden = {"CRITICA": 0, "ALTA": 1, "MEDIA": 2, "BAJA": 3}
        recommendations.sort(key=lambda x: (urgencia_orden.get(x.urgencia, 4), -x.cantidad_sugerida))
        
        # Calcular totales
        inversion_total = sum(r.costo_estimado or 0 for r in recommendations)
        
        resumen = {
            "criticos": len([r for r in recommendations if r.urgencia == "CRITICA"]),
            "alta_urgencia": len([r for r in recommendations if r.urgencia == "ALTA"]),
            "media_urgencia": len([r for r in recommendations if r.urgencia == "MEDIA"]),
            "productos_sin_movimiento": len([r for r in recommendations if r.consumo_diario == 0])
        }
        
        return RecommendationsResponse(
            fecha_generacion=datetime.now().isoformat(),
            total_productos_analizar=len(products),
            productos_a_pedir=recommendations,
            inversion_total_estimada=round(inversion_total, 2),
            resumen=resumen
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/reorder-point/{codigo}")
async def get_reorder_point(codigo: str, lead_time: int = 7):
    """
    Calcula el punto de reorden óptimo para un producto específico
    """
    try:
        products = get_products()
        movements = get_movements()
        
        product = products[products['codigo'] == codigo]
        if product.empty:
            raise HTTPException(status_code=404, detail="Producto no encontrado")
        
        product = product.iloc[0]
        consumo_diario = calcular_consumo_diario(movements, codigo)
        
        # Calcular variabilidad del consumo
        cutoff = datetime.now() - timedelta(days=30)
        product_movements = movements[
            (movements['codigo'] == codigo) & 
            (movements['tipo'] == 'salida') &
            (movements['created_at'] >= cutoff)
        ]
        
        if not product_movements.empty:
            daily_consumption = product_movements.groupby(
                product_movements['created_at'].dt.date
            )['cantidad'].sum()
            std_consumo = daily_consumption.std() or 0
        else:
            std_consumo = 0
        
        # Punto de reorden con stock de seguridad
        # ROP = (Consumo diario × Lead time) + Stock de seguridad
        # Stock de seguridad = Z × σ × √Lead time (Z=1.65 para 95% servicio)
        stock_seguridad = 1.65 * std_consumo * np.sqrt(lead_time)
        punto_reorden = (consumo_diario * lead_time) + stock_seguridad
        
        return {
            "codigo": codigo,
            "descripcion": product['descripcion'],
            "stock_actual": int(product['stock']),
            "consumo_diario_promedio": round(consumo_diario, 2),
            "variabilidad_consumo": round(std_consumo, 2),
            "lead_time_dias": lead_time,
            "stock_seguridad_sugerido": int(np.ceil(stock_seguridad)),
            "punto_reorden": int(np.ceil(punto_reorden)),
            "debe_pedir_ahora": int(product['stock']) <= int(np.ceil(punto_reorden))
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/auto-order-suggestion")
async def get_auto_order_suggestion(proveedor: Optional[str] = None, presupuesto_maximo: Optional[float] = None):
    """
    Genera una sugerencia de pedido automático optimizada.
    Puede filtrar por proveedor y limitar por presupuesto.
    """
    try:
        # Obtener lista de compras
        recommendations = await get_purchase_recommendations(
            dias_cobertura=30,
            lead_time=7,
            incluir_no_urgentes=False
        )
        
        productos_a_pedir = recommendations.productos_a_pedir
        
        # Filtrar solo urgentes
        productos_urgentes = [
            p for p in productos_a_pedir 
            if p.urgencia in ["CRITICA", "ALTA", "MEDIA"]
        ]
        
        # Si hay presupuesto máximo, optimizar
        if presupuesto_maximo and presupuesto_maximo > 0:
            # Priorizar por urgencia y costo-beneficio
            productos_seleccionados = []
            presupuesto_usado = 0
            
            for producto in productos_urgentes:
                costo = producto.costo_estimado or 0
                if presupuesto_usado + costo <= presupuesto_maximo:
                    productos_seleccionados.append(producto)
                    presupuesto_usado += costo
            
            productos_urgentes = productos_seleccionados
        
        return {
            "fecha": datetime.now().isoformat(),
            "productos": [
                {
                    "codigo": p.codigo,
                    "descripcion": p.descripcion,
                    "cantidad": p.cantidad_sugerida,
                    "urgencia": p.urgencia,
                    "costo": p.costo_estimado
                }
                for p in productos_urgentes
            ],
            "total_productos": len(productos_urgentes),
            "inversion_total": sum(p.costo_estimado or 0 for p in productos_urgentes),
            "presupuesto_maximo": presupuesto_maximo
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))