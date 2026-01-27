from fastapi import APIRouter, HTTPException
from typing import Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ML imports
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

import sys
sys.path.append('..')
from database import get_products, get_movements

router = APIRouter()

# ============================================
# APRIORI - Análisis de asociación
# ============================================

@router.get("/frequent-items")
async def get_frequent_itemsets(min_support: float = 0.1, dias: int = 90):
    """
    Apriori: Encuentra productos que se mueven frecuentemente juntos
    """
    try:
        movements = get_movements()
        products = get_products()
        
        if movements.empty:
            return {"itemsets": [], "mensaje": "No hay datos de movimientos"}
        
        # Filtrar por período
        cutoff_date = datetime.now() - timedelta(days=dias)
        recent = movements[movements['created_at'] >= cutoff_date].copy()
        
        if recent.empty:
            return {"itemsets": [], "mensaje": "No hay datos en el período"}
        
        # Crear "transacciones" basadas en movimientos del mismo día y usuario
        recent['date'] = recent['created_at'].dt.date
        recent['transaction_id'] = recent['date'].astype(str) + '_' + recent['usuario_email'].fillna('unknown')
        
        # Agrupar productos por transacción
        transactions = recent.groupby('transaction_id')['codigo'].apply(list).tolist()
        
        # Filtrar transacciones con más de 1 producto
        transactions = [t for t in transactions if len(t) > 1]
        
        if len(transactions) < 5:
            return {
                "reglas": [],
                "total_transacciones": len(transactions),
                "mensaje": "No hay suficientes transacciones para generar reglas"
            }
        
        # Aplicar TransactionEncoder
        te = TransactionEncoder()
        te_array = te.fit_transform(transactions)
        df_encoded = pd.DataFrame(te_array, columns=te.columns_)
        
        # Aplicar Apriori
        frequent_itemsets = apriori(
            df_encoded, 
            min_support=min_support, 
            use_colnames=True
        )
        
        if frequent_itemsets.empty:
            return {
                "itemsets": [],
                "mensaje": "No se encontraron patrones frecuentes con el soporte mínimo especificado"
            }
        
        # Convertir a lista
        result = []
        for _, row in frequent_itemsets.iterrows():
            items = list(row['itemsets'])
            if len(items) > 1:  # Solo conjuntos de 2 o más
                # Obtener descripciones
                item_details = []
                for codigo in items:
                    product = products[products['codigo'] == codigo]
                    desc = product['descripcion'].values[0] if not product.empty else codigo
                    item_details.append({"codigo": codigo, "descripcion": desc})
                
                result.append({
                    "productos": item_details,
                    "soporte": round(float(row['support']), 4),
                    "frecuencia": f"{row['support']*100:.1f}% de las transacciones"
                })
        
        # Ordenar por soporte
        result.sort(key=lambda x: x['soporte'], reverse=True)
        
        return {
            "itemsets": result[:20],  # Top 20
            "total_transacciones": len(transactions),
            "periodo_dias": dias,
            "soporte_minimo": min_support
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rules")
async def get_association_rules(
    min_support: float = 0.1, 
    min_confidence: float = 0.5,
    dias: int = 90
):
    """
    Genera reglas de asociación: "Si compran X, también compran Y"
    """
    try:
        movements = get_movements()
        products = get_products()
        
        if movements.empty:
            return {"reglas": [], "mensaje": "No hay datos de movimientos"}
        
        # Filtrar por período
        cutoff_date = datetime.now() - timedelta(days=dias)
        recent = movements[movements['created_at'] >= cutoff_date].copy()
        
        # Crear transacciones
        recent['date'] = recent['created_at'].dt.date
        recent['transaction_id'] = recent['date'].astype(str) + '_' + recent['usuario_email'].fillna('unknown')
        
        transactions = recent.groupby('transaction_id')['codigo'].apply(list).tolist()
        transactions = [t for t in transactions if len(t) > 1]
        
        if len(transactions) < 5:
            return {
                "reglas": [],
                "mensaje": "No hay suficientes transacciones para generar reglas"
            }
        
        # Encodear transacciones
        te = TransactionEncoder()
        te_array = te.fit_transform(transactions)
        df_encoded = pd.DataFrame(te_array, columns=te.columns_)
        
        # Apriori
        frequent_itemsets = apriori(
            df_encoded, 
            min_support=min_support, 
            use_colnames=True
        )
        
        if frequent_itemsets.empty:
            return {"reglas": [], "mensaje": "No se encontraron patrones frecuentes"}
        
        # Generar reglas de asociación
        rules = association_rules(
            frequent_itemsets, 
            metric="confidence", 
            min_threshold=min_confidence
        )
        
        if rules.empty:
            return {
                "reglas": [],
                "mensaje": "No se encontraron reglas con la confianza mínima especificada"
            }
        
        # Convertir a lista legible
        result = []
        for _, rule in rules.iterrows():
            antecedent = list(rule['antecedents'])
            consequent = list(rule['consequents'])
            
            # Obtener descripciones
            def get_desc(codigo):
                product = products[products['codigo'] == codigo]
                return product['descripcion'].values[0] if not product.empty else codigo
            
            ant_details = [{"codigo": c, "descripcion": get_desc(c)} for c in antecedent]
            cons_details = [{"codigo": c, "descripcion": get_desc(c)} for c in consequent]
            
            result.append({
                "si_compran": ant_details,
                "tambien_compran": cons_details,
                "confianza": round(float(rule['confidence']), 4),
                "soporte": round(float(rule['support']), 4),
                "lift": round(float(rule['lift']), 4),
                "interpretacion": f"El {rule['confidence']*100:.1f}% de las veces que se mueve {antecedent[0]}, también se mueve {consequent[0]}"
            })
        
        # Ordenar por lift (relevancia)
        result.sort(key=lambda x: x['lift'], reverse=True)
        
        return {
            "reglas": result[:20],  # Top 20 reglas
            "total_transacciones": len(transactions),
            "periodo_dias": dias,
            "confianza_minima": min_confidence,
            "soporte_minimo": min_support
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recommendations/{codigo}")
async def get_product_recommendations(codigo: str, dias: int = 90):
    """
    Recomienda productos relacionados basándose en patrones de compra
    """
    try:
        movements = get_movements()
        products = get_products()
        
        # Verificar que el producto existe
        product = products[products['codigo'] == codigo]
        if product.empty:
            raise HTTPException(status_code=404, detail="Producto no encontrado")
        
        # Filtrar por período
        cutoff_date = datetime.now() - timedelta(days=dias)
        recent = movements[movements['created_at'] >= cutoff_date].copy()
        
        # Crear transacciones
        recent['date'] = recent['created_at'].dt.date
        recent['transaction_id'] = recent['date'].astype(str) + '_' + recent['usuario_email'].fillna('unknown')
        
        # Encontrar transacciones que incluyen el producto
        product_transactions = recent[recent['codigo'] == codigo]['transaction_id'].unique()
        
        if len(product_transactions) < 3:
            return {
                "producto": {
                    "codigo": codigo,
                    "descripcion": product['descripcion'].values[0]
                },
                "recomendaciones": [],
                "mensaje": "Pocos datos para generar recomendaciones"
            }
        
        # Encontrar otros productos en esas transacciones
        related_movements = recent[
            (recent['transaction_id'].isin(product_transactions)) &
            (recent['codigo'] != codigo)
        ]
        
        # Contar frecuencia de productos relacionados
        related_counts = related_movements['codigo'].value_counts()
        
        # Calcular confianza (frecuencia / total transacciones del producto)
        recommendations = []
        
        for related_codigo, count in related_counts.head(10).items():
            related_product = products[products['codigo'] == related_codigo]
            if related_product.empty:
                continue
            
            confidence = count / len(product_transactions)
            
            recommendations.append({
                "codigo": related_codigo,
                "descripcion": related_product['descripcion'].values[0],
                "frecuencia_conjunta": int(count),
                "confianza": round(float(confidence), 4),
                "mensaje": f"Aparece junto a {codigo} el {confidence*100:.1f}% de las veces"
            })
        
        return {
            "producto": {
                "codigo": codigo,
                "descripcion": product['descripcion'].values[0]
            },
            "recomendaciones": recommendations,
            "total_transacciones_producto": len(product_transactions),
            "periodo_dias": dias
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))