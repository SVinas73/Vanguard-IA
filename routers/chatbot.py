from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import google.generativeai as genai

import sys
sys.path.append('..')
from database import get_products, get_movements

router = APIRouter()

# Configurar Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# ============================================
# SCHEMAS
# ============================================

class ChatMessage(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    data: Optional[dict] = None
    suggestions: list = []

# ============================================
# FUNCIONES AUXILIARES
# ============================================

def get_stock_minimo_col(df: pd.DataFrame) -> str:
    """Determina el nombre de la columna de stock mínimo"""
    if 'stock_minimo' in df.columns:
        return 'stock_minimo'
    elif 'stockMinimo' in df.columns:
        return 'stockMinimo'
    else:
        return 'stock_minimo'

def safe_get(row, col, default=0):
    """Obtiene un valor de forma segura"""
    try:
        return row[col] if col in row.index else default
    except:
        return default

# ============================================
# FUNCIONES DE ANÁLISIS
# ============================================

def get_inventory_context(products_df: pd.DataFrame, movements_df: pd.DataFrame) -> str:
    """Genera contexto del inventario para Gemini"""
    
    if products_df.empty:
        return "No hay productos en el inventario."
    
    # Determinar nombre de columna stock mínimo
    stock_min_col = get_stock_minimo_col(products_df)
    
    # Resumen general
    total_productos = len(products_df)
    total_items = products_df['stock'].sum() if 'stock' in products_df.columns else 0
    
    # Calcular valor total
    if 'precio' in products_df.columns and 'stock' in products_df.columns:
        valor_total = (products_df['precio'] * products_df['stock']).sum()
    else:
        valor_total = 0
    
    # Stock bajo
    if stock_min_col in products_df.columns and 'stock' in products_df.columns:
        stock_bajo = products_df[products_df['stock'] <= products_df[stock_min_col]]
        sin_stock = products_df[products_df['stock'] == 0]
    else:
        stock_bajo = pd.DataFrame()
        sin_stock = pd.DataFrame()
    
    # Movimientos recientes
    now = datetime.now()
    week_ago = now - timedelta(days=7)
    month_ago = now - timedelta(days=30)
    
    if not movements_df.empty and 'created_at' in movements_df.columns:
        mov_semana = movements_df[movements_df['created_at'] >= week_ago]
        mov_mes = movements_df[movements_df['created_at'] >= month_ago]
        
        salidas_semana = mov_semana[mov_semana['tipo'] == 'salida']['cantidad'].sum() if not mov_semana.empty else 0
        salidas_mes = mov_mes[mov_mes['tipo'] == 'salida']['cantidad'].sum() if not mov_mes.empty else 0
    else:
        mov_semana = pd.DataFrame()
        mov_mes = pd.DataFrame()
        salidas_semana = 0
        salidas_mes = 0
    
    # Top productos
    top_list = []
    if not mov_mes.empty:
        salidas_mes_df = mov_mes[mov_mes['tipo'] == 'salida']
        if not salidas_mes_df.empty:
            top_productos = salidas_mes_df.groupby('codigo')['cantidad'].sum().sort_values(ascending=False).head(5)
            for codigo, cantidad in top_productos.items():
                prod = products_df[products_df['codigo'] == codigo]
                if not prod.empty:
                    top_list.append(f"- {codigo}: {prod.iloc[0]['descripcion']} ({int(cantidad)} unidades)")
    
    if not top_list:
        top_list = ["No hay datos de ventas"]
    
    # Productos con stock bajo detallado
    stock_bajo_list = []
    for _, p in stock_bajo.head(10).iterrows():
        codigo = p['codigo']
        descripcion = p.get('descripcion', codigo)
        stock = int(p.get('stock', 0))
        stock_min = int(safe_get(p, stock_min_col, 5))
        
        # Calcular días restantes
        if not movements_df.empty:
            salidas_prod = movements_df[(movements_df['codigo'] == codigo) & (movements_df['tipo'] == 'salida')]
            if not salidas_prod.empty and 'created_at' in salidas_prod.columns:
                dias = (salidas_prod['created_at'].max() - salidas_prod['created_at'].min()).days or 1
                consumo_diario = salidas_prod['cantidad'].sum() / dias
                dias_restantes = stock / consumo_diario if consumo_diario > 0 else 999
            else:
                consumo_diario = 0
                dias_restantes = 999
        else:
            consumo_diario = 0
            dias_restantes = 999
        
        stock_bajo_list.append(
            f"- {codigo}: {descripcion} | Stock: {stock}/{stock_min} | "
            f"Consumo: {consumo_diario:.1f}/día | Días restantes: {dias_restantes:.0f}"
        )
    
    context = f"""
DATOS DEL INVENTARIO (Actualizado: {now.strftime('%d/%m/%Y %H:%M')}):

RESUMEN GENERAL:
- Total productos: {total_productos}
- Total items en stock: {int(total_items)}
- Valor total del inventario: ${valor_total:,.2f} UYU
- Productos con stock bajo: {len(stock_bajo)}
- Productos sin stock: {len(sin_stock)}

MOVIMIENTOS:
- Salidas última semana: {int(salidas_semana)} unidades
- Salidas último mes: {int(salidas_mes)} unidades

TOP 5 PRODUCTOS MÁS VENDIDOS (último mes):
{chr(10).join(top_list)}

PRODUCTOS CON STOCK BAJO (requieren atención):
{chr(10).join(stock_bajo_list) if stock_bajo_list else 'Ninguno - todo el stock está bien'}

LISTA COMPLETA DE PRODUCTOS:
"""
    
    # Agregar lista de productos
    for _, p in products_df.iterrows():
        codigo = p.get('codigo', 'N/A')
        descripcion = p.get('descripcion', 'N/A')
        stock = int(p.get('stock', 0))
        stock_min = int(safe_get(p, stock_min_col, 5))
        precio = float(p.get('precio', 0))
        context += f"- {codigo}: {descripcion} | Stock: {stock} | Mínimo: {stock_min} | Precio: ${precio:.2f}\n"
    
    return context


def get_product_detail(codigo: str, products_df: pd.DataFrame, movements_df: pd.DataFrame) -> Optional[str]:
    """Obtiene detalle de un producto específico"""
    
    if products_df.empty:
        return None
    
    # Determinar nombre de columna stock mínimo
    stock_min_col = get_stock_minimo_col(products_df)
    
    # Buscar por código
    product = products_df[products_df['codigo'].str.upper() == codigo.upper()]
    
    # Si no encuentra, buscar por descripción
    if product.empty:
        product = products_df[products_df['descripcion'].str.upper().str.contains(codigo.upper(), na=False)]
    
    if product.empty:
        return None
    
    p = product.iloc[0]
    
    codigo = p.get('codigo', 'N/A')
    descripcion = p.get('descripcion', 'N/A')
    categoria = p.get('categoria', 'N/A')
    stock = int(p.get('stock', 0))
    stock_min = int(safe_get(p, stock_min_col, 5))
    precio = float(p.get('precio', 0))
    
    # Calcular estadísticas
    if not movements_df.empty:
        salidas = movements_df[(movements_df['codigo'] == codigo) & (movements_df['tipo'] == 'salida')]
        entradas = movements_df[(movements_df['codigo'] == codigo) & (movements_df['tipo'] == 'entrada')]
        
        if not salidas.empty and 'created_at' in salidas.columns:
            dias = (salidas['created_at'].max() - salidas['created_at'].min()).days or 1
            consumo_diario = salidas['cantidad'].sum() / dias
            dias_restantes = stock / consumo_diario if consumo_diario > 0 else float('inf')
            total_vendido = salidas['cantidad'].sum()
        else:
            consumo_diario = 0
            dias_restantes = float('inf')
            total_vendido = 0
        
        total_entradas = entradas['cantidad'].sum() if not entradas.empty else 0
    else:
        consumo_diario = 0
        dias_restantes = float('inf')
        total_vendido = 0
        total_entradas = 0
    
    detail = f"""
DETALLE DEL PRODUCTO {codigo}:
- Descripción: {descripcion}
- Categoría: {categoria}
- Stock actual: {stock} unidades
- Stock mínimo: {stock_min} unidades
- Precio de venta: ${precio:.2f} UYU
- Consumo promedio diario: {consumo_diario:.2f} unidades/día
- Días estimados hasta agotamiento: {dias_restantes:.0f if dias_restantes != float('inf') else 'Indefinido (sin consumo)'}
- Total vendido histórico: {int(total_vendido)} unidades
- Total entradas histórico: {int(total_entradas)} unidades
"""
    return detail


# ============================================
# ENDPOINT PRINCIPAL
# ============================================

@router.post("/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    """
    Chatbot con Google Gemini para consultas de inventario
    """
    try:
        if not GEMINI_API_KEY:
            raise HTTPException(status_code=500, detail="GEMINI_API_KEY no configurada")
        
        products = get_products()
        movements = get_movements()
        
        # Generar contexto del inventario
        inventory_context = get_inventory_context(products, movements)
        
        # Detectar si pregunta por un producto específico
        product_detail = None
        if not products.empty:
            for _, p in products.iterrows():
                codigo = str(p.get('codigo', '')).lower()
                descripcion = str(p.get('descripcion', '')).lower()
                if codigo in message.message.lower() or descripcion in message.message.lower():
                    product_detail = get_product_detail(p['codigo'], products, movements)
                    break
        
        # Construir prompt para Gemini
        system_prompt = f"""Eres un asistente de inventario inteligente llamado "Vanguard AI". 
Tu trabajo es ayudar al usuario a gestionar su inventario de manera eficiente.

REGLAS:
1. Responde siempre en español
2. Sé conciso pero informativo
3. Usa datos específicos del inventario cuando sea relevante
4. Si un producto tiene stock bajo (stock <= stock_minimo), alerta al usuario
5. Calcula y sugiere cuándo pedir más productos basándote en el consumo diario
6. Si no tienes información suficiente, dilo claramente
7. Formatea las respuestas de manera clara con viñetas o listas cuando sea apropiado
8. No uses emojis, mantén un tono profesional

{inventory_context}

{f'DETALLE DEL PRODUCTO CONSULTADO:{product_detail}' if product_detail else ''}
"""

        # Llamar a Gemini
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        chat_session = model.start_chat(history=[])
        
        full_prompt = f"{system_prompt}\n\nPREGUNTA DEL USUARIO: {message.message}"
        
        response = chat_session.send_message(full_prompt)
        
        # Generar sugerencias contextuales
        suggestions = []
        if not products.empty:
            if 'stock bajo' in message.message.lower() or 'crítico' in message.message.lower():
                suggestions = ["¿Cuánto debo pedir de cada uno?", "¿Cuáles son los más urgentes?", "Generar lista de compras"]
            elif any(str(p.get('codigo', '')).lower() in message.message.lower() for _, p in products.iterrows()):
                suggestions = ["¿Qué otros productos tienen stock bajo?", "¿Cuánto vendí esta semana?", "Ver productos relacionados"]
            else:
                suggestions = ["¿Qué productos tienen stock bajo?", "¿Cuáles son los más vendidos?", "¿Cuánto vendí este mes?"]
        else:
            suggestions = ["Ayuda", "¿Cómo agrego productos?"]
        
        return ChatResponse(
            response=response.text,
            suggestions=suggestions
        )
        
    except Exception as e:
        # Fallback si Gemini falla
        return ChatResponse(
            response=f"Lo siento, hubo un error al procesar tu consulta. Por favor intenta de nuevo. Error: {str(e)}",
            suggestions=["¿Qué productos tienen stock bajo?", "Ayuda"]
        )


@router.get("/suggestions")
async def get_suggestions():
    """Obtiene sugerencias de preguntas frecuentes"""
    return {
        "suggestions": [
            "¿Qué productos tienen stock bajo?",
            "¿Cuáles son los más vendidos?",
            "¿Cuánto vendí esta semana?",
            "¿Cuándo debo pedir más WD40?",
            "¿Cuál es el valor total del inventario?",
            "¿Qué productos debo reponer urgente?"
        ]
    }