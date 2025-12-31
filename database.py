from supabase import create_client
from dotenv import load_dotenv
import os
import pandas as pd

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def get_products():
    """Obtener todos los productos"""
    response = supabase.table("productos").select("*").execute()
    return pd.DataFrame(response.data)

def get_movements():
    """Obtener todos los movimientos"""
    response = supabase.table("movimientos").select("*").execute()
    df = pd.DataFrame(response.data)
    if not df.empty and 'created_at' in df.columns:
        df['created_at'] = pd.to_datetime(df['created_at'])
    return df

def get_product_movements(codigo: str):
    """Obtener movimientos de un producto específico"""
    response = supabase.table("movimientos").select("*").eq("codigo", codigo).execute()
    df = pd.DataFrame(response.data)
    if not df.empty and 'created_at' in df.columns:
        df['created_at'] = pd.to_datetime(df['created_at'])
    return df