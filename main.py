from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Crear app primero (antes de importar routers)
app = FastAPI(
    title="Vanguard AI API",
    description="API de Machine Learning y Deep Learning para gestión de inventario",
    version="1.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Importar routers después de crear app
from routers import predictions, anomalies, associations

# Incluir routers
app.include_router(predictions.router, prefix="/api/predictions", tags=["Predicciones"])
app.include_router(anomalies.router, prefix="/api/anomalies", tags=["Anomalías"])
app.include_router(associations.router, prefix="/api/associations", tags=["Asociaciones"])

@app.get("/")
def root():
    return {
        "message": "Vanguard AI API",
        "status": "online",
        "endpoints": {
            "predictions": "/api/predictions",
            "anomalies": "/api/anomalies", 
            "associations": "/api/associations"
        }
    }

@app.get("/health")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)