from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.api.endpoints import sentiment, youtube
from backend.app.core.config import settings


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# Set all CORS enabled origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers at the root to match Chrome Extension contract
# Contract: https://your-api-domain.com/predict_batch
app.include_router(sentiment.router, tags=["sentiment"])
app.include_router(youtube.router, tags=["youtube"])

@app.get("/")
def health_check():
    return {"status": "healthy", "model": "LightGBM", "version": settings.VERSION}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
