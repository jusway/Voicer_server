from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes_pages import router as pages_router
from app.api.routes_audio import router as audio_router
from app.api.routes_jobs import router as jobs_router
from app.api.routes_polish import router as polish_router
from app.api.routes_admin import router as admin_router

app = FastAPI(title="Voicer Server")

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Include routers
app.include_router(pages_router)
app.include_router(audio_router, prefix="/api")
app.include_router(jobs_router, prefix="/api")
app.include_router(polish_router, prefix="/api")
app.include_router(admin_router, prefix="/api/admin")


@app.get("/health")
def health():
    return {"status": "ok"}

