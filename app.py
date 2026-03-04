"""
HuggingFace Spaces entry point.
Serves the Django REST API + built React frontend as a single app.
HF Spaces expects a Gradio or FastAPI app on port 7860.
We use FastAPI to mount both the Django WSGI app and serve the React build.
"""
import os
import sys
import subprocess

# Set Django settings before anything else
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'energy_api.settings')
os.environ['DEBUG'] = 'False'

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

import django
django.setup()

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from a2wsgi import WSGIMiddleware

# Import Django WSGI app
from energy_api.wsgi import application as django_app

app = FastAPI(title="Energy Forecasting API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Django under /api
app.mount("/api", WSGIMiddleware(django_app))

# Serve React frontend build
frontend_dist = os.path.join(os.path.dirname(__file__), "frontend", "dist")
if os.path.exists(frontend_dist):
    app.mount("/assets", StaticFiles(directory=os.path.join(frontend_dist, "assets")), name="assets")

    @app.get("/")
    async def root():
        return FileResponse(os.path.join(frontend_dist, "index.html"))

    @app.get("/{path:path}")
    async def spa_fallback(path: str):
        file_path = os.path.join(frontend_dist, path)
        if os.path.exists(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(frontend_dist, "index.html"))
else:
    @app.get("/")
    async def root():
        return {"message": "Energy Forecasting API. Frontend not built yet. Run: cd frontend && npm run build"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
