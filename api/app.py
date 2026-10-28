"""
FastAPI Application Entry Point.
Configures CORS, middleware, Swagger docs, and routes.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from idu_core.config import config
from api.routes import router

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("idu_api")

def create_app() -> FastAPI:
    app = FastAPI(
        title=config.app_name,
        version=config.version,
        description="""
# Intelligent Document Understanding (IDU) System
### Deep Learning Document AI REST API
Powered by Multimodal Transformers, 2D Spatial Positional Embeddings, and OCR.

### Capabilities:
- **Full Document Extraction**: Optical Character Recognition + Spatial Layout Analysis + Key Information Extraction (KIE).
- **Document Classification**: Multimodal classification across Invoices, Receipts, Forms, Resumes, Contracts, etc.
- **Table Detection & Extraction**: Grid reconstruction, headers, and rows exported as structured tabular data.
- **Document Visual Question Answering (DocVQA)**: Ask any natural language question about document semantics.
- **Visual Overlays**: Generates annotated images with color-coded bounding boxes and confidence tags.
        """,
        docs_url="/docs",
        redoc_url="/redoc"
    )

    # Enable CORS for web apps & frontends
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)

    @app.get("/", tags=["Root"])
    async def root():
        return {
            "message": "Welcome to the Intelligent Document Understanding System API",
            "docs_url": "/docs",
            "version": config.version,
            "device": config.model.device
        }

    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)
