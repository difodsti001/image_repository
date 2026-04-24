"""
app_edu.py — API REST del Buscador de Recursos Educativos MINEDU

Endpoints:
  GET  /health
  GET  /api/edu/resources                -> lista todos los recursos registrados
  GET  /api/edu/resources/filters        -> valores disponibles para filtros
  POST /api/edu/repository/scan          -> escanear carpeta local de PDFs
  POST /api/edu/repository/index         -> vectorizar PDFs pendientes
  POST /api/edu/repository/upload        -> subir y vectorizar un PDF manualmente
  POST /api/edu/repository/excel         -> aplicar metadata desde Excel
  POST /api/edu/search/text              -> buscar por texto (+ filtros opcionales)
  POST /api/edu/search/image             -> buscar por imagen (+ filtros opcionales)
  POST /api/edu/search/similar           -> buscar paginas similares a una pagina dada
"""

import logging
import tempfile
import os
from contextlib import asynccontextmanager
from typing import Annotated, Optional
from io import BytesIO

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image

import core
import core_edu

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Iniciando Buscador de Recursos Educativos...")
    try:
        core.get_model()
        core.get_qdrant()
        summary = core_edu.scan_repository()
        logger.info("Repositorio: %d recursos (%d indexados, %d pendientes)",
                    summary["total"], summary["indexed"], summary["pending"])
        logger.info("Servicio listo.")
    except Exception as exc:
        logger.error("Error al inicializar: %s", exc)
    yield
    logger.info("Servicio detenido.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Buscador de Recursos Educativos MINEDU",
    description="Busca fasciculos, guias y materiales educativos por texto o imagen.",
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ResourceItem(BaseModel):
    filename:   str
    pages:      int
    indexed:    bool
    indexed_at: Optional[str] = None
    area:       Optional[str] = None
    nivel:      Optional[str] = None
    grado:      Optional[str] = None
    tipo:       Optional[str] = None


class FilterOptions(BaseModel):
    areas:   list[str]
    niveles: list[str]
    tipos:   list[str]
    grados:  list[str]


class ScanResponse(BaseModel):
    total:     int
    indexed:   int
    pending:   int
    new_found: int
    removed:   int = 0


class IndexResponse(BaseModel):
    processed:   int
    pages_total: int
    errors:      list[dict]


class UploadResponse(BaseModel):
    filename:        str
    pages_indexed:   int
    elapsed_seconds: float
    area:  Optional[str] = None
    nivel: Optional[str] = None
    grado: Optional[str] = None
    tipo:  Optional[str] = None


class ExcelResponse(BaseModel):
    updated:             int
    not_found_in_index:  list[str]
    total_excel_entries: int


class SearchHit(BaseModel):
    id:           int | str
    score:        float
    image_base64: Optional[str] = None
    image_name:   Optional[str] = None
    source_file:  Optional[str] = None
    page_number:  Optional[int] = None
    total_pages:  Optional[int] = None
    area:         Optional[str] = None
    nivel:        Optional[str] = None
    grado:        Optional[str] = None
    tipo:         Optional[str] = None


class SearchResponse(BaseModel):
    query_type:      str
    limit:           int
    results:         list[SearchHit]
    filters_applied: dict = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Estado"])
def health():
    return {"status": "ok", "model": core.COLPALI_MODEL_NAME, "collection": core_edu.EDU_COLLECTION}


# ── Recursos ─────────────────────────────────────────────────────────────────

@app.get("/api/edu/resources", response_model=list[ResourceItem], tags=["Repositorio"])
def list_resources():
    """Lista todos los recursos registrados con su estado de indexacion."""
    return core_edu.get_resource_list()


@app.get("/api/edu/resources/filters", response_model=FilterOptions, tags=["Repositorio"])
def get_filters():
    """Valores disponibles para filtrar (area, nivel, tipo, grado)."""
    return core_edu.get_filter_options()


# ── Repositorio ───────────────────────────────────────────────────────────────

@app.post("/api/edu/repository/scan", response_model=ScanResponse, tags=["Repositorio"])
def scan_repository():
    """
    Escanea la carpeta local y actualiza el indice:
    - Registra nuevos PDFs encontrados.
    - Elimina del indice los PDFs pendientes que ya no existen en disco.
    """
    try:
        return core_edu.scan_repository()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/edu/repository/index", response_model=IndexResponse, tags=["Repositorio"])
def index_repository(max_files: int = 0):
    """
    Vectoriza PDFs pendientes. max_files=0 procesa todos.
    """
    try:
        return core_edu.index_pending(max_files=max_files)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/edu/repository/upload", response_model=UploadResponse, tags=["Repositorio"])
async def upload_resource(
    file: Annotated[UploadFile, File(description="PDF a subir e indexar")],
):
    """Sube un PDF manualmente, extrae metadata del nombre y lo vectoriza."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Solo se aceptan archivos PDF.")
    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Archivo vacio.")
    try:
        result = core_edu.index_single_upload(pdf_bytes, file.filename)
        return UploadResponse(**result)
    except Exception as exc:
        logger.exception("Error al subir recurso")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/edu/repository/excel", response_model=ExcelResponse, tags=["Repositorio"])
async def upload_excel_metadata(
    file: Annotated[UploadFile, File(description="Excel (.xlsx) con metadata de recursos")],
):
    """
    Carga metadata desde un archivo Excel y la aplica al indice.

    El Excel debe tener una columna 'filename' con el nombre exacto del PDF
    (campo comun entre el Excel y los PDFs). Columnas opcionales: area, nivel, grado, tipo.
    Cualquier columna adicional se guarda como metadata extra del recurso.

    Ejemplo:
      filename                        | area       | nivel    | grado | tipo
      Fasciculo_Matematica_4to.pdf    | Matematica | Primaria | 4°    | Fasciculo
    """
    if not file.filename.lower().endswith((".xlsx", ".xls")):
        raise HTTPException(status_code=400, detail="Solo se aceptan archivos Excel (.xlsx).")
    excel_bytes = await file.read()
    if not excel_bytes:
        raise HTTPException(status_code=400, detail="Archivo vacio.")

    # Guardar temporalmente para leerlo con openpyxl
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
        tmp.write(excel_bytes)
        tmp_path = tmp.name

    try:
        result = core_edu.apply_excel_metadata(tmp_path)
        return ExcelResponse(**result)
    except Exception as exc:
        logger.exception("Error al procesar Excel")
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        os.unlink(tmp_path)


# ── Busqueda ──────────────────────────────────────────────────────────────────

@app.post("/api/edu/search/text", response_model=SearchResponse, tags=["Busqueda"])
async def search_by_text(
    query: Annotated[str, Form()],
    limit: Annotated[int, Form(ge=1, le=20)] = core_edu.EDU_SEARCH_LIMIT,
    area:  Annotated[Optional[str], Form()] = None,
    nivel: Annotated[Optional[str], Form()] = None,
    tipo:  Annotated[Optional[str], Form()] = None,
    grado: Annotated[Optional[str], Form()] = None,
):
    """Busca recursos educativos por texto con filtros opcionales."""
    if not query.strip():
        raise HTTPException(status_code=400, detail="La consulta no puede estar vacia.")
    try:
        hits = core_edu.search_resources_by_text(query, limit, area, nivel, tipo, grado)
        return SearchResponse(
            query_type="text", limit=limit,
            results=[SearchHit(**h) for h in hits],
            filters_applied={k: v for k, v in
                             {"area": area, "nivel": nivel, "tipo": tipo, "grado": grado}.items() if v},
        )
    except Exception as exc:
        logger.exception("Error en busqueda por texto")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/edu/search/image", response_model=SearchResponse, tags=["Busqueda"])
async def search_by_image(
    file:  Annotated[UploadFile, File()],
    limit: Annotated[int, Form(ge=1, le=20)] = core_edu.EDU_SEARCH_LIMIT,
    area:  Annotated[Optional[str], Form()] = None,
    nivel: Annotated[Optional[str], Form()] = None,
    tipo:  Annotated[Optional[str], Form()] = None,
    grado: Annotated[Optional[str], Form()] = None,
):
    """Busca recursos educativos usando una imagen como consulta."""
    img_bytes = await file.read()
    if not img_bytes:
        raise HTTPException(status_code=400, detail="Archivo vacio.")
    try:
        image = Image.open(BytesIO(img_bytes)).convert("RGB")
        hits = core_edu.search_resources_by_image(image, limit, area, nivel, tipo, grado)
        return SearchResponse(
            query_type="image", limit=limit,
            results=[SearchHit(**h) for h in hits],
            filters_applied={k: v for k, v in
                             {"area": area, "nivel": nivel, "tipo": tipo, "grado": grado}.items() if v},
        )
    except Exception as exc:
        logger.exception("Error en busqueda por imagen")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/edu/search/similar", response_model=SearchResponse, tags=["Busqueda"])
async def search_similar(
    image_base64: Annotated[str, Form(description="Base64 PNG de la pagina de referencia")],
    limit: Annotated[int, Form(ge=1, le=20)] = core_edu.EDU_SEARCH_LIMIT,
):
    """
    Busca paginas visualmente similares a una pagina ya encontrada.
    Recibe el image_base64 de un resultado previo.
    """
    if not image_base64.strip():
        raise HTTPException(status_code=400, detail="image_base64 vacio.")
    try:
        hits = core_edu.search_similar_to_page(image_base64, limit)
        return SearchResponse(query_type="similar", limit=limit,
                              results=[SearchHit(**h) for h in hits])
    except Exception as exc:
        logger.exception("Error en busqueda por similitud")
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app_edu:app", host="0.0.0.0", port=8001, reload=False, log_level="info")
