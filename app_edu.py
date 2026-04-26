"""
app_edu.py — API REST Buscador de Recursos Educativos MINEDU v3

Tres archivos totales: core.py | app_edu.py | index_edu.html

Endpoints:
  GET  /health
  GET  /api/edu/resources/filters          -> filtros desde Qdrant (sin .json)
  POST /api/edu/repository/upload          -> subir PDF manual + metadata
  POST /api/edu/repository/excel/stream    -> Excel MINEDU con progreso SSE
  POST /api/edu/repository/cancel          -> cancelar procesamiento Excel
  POST /api/edu/search/text                -> buscar por texto + filtros
  POST /api/edu/search/image               -> buscar por imagen + filtros
  POST /api/edu/search/similar             -> buscar paginas similares
"""

import asyncio
import json
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from functools import partial
from typing import Annotated, Optional
from io import BytesIO

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from PIL import Image

import core

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Iniciando Buscador de Recursos Educativos MINEDU...")
    try:
        core.get_model()
        core.get_qdrant()
        logger.info("Servicio listo.")
    except Exception as exc:
        logger.error("Error al inicializar: %s", exc)
    yield


app = FastAPI(
    title="Buscador Recursos Educativos MINEDU",
    version="3.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class FilterOptions(BaseModel):
    tipos_recurso: list[str]
    niveles:       list[str]
    areas:         list[str]
    categorias:    list[str]


class SearchHit(BaseModel):
    id:                 int | str
    score:              float
    image_base64:       Optional[str] = None
    image_name:         Optional[str] = None
    source_file:        Optional[str] = None
    page_number:        Optional[int] = None
    total_pages:        Optional[int] = None
    # Metadata educativa (solo los campos permitidos)
    categoria:          Optional[str] = None
    sub_categoria:      Optional[str] = None
    tipo_recurso:       Optional[str] = None
    titulo:             Optional[str] = None
    modalidad:          Optional[str] = None
    servicio_educativo: Optional[str] = None
    autor:              Optional[str] = None
    derecho_autoridad:  Optional[str] = None
    anio_edicion:       Optional[str] = None
    lengua_idioma:      Optional[str] = None
    nivel:              Optional[str] = None
    area:               Optional[str] = None
    resumen:            Optional[str] = None
    competencias:       Optional[str] = None


class SearchResponse(BaseModel):
    query_type:      str
    limit:           int
    results:         list[SearchHit]
    filters_applied: dict = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_response(query_type, limit, hits, filters):
    return SearchResponse(
        query_type=query_type, limit=limit,
        results=[SearchHit(**h) for h in hits],
        filters_applied={k: v for k, v in filters.items() if v},
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Estado"])
def health():
    return {
        "status":     "ok",
        "model":      core.COLPALI_MODEL_NAME,
        "collection": core.EDU_COLLECTION,
    }


@app.get("/api/edu/resources/filters", response_model=FilterOptions, tags=["Repositorio"])
def get_filters():
    """Retorna valores unicos de filtros directamente desde Qdrant (sin archivo .json)."""
    try:
        return core.get_filter_options_from_qdrant(core.EDU_COLLECTION)
    except Exception as exc:
        raise HTTPException(500, str(exc))


# ── Indexacion ────────────────────────────────────────────────────────────────

@app.post("/api/edu/repository/upload", tags=["Repositorio"])
async def upload_manual(
    file:               Annotated[UploadFile, File()],
    titulo:             Annotated[str, Form()] = "",
    tipo_recurso:       Annotated[str, Form()] = "",
    nivel:              Annotated[str, Form()] = "",
    area:               Annotated[str, Form()] = "",
    categoria:          Annotated[str, Form()] = "",
    sub_categoria:      Annotated[str, Form()] = "",
    modalidad:          Annotated[str, Form()] = "",
    servicio_educativo: Annotated[str, Form()] = "",
    autor:              Annotated[str, Form()] = "",
    derecho_autoridad:  Annotated[str, Form()] = "",
    anio_edicion:       Annotated[str, Form()] = "",
    lengua_idioma:      Annotated[str, Form()] = "",
    resumen:            Annotated[str, Form()] = "",
    competencias:       Annotated[str, Form()] = "",
):
    """
    Sube e indexa un PDF con metadata ingresada manualmente.
    Verifica duplicados por titulo antes de procesar.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Solo se aceptan archivos PDF.")
    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(400, "Archivo vacio.")

    metadata = {
        "titulo":             titulo,
        "tipo_recurso":       tipo_recurso,
        "nivel":              nivel,
        "area":               area,
        "categoria":          categoria,
        "sub_categoria":      sub_categoria,
        "modalidad":          modalidad,
        "servicio_educativo": servicio_educativo,
        "autor":              autor,
        "derecho_autoridad":  derecho_autoridad,
        "anio_edicion":       anio_edicion,
        "lengua_idioma":      lengua_idioma,
        "resumen":            resumen,
        "competencias":       competencias,
    }

    try:
        # Ejecutar en threadpool para no bloquear el event loop
        # y registrar como _active_task para que cancel_processing() lo corte
        loop = asyncio.get_event_loop()
        fn   = partial(core.index_manual_upload, pdf_bytes, file.filename, metadata)
        core._active_task = loop.run_in_executor(None, fn)
        result = await core._active_task
        core._active_task = None

        if result.get("error"):
            status = result.get("status")
            if status == "cancelled":
                raise HTTPException(499, result["message"])   # 499 = client closed
            raise HTTPException(409, result["message"])       # 409 = conflict / duplicado
        return {"status": "ok", **result}
    except HTTPException:
        raise
    except asyncio.CancelledError:
        raise HTTPException(499, "Indexacion cancelada por el usuario")
    except Exception as exc:
        logger.exception("Error al subir PDF manual")
        raise HTTPException(500, str(exc))
    finally:
        core._active_task = None


@app.post("/api/edu/repository/excel/stream", tags=["Repositorio"])
async def process_excel_stream(
    file:           Annotated[UploadFile, File()],
    max_concurrent: Annotated[int, Form(ge=1, le=8)] = 4,
):
    """
    Procesa Excel MINEDU con progreso SSE en tiempo real.
    Filtra: LENGUA=Castellano, TIPO ENLACE=PDF, ESTADO=Activo.
    Verifica duplicados en Qdrant antes de descargar cada PDF.
    """
    if not file.filename.lower().endswith((".xlsx", ".xls")):
        raise HTTPException(400, "Solo Excel (.xlsx).")
    excel_bytes = await file.read()
    if not excel_bytes:
        raise HTTPException(400, "Archivo vacio.")

    tmp = tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False)
    tmp.write(excel_bytes)
    tmp.close()
    tmp_path = tmp.name

    async def event_stream():
        # Registrar el generador como tarea activa para que cancel_processing() lo corte
        gen = core.process_excel_minedu(tmp_path, max_concurrent)
        try:
            async for event in gen:
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                if event.get("type") in ("done", "cancelled"):
                    break
        except asyncio.CancelledError:
            yield f"data: {json.dumps({'type':'cancelled','msg':'Procesamiento cancelado'}, ensure_ascii=False)}\n\n"
        finally:
            try: os.unlink(tmp_path)
            except: pass
            core._cancel_event.clear()

    task = asyncio.ensure_future(event_stream().__anext__())
    core._active_task = task

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/edu/repository/cancel", tags=["Repositorio"])
async def cancel_processing():
    """
    Cancela cualquier operacion de indexacion en curso:
    - Excel MINEDU (descarga + vectorizacion)
    - Subida manual (vectorizacion)
    El abort es inmediato: corta entre paginas o entre descargas.
    """
    core.cancel_processing()
    return {"status": "ok", "message": "Cancelacion solicitada — la operacion se detendrá en el próximo punto de control"}


# ── Busqueda ──────────────────────────────────────────────────────────────────

@app.post("/api/edu/search/text", response_model=SearchResponse, tags=["Busqueda"])
async def search_text(
    query:        Annotated[str, Form()],
    limit:        Annotated[int, Form(ge=1, le=20)] = core.EDU_SEARCH_LIMIT,
    tipo_recurso: Annotated[Optional[str], Form()] = None,
    nivel:        Annotated[Optional[str], Form()] = None,
    area:         Annotated[Optional[str], Form()] = None,
    categoria:    Annotated[Optional[str], Form()] = None,
):
    if not query.strip():
        raise HTTPException(400, "Query vacio.")
    try:
        hits = core.search_by_text(query, limit, tipo_recurso, nivel, area, categoria)
        return _make_response("text", limit, hits,
                              {"tipo_recurso": tipo_recurso, "nivel": nivel,
                               "area": area, "categoria": categoria})
    except Exception as exc:
        raise HTTPException(500, str(exc))


@app.post("/api/edu/search/image", response_model=SearchResponse, tags=["Busqueda"])
async def search_image(
    file:         Annotated[UploadFile, File()],
    limit:        Annotated[int, Form(ge=1, le=20)] = core.EDU_SEARCH_LIMIT,
    tipo_recurso: Annotated[Optional[str], Form()] = None,
    nivel:        Annotated[Optional[str], Form()] = None,
    area:         Annotated[Optional[str], Form()] = None,
    categoria:    Annotated[Optional[str], Form()] = None,
):
    img_bytes = await file.read()
    if not img_bytes:
        raise HTTPException(400, "Archivo vacio.")
    try:
        image = Image.open(BytesIO(img_bytes)).convert("RGB")
        hits  = core.search_by_image_query(image, limit, tipo_recurso, nivel, area, categoria)
        return _make_response("image", limit, hits,
                              {"tipo_recurso": tipo_recurso, "nivel": nivel,
                               "area": area, "categoria": categoria})
    except Exception as exc:
        raise HTTPException(500, str(exc))


@app.post("/api/edu/search/similar", response_model=SearchResponse, tags=["Busqueda"])
async def search_similar(
    image_base64: Annotated[str, Form()],
    limit:        Annotated[int, Form(ge=1, le=20)] = core.EDU_SEARCH_LIMIT,
):
    """
    Busca paginas visualmente similares.
    La imagen se redimensiona internamente antes de embeddear
    para evitar el error 'Part exceeded maximum size of 1024KB'.
    """
    b64 = image_base64.strip()
    if not b64:
        raise HTTPException(400, "image_base64 vacio.")
    try:
        hits = core.search_similar_to_page(b64, limit)
        return _make_response("similar", limit, hits, {})
    except Exception as exc:
        logger.exception("Error en search/similar")
        raise HTTPException(500, str(exc))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app_edu:app", host="0.0.0.0", port=8001, reload=False, log_level="info")
