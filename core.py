"""
core.py — Motor de vectorizacion de imagenes con ColPali + Qdrant

Flujo completo:
  1. PDF -> imagenes (pdf2image, 300 dpi)
  2. Imagenes -> embeddings multivector (ColPali)
  3. Insercion en Qdrant
  4. Busqueda por texto o imagen (ColPali + Qdrant)
  5. Respuesta LLM con las paginas recuperadas (GPT-4o-mini con fallback a Gemini)
"""

import os
import base64
import logging
import time
from io import BytesIO
from pathlib import Path
from typing import Optional
import certifi

import stamina
import torch
from colpali_engine.models import ColPali, ColPaliProcessor
from PIL import Image
from pdf2image import convert_from_bytes, convert_from_path
from qdrant_client import QdrantClient
from qdrant_client.http import models
from dotenv import load_dotenv
import ssl

load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

os.environ["GRPC_DEFAULT_SSL_ROOTS_FILE_PATH"] = certifi.where()
ssl._create_default_https_context = ssl._create_unverified_context

# ---------------------------------------------------------------------------
# Configuracion
# ---------------------------------------------------------------------------
COLPALI_MODEL_NAME     = os.getenv("COLPALI_MODEL_NAME",     "vidore/colpali-v1.3")
COLPALI_PROCESSOR_NAME = os.getenv("COLPALI_PROCESSOR_NAME", "vidore/colpaligemma-3b-pt-448-base")
PDF_DPI                = int(os.getenv("PDF_DPI",            "300"))
BATCH_SIZE             = int(os.getenv("BATCH_SIZE",         "4"))
QDRANT_URL             = os.getenv("QDRANT_URL",             "http://91.99.108.245:6333")
QDRANT_API_KEY         = os.getenv("QDRANT_API_KEY",         "")
DEFAULT_COLLECTION     = os.getenv("DEFAULT_COLLECTION",     "imagenes_embeddings")
SEARCH_LIMIT           = int(os.getenv("SEARCH_LIMIT",       "5"))
POPPLER_PATH           = os.getenv("POPPLER_PATH",           r"G:\IA\poppler-25.12.0\Library\bin")
DEVICE_MAP             = os.getenv("DEVICE_MAP", " ")

# LLM
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY",  "")
GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY",  "")
GPT_MODEL       = os.getenv("GPT_MODEL",        "gpt-4o-mini")
GEMINI_MODEL    = os.getenv("GEMINI_MODEL",     "gemini-2.5-flash")

# ---------------------------------------------------------------------------
# Singletons  
# ---------------------------------------------------------------------------
_colpali_model: Optional[ColPali] = None
_colpali_processor: Optional[ColPaliProcessor] = None
_qdrant_client: Optional[QdrantClient] = None


# ---------------------------------------------------------------------------
# Carga del modelo
# ---------------------------------------------------------------------------

def _detect_device() -> str:
    """Detecta el mejor device disponible: CUDA > MPS > CPU."""
    env = os.getenv("DEVICE_MAP")
    if env:
        return env
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        logger.info("GPU detectada: %s -> usando cuda:0", name)
        return "cuda:0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("Apple Silicon detectado -> usando mps")
        return "mps"
    logger.info("No se detecto GPU -> usando cpu")
    return "cpu"


def get_model() -> tuple[ColPali, ColPaliProcessor]:
    """Carga ColPali y su processor (cacheado en memoria tras el primer arranque)."""
    global _colpali_model, _colpali_processor
    if _colpali_model is not None and _colpali_processor is not None:
        return _colpali_model, _colpali_processor

    device = _detect_device()
    logger.info("Cargando modelo ColPali '%s' (device=%s)...", COLPALI_MODEL_NAME, device)
    _colpali_model = ColPali.from_pretrained(
        COLPALI_MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )

    logger.info("Cargando processor '%s'...", COLPALI_PROCESSOR_NAME)
    _colpali_processor = ColPaliProcessor.from_pretrained(COLPALI_PROCESSOR_NAME)

    logger.info("Modelo y processor listos.")
    return _colpali_model, _colpali_processor


def get_qdrant() -> QdrantClient:
    """Retorna (y cachea) el cliente de Qdrant."""
    global _qdrant_client
    if _qdrant_client is None:
        logger.info("Conectando a Qdrant en %s...", QDRANT_URL)
        _qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY or None)
        logger.info("Conexion a Qdrant establecida.")
    return _qdrant_client


# ---------------------------------------------------------------------------
# Paso 1 - PDF a imagenes
# ---------------------------------------------------------------------------

def pdf_bytes_to_images(pdf_bytes: bytes, dpi: int = PDF_DPI, poppler_path = POPPLER_PATH) -> list[Image.Image]:
    """Convierte PDF (bytes) en lista de imagenes PIL, una por pagina."""
    logger.info("Convirtiendo PDF a imagenes (dpi=%d)...", dpi)
    pages = convert_from_bytes(pdf_bytes, dpi=dpi, poppler_path= poppler_path)
    images = [p.convert("RGB") for p in pages]
    logger.info("PDF convertido: %d paginas.", len(images))
    return images


def pdf_path_to_images(pdf_path: str | Path, dpi: int = PDF_DPI, poppler_path = POPPLER_PATH) -> list[Image.Image]:
    """Igual que pdf_bytes_to_images pero recibe ruta al archivo."""
    logger.info("Convirtiendo PDF desde disco: %s", pdf_path)
    pages = convert_from_path(str(pdf_path), dpi=dpi, poppler_path=poppler_path)
    images = [p.convert("RGB") for p in pages]
    logger.info("PDF convertido: %d paginas.", len(images))
    return images


def image_to_base64(img: Image.Image) -> str:
    """Serializa imagen PIL a base64 PNG."""
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def base64_to_image(b64: str) -> Image.Image:
    """Deserializa base64 a imagen PIL."""
    return Image.open(BytesIO(base64.b64decode(b64)))


# ---------------------------------------------------------------------------
# Paso 2 - Embeddings
# ---------------------------------------------------------------------------

def embed_images(images: list[Image.Image]) -> list[list[list[float]]]:
    """Genera embeddings multivector para una lista de imagenes."""
    model, processor = get_model()
    all_embeddings: list[list[list[float]]] = []
    for i in range(0, len(images), BATCH_SIZE):
        batch = images[i : i + BATCH_SIZE]
        logger.info("Embeddings: lote %d-%d de %d...", i + 1, i + len(batch), len(images))
        with torch.no_grad():
            inputs = processor.process_images(batch).to(model.device)
            embeddings = model(**inputs)
        for emb in embeddings:
            all_embeddings.append(emb.cpu().float().numpy().tolist())
    return all_embeddings


def embed_query_text(query: str) -> list[list[float]]:
    """Genera el multivector de una consulta en texto."""
    model, processor = get_model()
    with torch.no_grad():
        inputs = processor.process_queries([query]).to(model.device)
        embedding = model(**inputs)
    return embedding[0].cpu().float().numpy().tolist()


def embed_query_image(image: Image.Image) -> list[list[float]]:
    """Genera el multivector de una consulta en imagen."""
    model, processor = get_model()
    with torch.no_grad():
        inputs = processor.process_images([image]).to(model.device)
        embedding = model(**inputs)
    return embedding[0].cpu().float().numpy().tolist()


# ---------------------------------------------------------------------------
# Paso 3 - Insercion en Qdrant
# ---------------------------------------------------------------------------

@stamina.retry(on=Exception, attempts=3)
def _upsert_batch(client: QdrantClient, collection_name: str, points: list) -> None:
    client.upsert(collection_name=collection_name, points=points, wait=False)


def index_pdf(
    pdf_bytes: bytes,
    collection_name: str = DEFAULT_COLLECTION,
    filename: str = "document.pdf",
    extra_payload: dict | None = None,
) -> dict:
    """Pipeline completo: PDF -> imagenes -> embeddings -> Qdrant."""
    t0 = time.time()
    images = pdf_bytes_to_images(pdf_bytes)
    n_pages = len(images)
    embeddings = embed_images(images)
    client = get_qdrant()
    base_id = int(time.time() * 1000)

    for i, (img, emb) in enumerate(zip(images, embeddings)):
        payload = {
            "image_base64": image_to_base64(img),
            "image_name": f"{Path(filename).stem}_page_{i + 1}.png",
            "source_file": filename,
            "page_number": i + 1,
            "total_pages": n_pages,
        }
        if extra_payload:
            payload.update(extra_payload)
        try:
            _upsert_batch(client, collection_name, [
                models.PointStruct(id=base_id + i, vector=emb, payload=payload)
            ])
        except Exception as exc:
            logger.error("Error al insertar pagina %d: %s", i + 1, exc)

    elapsed = time.time() - t0
    logger.info("Indexacion completa: %d paginas en %.2fs", n_pages, elapsed)
    return {
        "collection": collection_name,
        "filename": filename,
        "pages_indexed": n_pages,
        "elapsed_seconds": round(elapsed, 3),
    }


# ---------------------------------------------------------------------------
# Paso 4 - Busqueda
# ---------------------------------------------------------------------------

def _run_search(collection_name: str, multivector_query: list[list[float]], limit: int) -> list[dict]:
    """Ejecuta la busqueda en Qdrant y devuelve resultados serializados."""
    client = get_qdrant()
    t0 = time.time()
    result = client.query_points(
        collection_name=collection_name,
        query=multivector_query,
        limit=limit,
        timeout=100,
        search_params=models.SearchParams(
            quantization=models.QuantizationSearchParams(
                ignore=False,
                rescore=True,
                oversampling=2.0,
            )
        ),
    )
    elapsed = time.time() - t0
    logger.info("Busqueda completada en %.4fs - %d resultados.", elapsed, len(result.points))

    hits = []
    for point in result.points:
        payload = point.payload or {}
        hits.append({
            "id": point.id,
            "score": point.score,
            "image_name": payload.get("image_name"),
            "source_file": payload.get("source_file"),
            "page_number": payload.get("page_number"),
            "total_pages": payload.get("total_pages"),
            "image_base64": payload.get("image_base64"),
        })
    return hits


def search_by_text(query: str, collection_name: str = DEFAULT_COLLECTION, limit: int = SEARCH_LIMIT) -> list[dict]:
    logger.info("Busqueda por texto: '%s' en '%s'", query, collection_name)
    return _run_search(collection_name, embed_query_text(query), limit)


def search_by_image(image: Image.Image, collection_name: str = DEFAULT_COLLECTION, limit: int = SEARCH_LIMIT) -> list[dict]:
    logger.info("Busqueda por imagen en '%s'", collection_name)
    return _run_search(collection_name, embed_query_image(image), limit)


def search_by_image_bytes(image_bytes: bytes, collection_name: str = DEFAULT_COLLECTION, limit: int = SEARCH_LIMIT) -> list[dict]:
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    return search_by_image(image, collection_name, limit)


# ---------------------------------------------------------------------------
# Paso 5 - Respuesta LLM (solo para busqueda por texto)
# GPT-4o-mini principal, Gemini 1.5 Flash como fallback
# ---------------------------------------------------------------------------

def _build_llm_messages(query: str, images_b64: list[str]) -> list[dict]:
    """
    Construye el payload de mensajes para la API de OpenAI (formato vision).
    Incluye la pregunta del usuario y las imagenes de contexto en base64.
    """
    content: list[dict] = [
        {
            "type": "text",
            "text": (
                "Eres un asistente experto en analisis de documentos. "
                "Se te proporcionan una o mas paginas de un documento como contexto visual. "
                "Responde la siguiente pregunta basandote UNICAMENTE en el contenido visible "
                "en las imagenes. Si la informacion no esta en las imagenes, indicalo claramente.\n\n"
                f"Pregunta: {query}"
            ),
        }
    ]
    for i, b64 in enumerate(images_b64):
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{b64}",
                "detail": "high",
            },
        })
    return [{"role": "user", "content": content}]


def _answer_with_gpt(query: str, images_b64: list[str]) -> str:
    """Llama a GPT-4o-mini con las imagenes de contexto."""
    from openai import OpenAI  # import lazy para no requerir openai si no se usa
    client = OpenAI(api_key=OPENAI_API_KEY)
    messages = _build_llm_messages(query, images_b64)
    logger.info("Llamando a GPT (%s)...", GPT_MODEL)
    response = client.chat.completions.create(
        model=GPT_MODEL,
        messages=messages,
        max_tokens=1500,
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()


def _answer_with_gemini(query: str, images_b64: list[str]) -> str:
    """Fallback: llama a Gemini 1.5 Flash con las imagenes de contexto."""
    import google.generativeai as genai  # import lazy
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
    import PIL.Image

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(GEMINI_MODEL)

    prompt = (
        "Eres un asistente experto en analisis de documentos. "
        "Se te proporcionan una o mas paginas de un documento como contexto visual. "
        "Responde la siguiente pregunta basandote UNICAMENTE en el contenido visible "
        "en las imagenes. Si la informacion no esta en las imagenes, indicalo claramente.\n\n"
        f"Pregunta: {query}"
    )

    parts = [prompt]
    for b64 in images_b64:
        img = PIL.Image.open(BytesIO(base64.b64decode(b64)))
        parts.append(img)

    logger.info("Llamando a Gemini (%s)...", GEMINI_MODEL)
    response = model.generate_content(
        parts,
        safety_settings={
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        },
        generation_config={"temperature": 0.2, "max_output_tokens": 1500},
    )
    return response.text.strip()


def answer_with_llm(query: str, hits: list[dict]) -> dict:
    """
    Toma la pregunta original y los hits del search (con image_base64),
    llama a GPT-4o-mini y si falla hace fallback a Gemini.

    Args:
        query:  Pregunta original del usuario.
        hits:   Lista de resultados de busqueda (cada uno con image_base64).

    Returns:
        Dict con 'answer', 'provider' y 'error' (si aplica).
    """
    # Tomar solo las imagenes disponibles (max 2, que son el top-2 del search)
    images_b64 = [h["image_base64"] for h in hits if h.get("image_base64")]

    if not images_b64:
        return {"answer": "No se encontraron imagenes de contexto para responder.", "provider": None}

    # Intentar GPT primero
    if OPENAI_API_KEY:
        try:
            answer = _answer_with_gpt(query, images_b64)
            logger.info("Respuesta obtenida de GPT.")
            return {"answer": answer, "provider": "gpt", "error": None}
        except Exception as exc:
            logger.warning("GPT fallo: %s — intentando Gemini como fallback...", exc)

    # Fallback a Gemini
    if GEMINI_API_KEY:
        try:
            answer = _answer_with_gemini(query, images_b64)
            logger.info("Respuesta obtenida de Gemini (fallback).")
            return {"answer": answer, "provider": "gemini", "error": None}
        except Exception as exc:
            logger.error("Gemini tambien fallo: %s", exc)
            return {"answer": None, "provider": None, "error": str(exc)}

    return {
        "answer": None,
        "provider": None,
        "error": "No hay API keys configuradas (OPENAI_API_KEY o GEMINI_API_KEY).",
    }


# ---------------------------------------------------------------------------
# Utilidades de colecciones
# ---------------------------------------------------------------------------

def list_collections() -> list[str]:
    return [c.name for c in get_qdrant().get_collections().collections]


def collection_info(collection_name: str) -> dict:
    info = get_qdrant().get_collection(collection_name)
    return {
        "name": collection_name,
        "vectors_count": info.vectors_count,
        "points_count": info.points_count,
        "status": str(info.status),
    }
