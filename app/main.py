from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.database import init_db
from app.ml.data_loader import load_scored_panel, latest_quarter
from app.routers.alerts import router as alerts_router
from app.routers.contagion import router as contagion_router
from app.routers.indicators import router as indicators_router
from app.routers.model_meta import router as model_meta_router
from app.routers.reports import router as reports_router
from app.routers.scores import router as scores_router, scenario_router
from app.routers.sectors import router as sectors_router
from app.services.persistence_service import seed_database_from_artifacts
from app.utils.cache import cache
from app.utils.logger import configure_logging, get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    try:
        await init_db()
        await seed_database_from_artifacts()
    except Exception as exc:  # pragma: no cover
        logger.warning("database_init_failed", extra={"error": str(exc)})
    yield
    await cache.close()


app = FastAPI(
    title="SectorSentinel API",
    version=settings.model_version,
    description="Early warning system dashboard APIs for Indian sector financial stress.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(sectors_router, prefix=settings.api_v1_prefix)
app.include_router(scores_router, prefix=settings.api_v1_prefix)
app.include_router(scenario_router, prefix=settings.api_v1_prefix)
app.include_router(indicators_router, prefix=settings.api_v1_prefix)
app.include_router(contagion_router, prefix=settings.api_v1_prefix)
app.include_router(model_meta_router, prefix=settings.api_v1_prefix)
app.include_router(alerts_router, prefix=settings.api_v1_prefix)
app.include_router(reports_router, prefix=settings.api_v1_prefix)


@app.get("/health", tags=["health"])
async def health() -> dict[str, str]:
    scored_panel = load_scored_panel()
    return {
        "status": "ok",
        "model_version": settings.model_version,
        "data_freshness": latest_quarter(scored_panel),
    }


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": True, "message": exc.detail, "code": exc.status_code},
    )


@app.exception_handler(Exception)
async def generic_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled_exception", extra={"error": str(exc)})
    return JSONResponse(
        status_code=500,
        content={"error": True, "message": "Internal server error", "code": 500},
    )
