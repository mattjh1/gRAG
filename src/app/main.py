from contextlib import asynccontextmanager
from pathlib import Path

from chainlit.utils import mount_chainlit
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from app.api.routes import api_router as api_router
from app.core.config import config


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("startup fastapi")
    yield
    # shutdown
    print("shutdown fastapi")


# Core Application Instance
app = FastAPI(
    title=config.PROJECT_NAME,
    openapi_url=f"{config.API_STR}/openapi.json",
    lifespan=lifespan,
)


# Set all CORS origins enabled
if config.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in config.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.include_router(api_router, prefix=config.API_STR)

cwd = Path(__file__).resolve().parent
cl_app_path = cwd / "cl_app.py"

if not cl_app_path.exists():
    raise FileNotFoundError(f"File does not exist: {cl_app_path}")

mount_chainlit(app=app, target=cl_app_path.__str__(), path="/chainlit")
