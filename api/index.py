try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = lambda: None

load_dotenv()

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from api.auth import route_auth
from api.extras import route_extras
from api.limiter import limiter
from api.path.tailor import route_tailor
from api.database import init_db

init_db()

app = FastAPI(title="Resume Optimizer API", version="1.0")


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    messages = []
    for error in exc.errors():
        field = error.get("loc", [])[-1] if error.get("loc") else "input"
        msg = error.get("msg", "is invalid.").replace("String should have at least", "must be at least")
        messages.append(f"{field.capitalize()} {msg.replace(' String', ' value')}")
    return JSONResponse(
        status_code=400,
        content={"detail": " ".join(messages)},
    )


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": exc.detail if getattr(exc, "detail", None) else "Too many requests. Please try again later."},
    )


app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)
app.include_router(route_auth)
app.include_router(route_tailor)
app.include_router(route_extras)
