try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = lambda: None

load_dotenv()

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from api.limiter import limiter
from api.path.tailor import route_tailor
from api.database import init_db

init_db()

app = FastAPI(title="Resume Optimizer API", version="1.0")


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": "You have reached the limit of 2 requests per hour. Please try again later."}
    )


app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)
app.include_router(route_tailor)
