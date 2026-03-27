import os

from dotenv import load_dotenv
from fastapi import FastAPI
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from slowapi.middleware import SlowAPIMiddleware

from path.tailor import route_tailor
load_dotenv()
REDIS_URL = os.getenv("REDIS_URL")
limiter = Limiter(key_func=get_remote_address, storage_uri=REDIS_URL)
app = FastAPI(title="Resume Optimizer API", version="1.0")

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)
app.include_router(route_tailor)



