import base64
import json

from slowapi import Limiter
from slowapi.util import get_remote_address


def get_rate_limit_key(request) -> str:
    """Key limits by authenticated user id + real client IP.

    - user id (unverified decode, just for bucketing; the real signature
      check happens in the endpoint) prevents IP-rotation abuse by
      registered users.
    - the Vercel-provided client IP header (preferred) or local client
      address keeps anonymous/parallel accounts stacked on the same IP
      sharing one bucket.
    """
    auth = request.headers.get("authorization", "")
    user_part = ""
    if auth.lower().startswith("bearer "):
        try:
            payload_b64 = auth.split(" ", 1)[1].split(".")[1]
            payload = json.loads(base64.urlsafe_b64decode(payload_b64 + "=" * (-len(payload_b64) % 4)))
            user_part = f"u{payload.get('sub')}"
        except Exception:
            user_part = "ubadtoken"
    else:
        user_part = "uanonymous"

    ip = (
        request.headers.get("x-vercel-forwarded-for-ip")
        or request.headers.get("x-real-ip")
        or get_remote_address(request)
        or "unknown"
    )
    ip_part = ip.split(",")[0].strip()
    return f"{user_part}:{ip_part}"


# In-memory storage (no external services). Counters live per instance.
limiter = Limiter(key_func=get_rate_limit_key)
