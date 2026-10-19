import base64
import hashlib
import hmac
import json
import os
import secrets
import time

from fastapi import APIRouter, HTTPException, Header, Request
from pydantic import BaseModel, EmailStr, Field

from api.limiter import limiter

from api.database import (
    USE_SUPABASE,
    user_by_email,
    user_by_id,
    insert_user,
    get_profile,
    save_profile,
    save_activity,
    get_activity,
    list_activities,
    delete_activity,
)

route_auth = APIRouter()


class RegisterUserRequest(BaseModel):
    name: str = Field(..., min_length=2, max_length=100)
    email: EmailStr
    password: str = Field(..., min_length=8)


class LoginUserRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8)


class ProfileRequest(BaseModel):
    target_role: str = ""
    years_experience: str = ""
    industry: str = ""
    locations: str = ""
    skills_emphasis: str = ""
    salary_expectations: str = ""


APP_SECRET_KEY = os.getenv("APP_SECRET_KEY", "change-me-in-production")


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 200_000)
    return f"pbkdf2_sha256$200000${salt}${digest.hex()}"


def verify_password(password: str, password_hash: str) -> bool:
    try:
        algorithm, iterations, salt, digest = password_hash.split("$")
        if algorithm != "pbkdf2_sha256":
            return False
        computed = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt.encode("utf-8"),
            int(iterations),
        ).hex()
        return secrets.compare_digest(computed, digest)
    except (ValueError, TypeError):
        return False


def create_token(user_id: int, email: str, name: str) -> str:
    payload = {
        "sub": user_id,
        "email": email,
        "name": name,
        "iat": int(time.time()),
        "exp": int(time.time()) + (60 * 60 * 24 * 7),
    }
    header = base64.urlsafe_b64encode(json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":")).encode()).rstrip(b"=").decode()
    body = base64.urlsafe_b64encode(json.dumps(payload, separators=(",", ":")).encode()).rstrip(b"=").decode()
    signing_input = f"{header}.{body}".encode()
    signature = hmac.new(APP_SECRET_KEY.encode("utf-8"), signing_input, hashlib.sha256).digest()
    signature_b64 = base64.urlsafe_b64encode(signature).rstrip(b"=").decode()
    return f"{header}.{body}.{signature_b64}"


def verify_token(token: str) -> dict:
    try:
        header_b64, payload_b64, signature_b64 = token.split(".")
        signing_input = f"{header_b64}.{payload_b64}".encode("utf-8")
        expected = base64.urlsafe_b64encode(hmac.new(APP_SECRET_KEY.encode("utf-8"), signing_input, hashlib.sha256).digest()).rstrip(b"=").decode()
        if not hmac.compare_digest(signature_b64, expected):
            raise ValueError("Signature mismatch")

        padded = payload_b64 + "=" * (-len(payload_b64) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded).decode("utf-8"))
        if int(time.time()) > int(payload.get("exp", 0)):
            raise ValueError("Token expired")
        return payload
    except (ValueError, TypeError, json.JSONDecodeError):
        raise HTTPException(status_code=401, detail="Invalid or expired token.")


def get_authenticated_user(request: object) -> dict:
    auth_header = getattr(request, "headers", {}).get("authorization", "")
    if not auth_header or not auth_header.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Authentication required.")
    token = auth_header.split(" ", 1)[1].strip()
    return verify_token(token)


@route_auth.post("/api/register", status_code=201)
@limiter.limit("5/hour", error_message="Too many registration attempts. Please try again later.")
@limiter.limit("20/day", error_message="Too many accounts created from this network today. Please try again tomorrow.")
async def register_user(request: Request, payload: RegisterUserRequest):
    name = payload.name.strip()
    email = payload.email.strip().lower()

    existing = user_by_email(email)
    if existing:
        raise HTTPException(status_code=400, detail="This email is already registered.")
    user_id = insert_user(name, email, hash_password(payload.password))

    return {
        "message": "User registered successfully.",
        "user": {
            "id": user_id,
            "name": name,
            "email": email,
        },
    }


@route_auth.post("/api/login")
@limiter.limit("10/hour", error_message="Too many login attempts. Please try again later.")
@limiter.limit("40/day", error_message="Too many login attempts from this network today. Please try again tomorrow.")
async def login_user(request: Request, payload: LoginUserRequest):
    email = payload.email.strip().lower()
    user = user_by_email(email, include_password=True)

    if not user or not verify_password(payload.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    token = create_token(user["id"], user["email"], user["name"])
    return {
        "message": "Login successful.",
        "token": token,
        "user": {
            "id": user["id"],
            "name": user["name"],
            "email": user["email"],
        },
    }


@route_auth.get("/api/me")
async def get_current_user(authorization: str = Header(default="")):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token.")
    payload = verify_token(authorization.split(" ", 1)[1].strip())

    user = user_by_id(payload["sub"])
    if not user:
        raise HTTPException(status_code=401, detail="User not found.")

    return {
        "id": user["id"],
        "name": user["name"],
        "email": user["email"],
    }


@route_auth.get("/api/dashboard")
async def get_dashboard(authorization: str = Header(default="")):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authentication required.")
    payload = verify_token(authorization.split(" ", 1)[1].strip())

    user = user_by_id(payload["sub"])
    if not user:
        raise HTTPException(status_code=401, detail="User not found.")

    try:
        recent_activity = list_activities(user["id"], limit=5)
    except Exception:
        recent_activity = []

    return {
        "user": user,
        "recent_activity": [
            {
                "id": row.get("id"),
                "title": row.get("title"),
                "ats_score": row.get("ats_score"),
                "ats_score_tailored": row.get("ats_score_tailored"),
                "style": row.get("style") or "standard",
                "created_at": row.get("created_at"),
            }
            for row in recent_activity
        ],
    }


# --------------------------------------------------------------------------
# Resume history (server-side)
# --------------------------------------------------------------------------

@route_auth.get("/api/history")
async def get_history(authorization: str = Header(default="")):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authentication required.")
    payload = verify_token(authorization.split(" ", 1)[1].strip())

    try:
        rows = list_activities(payload["sub"], limit=20)
    except Exception:
        raise HTTPException(status_code=503, detail="History storage is not configured. Run the latest schema migration in Supabase.")

    return {
        "history": [
            {
                "id": row.get("id"),
                "title": row.get("title"),
                "ats_score": row.get("ats_score"),
                "ats_score_tailored": row.get("ats_score_tailored"),
                "style": row.get("style") or "standard",
                "created_at": row.get("created_at"),
            }
            for row in rows
        ]
    }


@route_auth.get("/api/history/{activity_id}")
async def get_history_item(activity_id: int, authorization: str = Header(default="")):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authentication required.")
    payload = verify_token(authorization.split(" ", 1)[1].strip())

    row = get_activity(payload["sub"], activity_id)
    if not row:
        raise HTTPException(status_code=404, detail="Resume version not found.")
    return {
        "id": row.get("id"),
        "title": row.get("title"),
        "ats_score": row.get("ats_score"),
        "ats_score_tailored": row.get("ats_score_tailored"),
        "style": row.get("style") or "standard",
        "created_at": row.get("created_at"),
        "original_resume": row.get("original_resume") or "",
        "markdown_resume": row.get("markdown_resume") or "",
        "job_description": row.get("job_description") or "",
    }


@route_auth.delete("/api/history/{activity_id}")
async def delete_history_item(activity_id: int, authorization: str = Header(default="")):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authentication required.")
    payload = verify_token(authorization.split(" ", 1)[1].strip())
    delete_activity(payload["sub"], activity_id)
    return {"message": "Deleted.", "id": activity_id}


# --------------------------------------------------------------------------
# Targeting profile
# --------------------------------------------------------------------------

@route_auth.get("/api/profile")
async def read_profile(authorization: str = Header(default="")):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authentication required.")
    payload = verify_token(authorization.split(" ", 1)[1].strip())
    try:
        return get_profile(payload["sub"])
    except Exception:
        raise HTTPException(status_code=503, detail="Profile storage is not configured. Run the latest schema migration in Supabase.")


@route_auth.put("/api/profile")
async def update_profile(fields: ProfileRequest, authorization: str = Header(default="")):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authentication required.")
    payload = verify_token(authorization.split(" ", 1)[1].strip())
    try:
        save_profile(payload["sub"], fields.model_dump())
    except Exception:
        raise HTTPException(status_code=503, detail="Profile storage is not configured. Run the latest schema migration in Supabase.")
    return {"message": "Targeting profile saved successfully."}
