import base64
import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import time

from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel, EmailStr, Field

route_auth = APIRouter()

DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "users.db")


def get_db_path() -> str:
    return os.getenv("APP_DB_PATH", DEFAULT_DB_PATH)


class RegisterUserRequest(BaseModel):
    name: str = Field(..., min_length=2, max_length=100)
    email: EmailStr
    password: str = Field(..., min_length=8)


class LoginUserRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8)


APP_SECRET_KEY = os.getenv("APP_SECRET_KEY", "change-me-in-production")


def get_connection() -> sqlite3.Connection:
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS user_activity (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            ats_score REAL NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
        """
    )
    conn.commit()
    return conn


def init_auth_db() -> None:
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_activity (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                ats_score REAL NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
            """
        )
        conn.commit()


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
async def register_user(payload: RegisterUserRequest):
    name = payload.name.strip()
    email = payload.email.strip().lower()

    with get_connection() as conn:
        existing = conn.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()
        if existing:
            raise HTTPException(status_code=400, detail="This email is already registered.")

        password_hash = hash_password(payload.password)
        cursor = conn.execute(
            "INSERT INTO users (name, email, password_hash) VALUES (?, ?, ?)",
            (name, email, password_hash),
        )
        user_id = cursor.lastrowid

    return {
        "message": "User registered successfully.",
        "user": {
            "id": user_id,
            "name": name,
            "email": email,
        },
    }


@route_auth.post("/api/login")
async def login_user(payload: LoginUserRequest):
    email = payload.email.strip().lower()

    with get_connection() as conn:
        user = conn.execute(
            "SELECT id, name, email, password_hash FROM users WHERE email = ?",
            (email,),
        ).fetchone()

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

    token = authorization.split(" ", 1)[1].strip()
    payload = verify_token(token)

    with get_connection() as conn:
        user = conn.execute(
            "SELECT id, name, email FROM users WHERE id = ?",
            (payload["sub"],),
        ).fetchone()

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

    token = authorization.split(" ", 1)[1].strip()
    payload = verify_token(token)

    with get_connection() as conn:
        user = conn.execute(
            "SELECT id, name, email FROM users WHERE id = ?",
            (payload["sub"],),
        ).fetchone()
        recent_activity = conn.execute(
            "SELECT title, ats_score, created_at FROM user_activity WHERE user_id = ? ORDER BY id DESC LIMIT 5",
            (user["id"],),
        ).fetchall()

    if not user:
        raise HTTPException(status_code=401, detail="User not found.")

    return {
        "user": {
            "id": user["id"],
            "name": user["name"],
            "email": user["email"],
        },
        "recent_activity": [
            {
                "title": row["title"],
                "ats_score": row["ats_score"],
                "created_at": row["created_at"],
            }
            for row in recent_activity
        ],
    }


def save_user_activity(user_id: int, title: str, ats_score: float) -> None:
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO user_activity (user_id, title, ats_score) VALUES (?, ?, ?)",
            (user_id, title, float(ats_score)),
        )
        conn.commit()


init_auth_db()
