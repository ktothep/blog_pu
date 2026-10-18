import os
import sqlite3
from datetime import datetime, timezone

from supabase import create_client

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Supabase is used when credentials exist and no local path is forced (tests set APP_DB_PATH).
USE_SUPABASE = not os.getenv("APP_DB_PATH") and bool(SUPABASE_URL and SUPABASE_KEY)

_SUPABASE_CLIENT = None


def get_supabase_client():
    global _SUPABASE_CLIENT
    if not SUPABASE_URL or not SUPABASE_KEY:
        return None
    if _SUPABASE_CLIENT is None:
        _SUPABASE_CLIENT = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _SUPABASE_CLIENT


def _get_client():
    return get_supabase_client()


def _sb():
    client = get_supabase_client()
    if client is None:
        raise RuntimeError("Database is not configured.")
    return client


def init_db():
    pass  # Kept for compatibility; tables are created on demand.


DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "users.db")


def get_db_path() -> str:
    return os.getenv("APP_DB_PATH", DEFAULT_DB_PATH)


# --------------------------------------------------------------------------
# SQLite backend (local dev / tests)
# --------------------------------------------------------------------------

def sqlite_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(get_db_path(), check_same_thread=False)
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
            ats_score_tailored REAL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            original_resume TEXT,
            markdown_resume TEXT,
            job_description TEXT,
            style TEXT DEFAULT 'standard',
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS user_profile (
            user_id INTEGER PRIMARY KEY,
            target_role TEXT, years_experience TEXT, industry TEXT,
            locations TEXT, skills_emphasis TEXT, salary_expectations TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS applications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            title TEXT NOT NULL, company TEXT, job_url TEXT,
            status TEXT NOT NULL DEFAULT 'saved', notes TEXT,
            applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS artifacts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            activity_id INTEGER,
            kind TEXT NOT NULL, content TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
        """
    )
    conn.commit()
    return conn


# --------------------------------------------------------------------------
# Users
# --------------------------------------------------------------------------

def user_by_email(email: str, include_password: bool = False) -> dict | None:
    if USE_SUPABASE:
        fields = "id, name, email, password_hash" if include_password else "id, name, email"
        resp = _sb().table("users").select(fields).eq("email", email).limit(1).execute()
        return resp.data[0] if resp.data else None
    cols = "id, name, email, password_hash" if include_password else "id, name, email"
    with sqlite_connection() as conn:
        row = conn.execute(f"SELECT {cols} FROM users WHERE email = ?", (email,)).fetchone()
    return dict(row) if row is not None else None


def user_by_id(user_id: int) -> dict | None:
    if USE_SUPABASE:
        resp = _sb().table("users").select("id, name, email").eq("id", user_id).limit(1).execute()
        return resp.data[0] if resp.data else None
    with sqlite_connection() as conn:
        row = conn.execute("SELECT id, name, email FROM users WHERE id = ?", (user_id,)).fetchone()
    return dict(row) if row is not None else None


def insert_user(name: str, email: str, password_hash: str) -> int:
    if USE_SUPABASE:
        resp = _sb().table("users").insert({
            "name": name,
            "email": email,
            "password_hash": password_hash,
        }).execute()
        return resp.data[0]["id"]
    with sqlite_connection() as conn:
        cursor = conn.execute("INSERT INTO users (name, email, password_hash) VALUES (?, ?, ?)", (name, email, password_hash))
        conn.commit()
        return cursor.lastrowid


# --------------------------------------------------------------------------
# Resume activity history
# --------------------------------------------------------------------------

def save_activity(user_id: int, title: str, ats_score: float, ats_score_tailored: float = None,
                  original_resume: str = None, markdown_resume: str = None,
                  job_description: str = None, style: str = "standard") -> int:
    data = {
        "user_id": user_id,
        "title": title,
        "ats_score": float(ats_score),
        "ats_score_tailored": None if ats_score_tailored is None else float(ats_score_tailored),
        "original_resume": original_resume,
        "markdown_resume": markdown_resume,
        "job_description": job_description,
        "style": style,
    }
    try:
        if USE_SUPABASE:
            resp = _sb().table("user_activity").insert(data).execute()
            return resp.data[0]["id"]
        with sqlite_connection() as conn:
            cursor = conn.execute(
                """INSERT INTO user_activity
                   (user_id, title, ats_score, ats_score_tailored, original_resume, markdown_resume, job_description, style)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (user_id, title, float(ats_score), data["ats_score_tailored"], original_resume, markdown_resume, job_description, style),
            )
            conn.commit()
            return cursor.lastrowid
    except Exception as e:
        print(f"save_activity failed: {type(e).__name__}: {e}")
        return -1


def list_activities(user_id: int, limit: int = 20) -> list[dict]:
    if USE_SUPABASE:
        resp = _sb().table("user_activity").select(
            "id, title, ats_score, ats_score_tailored, created_at, style"
        ).eq("user_id", user_id).order("id", desc=True).limit(limit).execute()
        return resp.data or []
    with sqlite_connection() as conn:
        rows = conn.execute(
            "SELECT id, title, ats_score, ats_score_tailored, created_at, style FROM user_activity WHERE user_id = ? ORDER BY id DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def get_activity(user_id: int, activity_id: int) -> dict | None:
    if USE_SUPABASE:
        resp = _sb().table("user_activity").select("*").eq("user_id", user_id).eq("id", activity_id).limit(1).execute()
        return resp.data[0] if resp.data else None
    with sqlite_connection() as conn:
        row = conn.execute("SELECT * FROM user_activity WHERE user_id = ? AND id = ?", (user_id, activity_id)).fetchone()
    return dict(row) if row is not None else None


def delete_activity(user_id: int, activity_id: int) -> None:
    if USE_SUPABASE:
        _sb().table("user_activity").delete().eq("user_id", user_id).eq("id", activity_id).execute()
    else:
        with sqlite_connection() as conn:
            conn.execute("DELETE FROM user_activity WHERE user_id = ? AND id = ?", (user_id, activity_id))
            conn.commit()


# --------------------------------------------------------------------------
# Targeting profile
# --------------------------------------------------------------------------

PROFILE_FIELDS = ("target_role", "years_experience", "industry", "locations", "skills_emphasis", "salary_expectations")


def get_profile(user_id: int) -> dict:
    if USE_SUPABASE:
        resp = _sb().table("user_profile").select("*").eq("user_id", user_id).limit(1).execute()
        if resp.data:
            data = resp.data[0]
            data.pop("user_id", None)
            return data
        return {field: "" for field in PROFILE_FIELDS}
    with sqlite_connection() as conn:
        row = conn.execute("SELECT * FROM user_profile WHERE user_id = ?", (user_id,)).fetchone()
    if row is not None:
        data = dict(row)
        data.pop("user_id", None)
        return data
    return {field: "" for field in PROFILE_FIELDS}


def save_profile(user_id: int, fields: dict) -> None:
    clean = {k: fields.get(k, "") for k in PROFILE_FIELDS}
    if USE_SUPABASE:
        _sb().table("user_profile").upsert({"user_id": user_id, **clean}, on_conflict="user_id").execute()
    else:
        with sqlite_connection() as conn:
            existing = conn.execute("SELECT user_id FROM user_profile WHERE user_id = ?", (user_id,)).fetchone()
            if existing:
                conn.execute(
                    "UPDATE user_profile SET target_role=?, years_experience=?, industry=?, locations=?, skills_emphasis=?, salary_expectations=? WHERE user_id=?",
                    (clean["target_role"], clean["years_experience"], clean["industry"], clean["locations"], clean["skills_emphasis"], clean["salary_expectations"], user_id),
                )
            else:
                conn.execute(
                    "INSERT INTO user_profile (user_id, target_role, years_experience, industry, locations, skills_emphasis, salary_expectations) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (user_id, clean["target_role"], clean["years_experience"], clean["industry"], clean["locations"], clean["skills_emphasis"], clean["salary_expectations"]),
                )
            conn.commit()


# --------------------------------------------------------------------------
# Job applications tracker
# --------------------------------------------------------------------------

APP_STATUSES = ("saved", "applied", "interview", "offer", "rejected")


def list_applications(user_id: int) -> list[dict]:
    if USE_SUPABASE:
        resp = _sb().table("applications").select("*").eq("user_id", user_id).order("id", desc=True).execute()
        return resp.data or []
    with sqlite_connection() as conn:
        rows = conn.execute("SELECT * FROM applications WHERE user_id = ? ORDER BY id DESC", (user_id,)).fetchall()
    return [dict(r) for r in rows]


def create_application(user_id: int, title: str, company: str = "", job_url: str = "", status: str = "saved", notes: str = "") -> int:
    if USE_SUPABASE:
        resp = _sb().table("applications").insert({
            "user_id": user_id, "title": title, "company": company,
            "job_url": job_url, "status": status, "notes": notes,
        }).execute()
        return resp.data[0]["id"]
    with sqlite_connection() as conn:
        cursor = conn.execute(
            "INSERT INTO applications (user_id, title, company, job_url, status, notes) VALUES (?, ?, ?, ?, ?, ?)",
            (user_id, title, company, job_url, status, notes),
        )
        conn.commit()
        return cursor.lastrowid


def update_application(user_id: int, application_id: int, fields: dict) -> dict | None:
    allowed = {k: v for k, v in fields.items() if k in ("title", "company", "job_url", "status", "notes")}
    if USE_SUPABASE:
        resp = _sb().table("applications").update(allowed).eq("user_id", user_id).eq("id", application_id).execute()
        return resp.data[0] if resp.data else None
    with sqlite_connection() as conn:
        sets = ", ".join(f"{k}=?" for k in allowed)
        values = list(allowed.values()) + [user_id, application_id]
        conn.execute(f"UPDATE applications SET {sets} WHERE user_id = ? AND id = ?", values)
        conn.commit()
        row = conn.execute("SELECT * FROM applications WHERE user_id = ? AND id = ?", (user_id, application_id)).fetchone()
    return dict(row) if row is not None else None


def delete_application(user_id: int, application_id: int) -> None:
    if USE_SUPABASE:
        _sb().table("applications").delete().eq("user_id", user_id).eq("id", application_id).execute()
    else:
        with sqlite_connection() as conn:
            conn.execute("DELETE FROM applications WHERE user_id = ? AND id = ?", (user_id, application_id))
            conn.commit()


# --------------------------------------------------------------------------
# Artifacts (cover letters, interview prep)
# --------------------------------------------------------------------------

def save_artifact(user_id: int, activity_id: int, kind: str, content: str) -> int:
    if USE_SUPABASE:
        resp = _sb().table("artifacts").insert({
            "user_id": user_id, "activity_id": activity_id, "kind": kind, "content": content,
        }).execute()
        return resp.data[0]["id"]
    with sqlite_connection() as conn:
        cursor = conn.execute(
            "INSERT INTO artifacts (user_id, activity_id, kind, content) VALUES (?, ?, ?, ?)",
            (user_id, activity_id, kind, content),
        )
        conn.commit()
        return cursor.lastrowid


def list_artifacts(user_id: int, kind: str = None, limit: int = 10) -> list[dict]:
    if USE_SUPABASE:
        query = _sb().table("artifacts").select("id, activity_id, kind, created_at").eq("user_id", user_id)
        if kind:
            query = query.eq("kind", kind)
        resp = query.order("id", desc=True).limit(limit).execute()
        return resp.data or []
    with sqlite_connection() as conn:
        if kind:
            rows = conn.execute("SELECT id, activity_id, kind, created_at FROM artifacts WHERE user_id = ? AND kind = ? ORDER BY id DESC LIMIT ?", (user_id, kind, limit)).fetchall()
        else:
            rows = conn.execute("SELECT id, activity_id, kind, created_at FROM artifacts WHERE user_id = ? ORDER BY id DESC LIMIT ?", (user_id, limit)).fetchall()
    return [dict(r) for r in rows]


def get_artifact(user_id: int, artifact_id: int) -> dict | None:
    if USE_SUPABASE:
        resp = _sb().table("artifacts").select("*").eq("user_id", user_id).eq("id", artifact_id).limit(1).execute()
        return resp.data[0] if resp.data else None
    with sqlite_connection() as conn:
        row = conn.execute("SELECT * FROM artifacts WHERE user_id = ? AND id = ?", (user_id, artifact_id)).fetchone()
    return dict(row) if row is not None else None


def log_interaction(job_url: str, filename: str, resume_text: str, status: str, error_message: str = None):
    if not USE_SUPABASE:
        return
    try:
        _sb().table("interactions").insert({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "job_url": job_url,
            "filename": filename,
            "resume_text": resume_text,
            "status": status,
            "error_message": error_message,
        }).execute()
    except Exception:
        pass
