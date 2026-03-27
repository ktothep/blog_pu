import os
from datetime import datetime, timezone

from supabase import create_client

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")


def _get_client():
    if not SUPABASE_URL or not SUPABASE_KEY:
        return None
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def init_db():
    pass  # Table is created manually in Supabase SQL editor


def log_interaction(job_url: str, filename: str, resume_text: str, status: str, error_message: str = None):
    client = _get_client()
    if not client:
        return
    try:
        client.table("interactions").insert({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "job_url": job_url,
            "filename": filename,
            "resume_text": resume_text,
            "status": status,
            "error_message": error_message,
        }).execute()
    except Exception:
        pass
