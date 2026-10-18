import io
import os
import re
from datetime import datetime

import anthropic
from docx import Document as DocxDocument
from docx.shared import Pt
from fastapi import APIRouter, HTTPException, Body, Header, Request
from fastapi.responses import Response
from pydantic import BaseModel

from api.auth import get_authenticated_user, verify_token
from api.limiter import limiter
from api.database import (
    list_applications,
    create_application,
    update_application,
    delete_application,
    get_activity,
    save_artifact,
    APP_STATUSES,
)

route_extras = APIRouter()

DIFFICULT_WORDS_RE = re.compile(r"\b(spearhead|architect\w*|orchestrat\w*|streamlin\w*|cross-functional|stakeholder\w*|kpi|roadmap\w*|scalab\w*)\b", re.IGNORECASE)
METRICS_RE = re.compile(r"\d+%|\$\s?\d|\b\d+(?:\.\d+)?[kKmM]\b")
GENERIC_SUMMARY_RE = re.compile(r"\b(hard[- ]working|results[- ]oriented|detail[- ]oriented|team player|passionate)\b", re.IGNORECASE)


def _user_from_header(authorization: str) -> dict:
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authentication required.")
    return verify_token(authorization.split(" ", 1)[1].strip())


# --------------------------------------------------------------------------
# Feature #9: targeted "fix weaknesses" recommendations (no AI call needed)
# --------------------------------------------------------------------------

def generate_recommendations(resume_text: str, job_description: str, missing_keywords: list[str]) -> list[str]:
    recs = []
    text_sample = resume_text[:6000]

    if not METRICS_RE.search(text_sample):
        recs.append("Add quantified impact metrics — none of your bullets contain numbers or percentages (e.g. 'reduced processing time by 35%').")

    if GENERIC_SUMMARY_RE.search(resume_text[:1200].lower()) or resume_text.lower().count("responsible for") > 2:
        recs.append("Your summary/action language is generic. Replace 'responsible for' and clichés like 'hard-working' with strong action verbs.")

    if job_description:
        jd_first = [w for w in re.split(r"[^a-z]", job_description[:400].lower()) if len(w) > 3]
        resume_head = text_sample[:2000].lower()
        if jd_first and not any(w in resume_head for w in jd_first[:5]):
            recs.append("Reframe your most recent job title to mirror the target role's language so ATS matching hits the top of your resume.")

    if missing_keywords:
        recs.append(f"Weave these missing keywords into your experience where genuinely applicable: {', '.join(missing_keywords[:3])}.")

    if not DIFFICULT_WORDS_RE.search(text_sample) and job_description and any(
        w in job_description.lower() for w in ("lead", "managed", "direct", "strategy")
    ):
        recs.append("The role implies leadership scope, but your resume rarely uses leadership verbs (led, drove, owned). Elevate 2–3 bullets.")

    if len(resume_text) > 18000:
        recs.append("Your resume is very long (>4 pages). Trim older roles to one line each to keep recruiters past page 1.")

    if not recs:
        recs.append("No major weaknesses detected — your resume already aligns well with this job description.")
    return recs[:6]


# --------------------------------------------------------------------------
# Feature #8: DOCX export
# --------------------------------------------------------------------------

def markdown_to_docx(markdown_resume: str):
    doc = DocxDocument()
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(10.5)

    for line in markdown_resume.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("# "):
            doc.add_heading(stripped[2:].strip(), level=1)
        elif stripped.startswith("## "):
            doc.add_heading(stripped[3:].strip(), level=2)
        elif stripped.startswith("### "):
            doc.add_heading(stripped[4:].strip(), level=3)
        elif stripped.startswith("- ") or stripped.startswith("• "):
            doc.add_paragraph(stripped[2:].strip(), style="List Bullet")
        elif re.match(r"^\d+\.\s", stripped):
            doc.add_paragraph(re.sub(r"^\d+\.\s", "", stripped), style="List Number")
        elif stripped.startswith("**") and stripped.endswith("**"):
            p = doc.add_paragraph()
            run = p.add_run(stripped[2:-2])
            run.bold = True
        elif stripped.count("|") >= 2:
            doc.add_paragraph(" | ".join(part.strip() for part in stripped.split("|") if part.strip()))
        else:
            doc.add_paragraph(re.sub(r"\*\*(.*?)\*\*", r"\1", stripped))
    return doc


@route_extras.post("/api/export/docx")
async def export_docx(request: Request, payload: dict = Body(...)):
    get_authenticated_user(request)
    markdown_resume = payload.get("markdown_resume") or ""
    if not markdown_resume.strip():
        raise HTTPException(status_code=400, detail="No resume content to export.")

    buffer = io.BytesIO()
    markdown_to_docx(markdown_resume).save(buffer)
    buffer.seek(0)
    filename = f"tailored_resume_{datetime.now().strftime('%Y%m%d')}.docx"
    return Response(
        content=buffer.getvalue(),
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# --------------------------------------------------------------------------
# Features #11 / #12: cover letter and interview prep from a saved result
# --------------------------------------------------------------------------

COVER_LETTER_PROMPT = """You are an expert career coach and professional writer.

Write a compelling one-page cover letter in Markdown for the given job, based ONLY on the candidate's resume and profile below.

Rules:
- Professional, confident, warm — no buzzword soup, no flattery.
- Open with why the candidate is a strong fit for THIS specific role.
- In the body, connect 2–3 concrete resume achievements to the job's core requirements, quantifying impact exactly as stated in the resume.
- Never invent experience, skills, companies, titles, dates, or metrics the candidate does not have.
- Close with a clear, forward-moving call to action.
- Max 350 words. Output only the letter in Markdown (no preamble or commentary)."""

INTERVIEW_PREP_PROMPT = """You are an expert interview coach.

Using ONLY the candidate's resume and target role details below, produce an interview preparation brief in Markdown with these exact sections:

## Most likely interview questions
- 8 questions that assess the candidate's specific experience against this role.

## STAR answer bank
- For the 5 highest-priority questions, write a complete STAR answer grounded ONLY in the candidate's real resume (Situation, Task, Action, Result — keep numbers exactly as in the resume).

## Role-specific strengths to lead with
- 3–4 bullets tied to their resume evidence.

## Honest gaps and how to handle them
- 2–3 areas where the candidate is weaker than the job requires, each with a one-line bridging strategy they can say out loud.

## Questions to ask the interviewer
- 3 smart, role-specific questions.

Never invent experience the candidate does not have. Output only the Markdown brief."""


def _run_ai(system_prompt: str, user_content: str) -> str:
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4000,
        system=system_prompt,
        messages=[{"role": "user", "content": user_content}],
    )
    return next(b.text for b in response.content if hasattr(b, "text"))


class ExtrasRequest(BaseModel):
    activity_id: int
    extra_instructions: str = ""


def _generate_extra(request: Request, body: ExtrasRequest, system_prompt: str, kind: str):
    user = get_authenticated_user(request)
    activity = get_activity(user["sub"], body.activity_id)
    if not activity:
        raise HTTPException(status_code=404, detail="Resume version not found. Optimize a resume first, then try again.")
    if not activity.get("original_resume") and not activity.get("markdown_resume"):
        raise HTTPException(status_code=400, detail="This saved version is too old to reuse. Re-optimize your resume first.")

    instructions = f"Additional context from the candidate: {body.extra_instructions.strip()}\n\n" if body.extra_instructions.strip() else ""
    try:
        content = _run_ai(system_prompt, instructions + _activity_context(activity))
    except Exception as e:
        err_str = str(e)
        lowered = err_str.lower()
        if "529" in err_str or "overloaded" in lowered:
            raise HTTPException(status_code=503, detail="The AI service is temporarily overloaded. Please try again in a moment.")
        if "credit balance" in lowered or "billing" in lowered or "authentication_error" in lowered:
            raise HTTPException(status_code=503, detail="The AI service is temporarily unavailable. Please try again later.")
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")

    try:
        save_artifact(user["sub"], activity_id=body.activity_id, kind=kind, content=content)
    except Exception:
        pass
    return {"status": "success", kind: content, "activity_id": body.activity_id}


def _activity_context(activity: dict) -> str:
    return (
        f"CANDIDATE RESUME:\n{activity.get('original_resume') or '(not stored)'}\n\n"
        f"TAILORED RESUME:\n{activity.get('markdown_resume') or '(not stored)'}\n\n"
        f"JOB DESCRIPTION:\n{(activity.get('job_description') or activity.get('title') or '(not stored)')}\n"
    )


@route_extras.post("/api/cover-letter")
@limiter.limit("4/hour")
async def generate_cover_letter(request: Request, body: ExtrasRequest):
    return _generate_extra(request, body, COVER_LETTER_PROMPT, "cover_letter")


@route_extras.post("/api/interview-prep")
@limiter.limit("4/hour")
async def generate_interview_prep(request: Request, body: ExtrasRequest):
    return _generate_extra(request, body, INTERVIEW_PREP_PROMPT, "interview_prep")


# --------------------------------------------------------------------------
# Feature #10: applications tracker
# --------------------------------------------------------------------------

class ApplicationCreate(BaseModel):
    title: str
    company: str = ""
    job_url: str = ""
    status: str = "saved"
    notes: str = ""


class ApplicationUpdate(BaseModel):
    title: str | None = None
    company: str | None = None
    job_url: str | None = None
    status: str | None = None
    notes: str | None = None


@route_extras.get("/api/applications")
async def read_applications(authorization: str = Header(default="")):
    user = _user_from_header(authorization)
    try:
        rows = list_applications(user["sub"])
    except Exception:
        raise HTTPException(status_code=503, detail="Applications storage is not configured. Run the latest schema migration in Supabase.")
    return {"applications": rows}


@route_extras.post("/api/applications", status_code=201)
async def create_application_route(item: ApplicationCreate, authorization: str = Header(default="")):
    user = _user_from_header(authorization)
    if item.status not in APP_STATUSES:
        raise HTTPException(status_code=400, detail="Invalid status.")
    if not item.title.strip():
        raise HTTPException(status_code=400, detail="Job title is required.")
    try:
        app_id = create_application(
            user["sub"], item.title.strip(), item.company.strip(),
            item.job_url.strip(), item.status, item.notes.strip(),
        )
    except Exception:
        raise HTTPException(status_code=503, detail="Applications storage is not configured. Run the latest schema migration in Supabase.")
    return {"message": "Application saved.", "id": app_id}


@route_extras.patch("/api/applications/{application_id}")
async def update_application_route(application_id: int, item: ApplicationUpdate, authorization: str = Header(default="")):
    user = _user_from_header(authorization)
    fields = {k: v for k, v in item.model_dump().items() if v is not None}
    if "status" in fields and fields["status"] not in APP_STATUSES:
        raise HTTPException(status_code=400, detail="Invalid status.")
    try:
        row = update_application(user["sub"], application_id, fields)
    except Exception:
        raise HTTPException(status_code=503, detail="Applications storage is not configured. Run the latest schema migration in Supabase.")
    if not row:
        raise HTTPException(status_code=404, detail="Application not found.")
    return {"message": "Application updated.", "application": row}


@route_extras.delete("/api/applications/{application_id}")
async def delete_application_route(application_id: int, authorization: str = Header(default="")):
    user = _user_from_header(authorization)
    try:
        delete_application(user["sub"], application_id)
    except Exception:
        raise HTTPException(status_code=503, detail="Applications storage is not configured. Run the latest schema migration in Supabase.")
    return {"message": "Application deleted.", "id": application_id}
