import io
import os
import re
from collections import OrderedDict

import PyPDF2
import anthropic
import requests
from bs4 import BeautifulSoup
from docx import Document as DocxDocument
from fastapi import HTTPException, APIRouter, Form, Request, UploadFile, File
from starlette.responses import HTMLResponse

from api.limiter import limiter
from api.database import log_interaction, save_activity, get_profile
from api.auth import get_authenticated_user
from api.extras import generate_recommendations

route_tailor = APIRouter()

SYSTEM_PROMPT = """You are a professional resume writer and ATS optimization expert with 15+ years of experience helping candidates land interviews at top companies.

Your ONLY output must be the complete, rewritten resume in Markdown. No preamble, no commentary, no advice — just the resume.

## YOUR PROCESS

1. **Analyze the job description** — identify:
   - Required and preferred skills, tools, and technologies
   - Key responsibilities and deliverables
   - Seniority level and leadership expectations
   - Exact keywords and phrases the ATS will scan for

2. **Audit the candidate's resume** — identify:
   - Strongest experiences that map to the role
   - Transferable skills that can be reframed
   - Gaps to de-emphasize (not hide)
   - Quantified achievements to preserve and promote

3. **Rewrite the resume** following these rules:

### CONTENT RULES
- Mirror the exact keywords and phrases from the job description naturally throughout the resume
- Lead every bullet point with a strong action verb (Engineered, Architected, Led, Reduced, Increased, Automated, etc.)
- Quantify impact wherever the original resume has numbers — preserve all metrics exactly
- Reorder bullet points so the most job-relevant ones appear first
- Reframe job titles and section headers to match industry-standard terminology when appropriate
- Surface transferable skills: if the JD requires a technology and the candidate used an equivalent (e.g., JD says Kubernetes, candidate used Docker Swarm), include both with a brief clarifying phrase
- Keep the summary tightly focused on what the candidate brings to THIS specific role

### STRICT PROHIBITIONS
- NEVER invent experience, skills, tools, companies, titles, dates, or metrics the candidate does not have
- NEVER add a skill just because it appears in the JD if the candidate has no evidence of it
- NEVER include advice, tips, cover letter text, or explanations outside the resume itself
- NEVER truncate or omit sections from the original resume — rewrite everything

### ATS COMPATIBILITY RULES (CRITICAL)
- Use only standard section headings: "Summary", "Experience", "Skills", "Education", "Certifications", "Projects" — ATS systems fail to parse creative headings like "My Journey" or "What I Bring"
- NO tables, columns, or multi-column layouts — ATS parsers read left to right and will mangle columns
- NO graphics, icons, logos, or special Unicode symbols (✓ ★ → etc.) — use plain hyphens for bullets
- NO headers or footers with contact info — place all contact details in the body at the top
- Use standard date format: "Month Year – Month Year" (e.g., "Jan 2022 – Mar 2024") for all positions
- Spell out abbreviations at least once (e.g., "Search Engine Optimization (SEO)") so both forms are scannable
- Put the Skills section as a flat comma-separated or line-separated list — not a visual bar chart or rating system
- Contact details: reproduce ONLY the contact information actually found in the original resume (Email | Phone | LinkedIn | Location format). If a detail is not present in the original, omit it entirely — NEVER print blank placeholders like "Email: | Phone: |". If the original resume has no contact info at all, skip the contact line completely.
- Avoid text inside parentheses for job titles — ATS may strip them

### FORMAT RULES
- Start directly with the candidate's name as a top-level heading (`#`)
- Use `##` for section headings, `-` for all bullet points
- Standard section order: Name & Contact → Summary → Experience → Skills → Education → Certifications/Projects
- One bullet point = one achievement, max 2 lines, starts with action verb, ends with quantified impact where possible"""

STYLE_PROMPTS = OrderedDict([
    ("standard", ""),
    ("ats", "\n\nADDITIONAL STYLE DIRECTIVE — ATS-ONLY VERSION:\nMaximize keyword coverage and machine readability over human style. Mirror the job description's exact terminology aggressively, keep formatting ultra-simple, expand all abbreviations, and prioritize the Skills section with a dense keyword list."),
    ("senior", "\n\nADDITIONAL STYLE DIRECTIVE — SENIOR / LEADERSHIP VERSION:\nReframe every bullet to emphasize scope, ownership, leadership, and business outcomes. Use executive language (owned, drove, directed, led). Highlight team size, budget, and strategic impact. De-emphasize hands-on technical minutiae."),
    ("startup", "\n\nADDITIONAL STYLE DIRECTIVE — STARTUP VERSION:\nEmphasize versatility, speed, wearing multiple hats, ambiguity tolerance, and shipping fast. Use energetic, direct language. Highlight scrappy wins, side projects, and end-to-end ownership."),
    ("technical", "\n\nADDITIONAL STYLE DIRECTIVE — TECHNICAL VERSION:\nInclude precise tech stacks, tools, architectures, and methodologies. Add a dense technologies line per role. Quantify technical outcomes (latency, throughput, reliability, cost)."),
    ("career_switch", "\n\nADDITIONAL STYLE DIRECTIVE — CAREER-SWITCH VERSION:\nLead with a strong repositioning summary that bridges past experience to the new field. Surface transferable skills explicitly (e.g. 'Applied X, which translates directly to Y'). Reframe titles/achievements in the target industry's vocabulary. De-emphasize unrelated industry jargon."),
])


STOP_WORDS = {
    "the", "and", "for", "with", "that", "this", "into", "from", "your", "their", "have",
    "has", "will", "about", "upon", "over", "under", "more", "most", "such", "than", "then",
    "team", "role", "jobs", "job", "company", "companies", "candidate", "position", "strong",
    "support", "related", "including", "across", "through", "within", "around", "without"
}


def _tokenize_keywords(text: str) -> set[str]:
    if not text:
        return set()
    clean = re.sub(r"[^a-z0-9\s+\-]", " ", text.lower())
    tokens = []
    for token in clean.split():
        token = token.strip("-")
        if len(token) <= 2:
            continue
        if token in STOP_WORDS:
            continue
        tokens.append(token)
    return set(tokens)


def extract_keyword_hits(job_description: str, resume_text: str) -> set[str]:
    jd_terms = _tokenize_keywords(job_description)
    resume_terms = _tokenize_keywords(resume_text)
    if not jd_terms:
        return set()
    return {term for term in jd_terms if term in resume_terms}


def _weak_coverage_keywords(jd_terms: set[str], matched: set[str], resume_text: str) -> set[str]:
    """Terms not exactly present but closely related to something in the resume (#2)."""
    resume_words = [w for w in re.split(r"[^a-z0-9+\-]", resume_text.lower()) if len(w) >= 5]
    weak = set()
    for term in jd_terms - matched:
        if len(term) < 5:
            continue
        if any(term in w or w in term for w in resume_words):
            weak.add(term)
    return weak


def calculate_ats_score(job_description: str, resume_text: str) -> float:
    jd_terms = _tokenize_keywords(job_description)
    matched = extract_keyword_hits(job_description, resume_text)
    if not jd_terms:
        return 0.0
    weak = _weak_coverage_keywords(jd_terms, matched, resume_text)
    return round((len(matched) / len(jd_terms)) * 100, 1)


def analyze_resume_for_ats(job_description: str, resume_text: str) -> dict:
    jd_terms = _tokenize_keywords(job_description)
    matched = extract_keyword_hits(job_description, resume_text)
    weak = _weak_coverage_keywords(jd_terms, matched, resume_text)
    missing = sorted(jd_terms - matched - weak)
    return {
        "score": calculate_ats_score(job_description, resume_text),
        "matched_keywords": [
            {"keyword": k, "coverage": "strong"} for k in sorted(matched)
        ] + [
            {"keyword": k, "coverage": "weak"} for k in sorted(weak)
        ],
        "missing_keywords": missing[:10],
        "coverage_counts": {
            "strong": len(matched),
            "weak": len(weak),
            "missing": len(missing),
        } if jd_terms else {"strong": 0, "weak": 0, "missing": 0},
    }


def clean_job_text(text: str) -> str:
    """Feature #5: strip boilerplate/UI noise from scraped or pasted job descriptions."""
    if not text:
        return ""
    boilerplate_patterns = re.compile(
        r"^(apply now|apply on company site|show more|show less|report( this)? job|share( this)?|save(d)? job|"
        r"sign in|join now|see all|easy apply|get matched|log ?in|"
        r"we use cookies|cookie preferences|manage preferences|accept all|reject all|"
        r"(c|©)\s*\d{4}.*|all rights reserved|privacy policy|terms of service|"
        r"oldest first|sort by|posted (on|:)|reposted🕧?|day[s]? ago|posted.*ago|"
        r"follow us|contact us|visit website|click here|subscribe|newsletter)$",
        re.IGNORECASE,
    )
    seen_lines = set()
    cleaned = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            if cleaned and cleaned[-1] != "":
                cleaned.append("")
            continue
        if len(line) > 2 and len(set(line)) <= 2:  # banner separators / dots
            continue
        if boilerplate_patterns.match(line):
            continue
        if line.lower() in seen_lines:
            continue
        seen_lines.add(line.lower())
        cleaned.append(line)
    while cleaned and not cleaned[0]:
        cleaned.pop(0)
    while cleaned and not cleaned[-1]:
        cleaned.pop()
    return "\n".join(cleaned).strip()


def coarse_company_blurb(job_description: str) -> str:
    """Drop commonly repeated 'about the company' trailing sections if clearly present."""
    for marker in ("About the Company", "About us", "About Us", "Who We Are", "Why Join Us"):
        match = re.search(rf"\n{re.escape(marker)}\s*\n", job_description, re.IGNORECASE)
        if match and len(job_description) - match.start() < 1500:
            return job_description[:match.start()].strip()
    return job_description.strip()


def scrape_url(url: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    resp = requests.get(url, headers=headers, timeout=15)
    if resp.status_code != 200:
        return ""
    soup = BeautifulSoup(resp.content, "html.parser")
    for tag in soup(["script", "style", "meta", "noscript", "head", "nav", "footer", "header"]):
        tag.decompose()
    lines = [l for l in soup.get_text(separator="\n", strip=True).splitlines() if l.strip()]
    return coarse_company_blurb(clean_job_text("\n".join(lines)))


def parse_file(file_bytes: bytes, filename: str) -> str:
    if filename.endswith(".pdf"):
        text = "\n".join(p.extract_text() or "" for p in PyPDF2.PdfReader(io.BytesIO(file_bytes)).pages)
        if not text.strip():
            raise ValueError("This PDF looks like a scan (no selectable text). Export a text-based PDF or paste your resume text.")
        return text
    elif filename.endswith(".txt"):
        return file_bytes.decode("utf-8")
    elif filename.endswith(".docx"):
        doc = DocxDocument(io.BytesIO(file_bytes))
        parts = [p.text for p in doc.paragraphs if p.text.strip()]
        for table in doc.tables:
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells if c.text.strip()]
                if cells:
                    parts.append(" | ".join(cells))
        return "\n".join(parts)
    elif filename.endswith(".doc"):
        raise ValueError("Legacy .doc files aren't supported. Save the file as .docx or PDF, or paste the text.")
    else:
        raise ValueError("Only PDF, DOCX and TXT files are supported.")


TOOLS = [
    {
        "name": "scrape_job_url",
        "description": "Fetches and extracts the plain text content of a job posting URL. Call this first to retrieve the job description before tailoring the resume.",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL of the job posting to scrape."
                }
            },
            "required": ["url"]
        }
    }
]


@route_tailor.post("/api/optimize")
@limiter.limit("2/hour", error_message="Free limit reached: 2 optimizations per hour. Upgrade or try again later.")
@limiter.limit("6/day", error_message="Daily limit reached: 6 optimizations per day.")
async def optimize_resume(
    request: Request,
    job_url: str = Form(default=""),
    job_description_text: str = Form(default=""),
    style: str = Form(default="standard"),
    resume_text: str = Form(default=""),
    resume_file: UploadFile | None = File(None),
):
    auth_user = get_authenticated_user(request)

    if style not in STYLE_PROMPTS:
        style = "standard"

    resume_text_extracted = None
    filename_for_log = "pasted_text"
    job_description_for_analysis = ""
    try:
        if not job_url.strip() and not job_description_text.strip():
            raise HTTPException(status_code=400, detail="Please provide either a job posting URL or paste the job description.")

        if resume_file is not None and resume_file.filename:
            file_bytes = await resume_file.read()
            filename_for_log = resume_file.filename
            try:
                resume_text_extracted = parse_file(file_bytes, filename_for_log.lower())
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
        elif resume_text.strip():
            resume_text_extracted = resume_text.strip()
        else:
            raise HTTPException(status_code=400, detail="Please attach a resume file (PDF, DOCX, TXT) or paste your resume text.")

        if not resume_text_extracted.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from the uploaded file.")

        # Feature #6: pull the saved targeting profile into the prompt
        try:
            profile = get_profile(auth_user["sub"])
        except Exception:
            profile = {}
        profile_lines = [f"{label}: {value}" for label, value in [
            ("Target role", profile.get("target_role")),
            ("Years of experience", profile.get("years_experience")),
            ("Industry", profile.get("industry")),
            ("Preferred locations", profile.get("locations")),
            ("Skills the candidate wants to emphasize", profile.get("skills_emphasis")),
            ("Salary expectations", profile.get("salary_expectations")),
        ] if value]
        profile_block = ""
        if profile_lines:
            profile_block = "CANDIDATE TARGETING PROFILE (use this context to tailor the resume consistently):\n" + "\n".join(profile_lines) + "\n\n"

        style_directive = STYLE_PROMPTS.get(style, "")

        if job_description_text.strip():
            job_description_for_analysis = coarse_company_blurb(clean_job_text(job_description_text.strip()))

        resume_text = resume_text_extracted

        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        def _finalize(jd_for_ats: str, title: str):
            ats_summary = analyze_resume_for_ats(jd_for_ats, resume_text)
            ats_tailored_summary = analyze_resume_for_ats(jd_for_ats, final_text) if jd_for_ats else None
            recommendations = generate_recommendations(resume_text, jd_for_ats, ats_summary["missing_keywords"])
            headline = job_url.strip() or ("Pasted job description" if job_description_text.strip() else "Resume optimization")
            activity_id = save_activity(
                user_id=auth_user["sub"],
                title=f"{headline} ({style})" if style != "standard" else headline,
                ats_score=ats_summary["score"],
                ats_score_tailored=ats_tailored_summary["score"] if ats_tailored_summary else None,
                original_resume=resume_text,
                markdown_resume=final_text,
                job_description=jd_for_ats,
                style=style,
            )
            log_interaction(job_url or "text-input", filename_for_log, resume_text, "success")
            warning = None
            if job_url.strip() and not jd_for_ats:
                warning = "We couldn't read the job description from that URL, so ATS scoring was skipped. Paste the job description text for accurate scores and keyword analysis."
            return {
                "status": "success",
                "markdown_resume": final_text,
                "original_resume_text": resume_text,
                "ats_score": ats_summary["score"],
                "ats_score_tailored": ats_tailored_summary["score"] if ats_tailored_summary else None,
                "matched_keywords": ats_summary["matched_keywords"],
                "missing_keywords": ats_summary["missing_keywords"],
                "coverage_counts": ats_summary["coverage_counts"],
                "recommendations": recommendations,
                "activity_id": activity_id,
                "style": style,
                "warning": warning,
            }

        if job_description_text.strip():
            user_content = f"{profile_block}Here is the job description:\n\n{job_description_for_analysis}\n\nHere is my current resume:\n\n{resume_text}"
            if job_url.strip():
                user_content = f"Job posting URL: {job_url}\n\n" + user_content
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=4096,
                system=SYSTEM_PROMPT + style_directive,
                messages=[{"role": "user", "content": user_content}]
            )
            final_text = next(b.text for b in response.content if hasattr(b, "text"))
            return _finalize(job_description_for_analysis, job_url or "Resume optimization")

        messages = [
            {
                "role": "user",
                "content": f"{profile_block}Please tailor my resume for this job posting: {job_url}\n\nHere is my current resume:\n\n{resume_text}"
            }
        ]
        final_job_description = ""

        while True:
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=4096,
                system=SYSTEM_PROMPT + style_directive,
                tools=TOOLS,
                messages=messages
            )

            if response.stop_reason == "tool_use":
                tool_use_block = next(b for b in response.content if b.type == "tool_use")
                final_job_description = scrape_url(tool_use_block.input["url"])

                if not final_job_description:
                    final_job_description = "Could not retrieve the job description from this URL. Please tailor the resume based on the URL context and any visible keywords."
                else:
                    final_job_description = coarse_company_blurb(clean_job_text(final_job_description))

                messages.append({"role": "assistant", "content": response.content})
                messages.append({
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": tool_use_block.id, "content": final_job_description}]
                })

            elif response.stop_reason == "end_turn":
                final_text = next(b.text for b in response.content if hasattr(b, "text"))
                jd_for_ats = final_job_description if final_job_description and "Could not retrieve" not in final_job_description[:40] else ""
                return _finalize(jd_for_ats, job_url)

            else:
                raise HTTPException(status_code=500, detail=f"Unexpected stop reason: {response.stop_reason}")

    except HTTPException as e:
        log_interaction(job_url or "text-input", filename_for_log, "", "error", e.detail)
        raise
    except Exception as e:
        err_str = str(e)
        lowered = err_str.lower()
        if "529" in err_str or "overloaded" in lowered:
            msg = "The AI service is temporarily overloaded. Please try again in a moment."
        elif "credit balance" in lowered or "billing" in lowered or "authentication_error" in lowered:
            msg = "The AI service is temporarily unavailable. Please try again later."
        else:
            log_interaction(job_url or "text-input", filename_for_log, "", "error", f"{type(e).__name__}: {e}")
            raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")
        log_interaction(job_url or "text-input", filename_for_log, "", "error", msg)
        raise HTTPException(status_code=503, detail=msg)


@route_tailor.get("/")
async def serve_ui():
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Resume Tailor</title>
    <script src="https://cdn.tailwindcss.com?plugins=typography"></script>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <script async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client=ca-pub-2654835390881247"
     crossorigin="anonymous"></script>
    </head>
    <body class="bg-slate-950 text-gray-100 min-h-screen font-sans">
        <div class="max-w-7xl mx-auto px-4 py-8 sm:px-6 lg:px-8">
            <header class="bg-white/5 backdrop-blur border border-white/10 rounded-2xl px-5 py-4 mb-8">
                <div class="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
                    <div class="flex items-center gap-3">
                        <div class="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-500 to-cyan-400 flex items-center justify-center text-lg font-bold text-white">A</div>
                        <div>
                            <div class="text-lg font-semibold text-white">AI Resume Tailor</div>
                            <div class="text-xs text-slate-300">Career acceleration platform</div>
                        </div>
                    </div>

                    <div class="flex items-center gap-3">
                        <div id="authStatus" class="hidden text-sm text-slate-200 bg-white/10 rounded-full px-3 py-2 border border-white/10">Not signed in</div>
                        <button type="button" id="logoutBtn" class="hidden bg-white text-slate-900 px-4 py-2 rounded-lg hover:bg-slate-100 text-sm font-semibold">Log out</button>
                    </div>
                </div>
            </header>

            <main class="space-y-10">
                <section class="bg-gradient-to-br from-slate-900 via-blue-950 to-slate-900 rounded-3xl border border-white/10 p-6 md:p-10 shadow-2xl">
                    <div class="grid lg:grid-cols-[1.3fr_0.7fr] gap-8 items-center">
                        <div>
                            <span class="inline-flex items-center rounded-full border border-blue-400/30 bg-blue-500/10 px-3 py-1 text-xs font-medium text-blue-200 uppercase tracking-[0.18em]">Built for modern job search</span>
                            <h1 class="mt-5 text-4xl md:text-5xl font-black tracking-tight text-white leading-tight">
                                Tailor every application to the role you want.
                            </h1>
                            <p class="mt-4 max-w-xl text-lg text-slate-300 leading-relaxed">
                                Turn one resume into a high-impact application for any job description with ATS-aware keyword optimization, polished rewrite suggestions, and a private dashboard to track your wins.
                            </p>
                            <div class="mt-6 flex flex-wrap gap-3">
                                <a href="#optimizer" class="inline-flex items-center justify-center rounded-xl bg-blue-600 px-5 py-3 text-sm font-semibold text-white hover:bg-blue-500 transition-colors">Start optimizing</a>
                                <a href="#features" class="inline-flex items-center justify-center rounded-xl border border-white/15 bg-white/5 px-5 py-3 text-sm font-semibold text-white hover:bg-white/10 transition-colors">See how it works</a>
                            </div>
                            <div class="mt-8 flex flex-wrap gap-6 text-sm text-slate-300">
                                <div><span class="block text-2xl font-bold text-white">94%</span> ATS relevance focus</div>
                                <div><span class="block text-2xl font-bold text-white">2x</span> faster applications</div>
                                <div><span class="block text-2xl font-bold text-white">1-click</span> resume rewrite</div>
                            </div>
                        </div>

                        <div class="rounded-2xl border border-white/10 bg-white/5 p-5 shadow-xl">
                            <div class="rounded-xl border border-blue-400/20 bg-slate-900/80 p-4">
                                <div class="flex items-center justify-between text-xs uppercase tracking-[0.15em] text-slate-400">
                                    <span>Application fit</span>
                                    <span>AI score</span>
                                </div>
                                <div class="mt-5 text-4xl font-black text-blue-300">92%</div>
                                <div class="mt-3 h-2 w-full rounded-full bg-slate-700">
                                    <div class="h-2 w-[92%] rounded-full bg-gradient-to-r from-blue-500 to-cyan-400"></div>
                                </div>
                                <div class="mt-5 space-y-3 text-sm text-slate-200">
                                    <div class="flex items-center justify-between rounded-lg bg-slate-800/80 px-3 py-2">
                                        <span>Leadership</span>
                                        <span class="text-emerald-300">Matched</span>
                                    </div>
                                    <div class="flex items-center justify-between rounded-lg bg-slate-800/80 px-3 py-2">
                                        <span>Python / SQL</span>
                                        <span class="text-emerald-300">Matched</span>
                                    </div>
                                    <div class="flex items-center justify-between rounded-lg bg-slate-800/80 px-3 py-2">
                                        <span>Stakeholder communication</span>
                                        <span class="text-amber-300">Improved</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </section>

                <section id="features" class="grid md:grid-cols-3 gap-5">
                    <div class="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
                        <div class="w-11 h-11 rounded-xl bg-blue-100 text-blue-700 flex items-center justify-center font-bold text-lg mb-4">01</div>
                        <h3 class="text-xl font-semibold text-slate-900">ATS-first rewrite</h3>
                        <p class="mt-2 text-sm leading-6 text-slate-600">Match real job keywords, preserve your experience, and reframe bullet points to stand out to hiring managers and recruiters.</p>
                    </div>

                    <div class="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
                        <div class="w-11 h-11 rounded-xl bg-emerald-100 text-emerald-700 flex items-center justify-center font-bold text-lg mb-4">02</div>
                        <h3 class="text-xl font-semibold text-slate-900">Job URL or pasted JD</h3>
                        <p class="mt-2 text-sm leading-6 text-slate-600">Paste a description or drop a job URL, and the app can find the relevant requirements and align your resume to them.</p>
                    </div>

                    <div class="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
                        <div class="w-11 h-11 rounded-xl bg-violet-100 text-violet-700 flex items-center justify-center font-bold text-lg mb-4">03</div>
                        <h3 class="text-xl font-semibold text-slate-900">Track your progress</h3>
                        <p class="mt-2 text-sm leading-6 text-slate-600">Review your recent optimizations in a private dashboard and keep a running history of the roles you are targeting.</p>
                    </div>
                </section>

                <section class="grid lg:grid-cols-2 gap-6">
                    <div class="border border-slate-200 rounded-2xl p-6 bg-white shadow-sm">
                        <h2 class="text-2xl font-bold text-slate-900 mb-5">Create your account</h2>
                        <form id="registerForm" class="space-y-3">
                            <input id="registerName" type="text" placeholder="Full name" class="w-full px-4 py-3 text-gray-900 placeholder:text-gray-400 border border-slate-300 rounded-xl focus:ring-2 focus:ring-blue-500 outline-none" required>
                            <input id="registerEmail" type="email" placeholder="Email" class="w-full px-4 py-3 text-gray-900 placeholder:text-gray-400 border border-slate-300 rounded-xl focus:ring-2 focus:ring-blue-500 outline-none" required>
                            <input id="registerPassword" type="password" placeholder="Password" class="w-full px-4 py-3 text-gray-900 placeholder:text-gray-400 border border-slate-300 rounded-xl focus:ring-2 focus:ring-blue-500 outline-none" required>
                            <button type="submit" class="w-full bg-blue-600 text-white font-semibold py-3 rounded-xl hover:bg-blue-700 transition-colors">Create account</button>
                        </form>
                    </div>

                    <div class="border border-slate-200 rounded-2xl p-6 bg-white shadow-sm">
                        <h2 class="text-2xl font-bold text-slate-900 mb-5">Welcome back</h2>
                        <form id="loginForm" class="space-y-3">
                            <input id="loginEmail" type="email" placeholder="Email" class="w-full px-4 py-3 text-gray-900 placeholder:text-gray-400 border border-slate-300 rounded-xl focus:ring-2 focus:ring-blue-500 outline-none" required>
                            <input id="loginPassword" type="password" placeholder="Password" class="w-full px-4 py-3 text-gray-900 placeholder:text-gray-400 border border-slate-300 rounded-xl focus:ring-2 focus:ring-blue-500 outline-none" required>
                            <button type="submit" class="w-full bg-emerald-600 text-white font-semibold py-3 rounded-xl hover:bg-emerald-700 transition-colors">Sign in</button>
                        </form>
                    </div>
                </section>

                <section id="dashboardPanel" class="hidden border border-blue-100 bg-blue-50 rounded-2xl p-6 shadow-sm">
                    <div class="flex items-center justify-between gap-3 mb-3">
                        <h2 class="text-2xl font-bold text-blue-900">Dashboard</h2>
                        <span id="dashboardUser" class="text-sm font-medium text-blue-700"></span>
                    </div>
                    <div id="dashboardActivity" class="space-y-3 text-sm text-blue-900"></div>
                    <div class="mt-5 pt-5 border-t border-blue-200">
                        <div class="flex items-center justify-between gap-3">
                            <h3 class="text-sm font-bold text-blue-900 uppercase tracking-wide">Applications tracker</h3>
                            <button type="button" id="showAppFormBtn" class="bg-blue-600 text-white text-xs font-semibold px-3 py-2 rounded-lg hover:bg-blue-500">+ Add application</button>
                        </div>
                        <form id="appForm" class="hidden mt-3 grid md:grid-cols-5 gap-2">
                            <input id="appTitle" type="text" placeholder="Job title *" required class="px-3 py-2 text-gray-900 border border-blue-200 rounded-lg text-sm outline-none">
                            <input id="appCompany" type="text" placeholder="Company" class="px-3 py-2 text-gray-900 border border-blue-200 rounded-lg text-sm outline-none">
                            <input id="appUrl" type="url" placeholder="Job link" class="px-3 py-2 text-gray-900 border border-blue-200 rounded-lg text-sm outline-none">
                            <select id="appStatus" class="px-3 py-2 text-gray-900 bg-white border border-blue-200 rounded-lg text-sm outline-none">
                                <option value="saved">Saved</option>
                                <option value="applied">Applied</option>
                                <option value="interview">Interview</option>
                                <option value="offer">Offer</option>
                                <option value="rejected">Rejected</option>
                            </select>
                            <button type="submit" class="bg-slate-900 text-white text-sm font-semibold rounded-lg hover:bg-slate-800">Track it</button>
                            <textarea id="appNotes" rows="2" placeholder="Notes: recruiter name, referral, resume style used..." class="md:col-span-4 px-3 py-2 text-gray-900 border border-blue-200 rounded-lg text-sm outline-none"></textarea>
                            <button type="button" id="cancelAppBtn" class="text-xs text-slate-500 hover:text-slate-700 font-medium">Cancel</button>
                        </form>
                        <div id="appsList" class="mt-3 space-y-2 text-sm"></div>
                    </div>
                </section>

                <section id="optimizer" class="rounded-3xl border border-slate-200 bg-white p-6 md:p-8 shadow-sm">
                    <div class="mb-6">
                        <p class="text-sm uppercase tracking-[0.18em] text-blue-600 font-semibold">Resume optimizer</p>
                        <h2 class="mt-2 text-3xl font-black text-slate-900">Optimize your resume for the next role</h2>
                    </div>

                    <form id="resumeForm" class="space-y-6">
                        <div class="rounded-xl border border-slate-200 bg-slate-50 p-4">
                            <div class="flex items-center justify-between gap-3">
                                <div>
                                    <h3 class="text-sm font-bold text-slate-800">Targeting profile</h3>
                                    <p class="text-xs text-slate-500">Auto-attached to every optimization so the AI knows your context.</p>
                                </div>
                                <button type="button" id="editProfileBtn" class="text-xs font-semibold text-blue-700 hover:underline">Edit</button>
                            </div>
                            <div id="profileView" class="mt-3 text-sm text-slate-700">Not set yet — click Edit to give the AI your target role, experience and focus skills.</div>
                            <div id="profileForm" class="hidden mt-3 grid md:grid-cols-2 gap-3">
                                <div><label class="text-xs font-semibold text-slate-600">Target role</label>
                                    <input id="profTargetRole" type="text" placeholder="e.g. Senior Product Manager" class="mt-1 w-full px-3 py-2 text-gray-900 border border-slate-300 rounded-lg text-sm outline-none focus:ring-2 focus:ring-blue-500"></div>
                                <div><label class="text-xs font-semibold text-slate-600">Years of experience</label>
                                    <select id="profYears" class="mt-1 w-full px-3 py-2 text-gray-900 bg-white border border-slate-300 rounded-lg text-sm outline-none focus:ring-2 focus:ring-blue-500">
                                        <option value="">Select</option><option>0–2</option><option>2–5</option><option>5–8</option><option>8–12</option><option>12+</option>
                                    </select></div>
                                <div><label class="text-xs font-semibold text-slate-600">Industry</label>
                                    <input id="profIndustry" type="text" placeholder="e.g. Fintech, Healthcare" class="mt-1 w-full px-3 py-2 text-gray-900 border border-slate-300 rounded-lg text-sm outline-none focus:ring-2 focus:ring-blue-500"></div>
                                <div><label class="text-xs font-semibold text-slate-600">Preferred locations</label>
                                    <input id="profLocations" type="text" placeholder="e.g. Bengaluru, Remote (India)" class="mt-1 w-full px-3 py-2 text-gray-900 border border-slate-300 rounded-lg text-sm outline-none focus:ring-2 focus:ring-blue-500"></div>
                                <div><label class="text-xs font-semibold text-slate-600">Skills to emphasize</label>
                                    <input id="profSkills" type="text" placeholder="e.g. Python, product analytics, SQL" class="mt-1 w-full px-3 py-2 text-gray-900 border border-slate-300 rounded-lg text-sm outline-none focus:ring-2 focus:ring-blue-500"></div>
                                <div><label class="text-xs font-semibold text-slate-600">Salary expectations (optional)</label>
                                    <input id="profSalary" type="text" placeholder="Shared with AI only, never stored elsewhere" class="mt-1 w-full px-3 py-2 text-gray-900 border border-slate-300 rounded-lg text-sm outline-none focus:ring-2 focus:ring-blue-500"></div>
                                <div class="md:col-span-2 flex gap-2 justify-end">
                                    <button type="button" id="profileCancelBtn" class="text-sm text-slate-500 hover:text-slate-700 font-medium px-3 py-2">Cancel</button>
                                    <button type="button" id="profileSaveBtn" class="bg-slate-900 text-white text-sm font-semibold px-4 py-2 rounded-lg hover:bg-slate-800">Save profile</button>
                                </div>
                            </div>
                        </div>
                    <div>
                            <label class="block text-sm font-medium text-slate-700 mb-2">Job Posting URL <span class="text-slate-400 font-normal">(optional if you paste below)</span></label>
                            <input type="url" id="jobUrl" placeholder="https://linkedin.com/jobs/..."
                                   class="w-full px-4 py-3 text-gray-900 placeholder:text-gray-400 border border-slate-300 rounded-xl focus:ring-2 focus:ring-blue-500 outline-none">
                        </div>
                        <div>
                            <label class="block text-sm font-medium text-slate-700 mb-2">
                                Job Description
                                <span class="text-slate-400 font-normal">(paste here if URL can't be parsed)</span>
                            </label>
                            <textarea id="jobDescriptionText" rows="6" placeholder="Paste the full job description here..."
                                      class="w-full px-4 py-3 text-gray-900 placeholder:text-gray-400 border border-slate-300 rounded-xl focus:ring-2 focus:ring-blue-500 outline-none resize-y text-sm"></textarea>
                        </div>
                        <div>
                            <label class="block text-sm font-medium text-slate-700 mb-2">Your Current Resume (PDF/DOCX/TXT)</label>
                            <input type="file" id="resumeFile" accept=".pdf,.docx,.txt"
                                   class="w-full px-4 py-3 border border-slate-300 rounded-xl file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700">
                            <p class="text-xs text-slate-400 mt-1">Or paste your resume text below instead of uploading a file:</p>
                            <textarea id="resumeTextPaste" rows="6" placeholder="Paste your resume text here (optional if you upload a file above)..."
                                      class="w-full mt-2 px-4 py-3 text-gray-900 placeholder:text-gray-400 border border-slate-300 rounded-xl focus:ring-2 focus:ring-blue-500 outline-none resize-y text-sm"></textarea>
                        </div>
                        <div>
                            <label class="block text-sm font-medium text-slate-700 mb-2">Tailoring style</label>
                            <select id="styleSelect" class="w-full px-4 py-3 text-gray-900 bg-white border border-slate-300 rounded-xl focus:ring-2 focus:ring-blue-500 outline-none text-sm">
                                <option value="standard">Standard (balanced)</option>
                                <option value="ats">ATS-only version</option>
                                <option value="senior">Senior / leadership version</option>
                                <option value="startup">Startup version</option>
                                <option value="technical">Technical version</option>
                                <option value="career_switch">Career switch version</option>
                            </select>
                        </div>
                        <button type="submit" id="submitBtn"
                                class="w-full bg-slate-900 text-white font-bold py-3 px-4 rounded-xl hover:bg-slate-800 transition-colors">
                            Optimize Resume
                        </button>
                    </form>

                    <div id="loading" class="hidden mt-8 text-center text-blue-600 font-medium animate-pulse">
                        Processing document and generating tailored resume... (20-30s)
                    </div>

                    <div id="results" class="hidden mt-10 pt-8 border-t border-slate-200 space-y-6">
                        <div id="scoreWarning" class="hidden bg-amber-50 border border-amber-300 text-amber-800 text-sm rounded-xl p-4"></div>
                        <div class="grid md:grid-cols-3 gap-6">
                            <div class="bg-blue-50 border border-blue-100 rounded-xl p-5">
                                <div class="text-sm text-blue-700 font-semibold uppercase tracking-wide">ATS: Before → After</div>
                                <div class="flex flex-wrap items-baseline gap-x-2 gap-y-1 mt-2">
                                    <span id="atsScoreBefore" class="text-2xl font-semibold text-slate-400">0%</span>
                                    <span class="text-gray-400">→</span>
                                    <span id="atsScore" class="text-3xl font-bold text-blue-900">0%</span>
                                    <span id="atsDelta" class="hidden text-xs font-bold text-emerald-600"></span>
                                </div>
                            </div>
                            <div class="bg-emerald-50 border border-emerald-100 rounded-xl p-5">
                                <div class="text-sm text-emerald-700 font-semibold uppercase tracking-wide">Keyword coverage</div>
                                <div class="mt-2 space-y-1 text-sm text-slate-800">
                                    <div><span class="font-semibold text-emerald-800">Strong:</span> <span id="strongCount">0</span></div>
                                    <div><span class="font-semibold text-amber-600">Weak:</span> <span id="weakCount">0</span></div>
                                    <div><span class="font-semibold text-red-600">Missing:</span> <span id="missingCount">0</span></div>
                                </div>
                            </div>
                            <div class="bg-slate-50 border border-slate-200 rounded-xl p-5">
                                <div class="text-sm text-slate-700 font-semibold uppercase tracking-wide">Fix weaknesses</div>
                                <ul id="recommendationList" class="mt-2 space-y-2 text-sm text-slate-800 list-disc pl-4"></ul>
                            </div>
                        </div>

                        <div class="bg-white border border-slate-200 rounded-xl p-4">
                            <h2 class="text-lg font-semibold text-slate-800 mb-3">Keyword coverage vs job description</h2>
                            <div id="keywordsPanel" class="space-y-2 text-sm"></div>
                        </div>

                        <div class="bg-white border border-slate-200 rounded-xl p-4">
                            <div class="flex items-center justify-between flex-wrap gap-2 mb-3">
                                <h2 class="text-lg font-semibold text-slate-800">Before vs After</h2>
                                <div class="flex gap-2 flex-wrap">
                                    <button type="button" id="copyBtn" class="bg-blue-600 text-white text-xs font-semibold px-3 py-2 rounded-lg hover:bg-blue-500 transition-colors">Copy text</button>
                                    <button type="button" id="downloadMdBtn" class="border border-blue-600 text-blue-700 text-xs font-semibold px-3 py-2 rounded-lg hover:bg-blue-50 transition-colors">.MD</button>
                                    <button type="button" id="downloadTxtBtn" class="border border-blue-600 text-blue-700 text-xs font-semibold px-3 py-2 rounded-lg hover:bg-blue-50 transition-colors">.TXT</button>
                                    <button type="button" id="downloadDocxBtn" class="border border-blue-600 text-blue-700 text-xs font-semibold px-3 py-2 rounded-lg hover:bg-blue-50 transition-colors">.DOCX</button>
                                    <button type="button" id="printPdfBtn" class="border border-slate-300 text-slate-700 text-xs font-semibold px-3 py-2 rounded-lg hover:bg-slate-50 transition-colors">Print / Save PDF</button>
                                </div>
                            </div>
                            <div class="grid lg:grid-cols-2 gap-4">
                                <div class="bg-slate-50 p-4 rounded-xl border border-slate-200">
                                    <h3 class="text-xs font-bold text-slate-500 uppercase tracking-wide mb-3">Original</h3>
                                    <pre id="originalResumeText" class="whitespace-pre-wrap text-sm text-slate-700 max-h-96 overflow-auto"></pre>
                                </div>
                                <div class="bg-blue-50 p-4 rounded-xl border border-blue-100">
                                    <h3 class="text-xs font-bold text-blue-700 uppercase tracking-wide mb-3">Tailored <span id="styleBadge" class="hidden bg-blue-600 text-white rounded-full px-2 py-0.5 text-[10px] align-middle ml-1"></span></h3>
                                    <div id="markdownOutput" class="prose prose-sm max-w-none text-slate-800"></div>
                                    <p class="text-[11px] text-amber-600 mt-3">Highlighted terms are keywords matched with the job description.</p>
                                </div>
                            </div>
                        </div>

                        <div class="bg-white border border-slate-200 rounded-xl p-4">
                            <h2 class="text-lg font-semibold text-slate-800 mb-3">Go further with this application</h2>
                            <div class="flex gap-3 flex-wrap items-center">
                                <button type="button" id="coverLetterBtn" class="bg-emerald-600 text-white text-sm font-semibold px-4 py-2 rounded-lg hover:bg-emerald-500 transition-colors">Generate cover letter</button>
                                <button type="button" id="interviewPrepBtn" class="bg-violet-600 text-white text-sm font-semibold px-4 py-2 rounded-lg hover:bg-violet-500 transition-colors">Interview prep brief</button>
                                <span id="extrasHint" class="text-xs text-slate-500">Generated from your latest saved optimization (max 4/hour each).</span>
                            </div>
                            <div id="extrasOutput" class="hidden mt-4 pt-4 border-t border-slate-100">
                                <div class="flex items-center justify-between">
                                    <h3 id="extrasTitle" class="text-xs font-bold text-slate-800 uppercase tracking-wide"></h3>
                                    <button type="button" id="extrasCopyBtn" class="text-xs text-blue-700 hover:underline">Copy</button>
                                </div>
                                <div id="extrasContent" class="prose prose-sm max-w-none text-slate-800 mt-3"></div>
                            </div>
                        </div>

                        <div class="bg-white border border-slate-200 rounded-xl p-4">
                            <div class="flex items-center justify-between mb-3">
                                <h2 class="text-lg font-semibold text-slate-800">Your resume versions</h2>
                                <button type="button" id="clearHistoryBtn" class="text-sm text-slate-500 hover:text-slate-700">Clear offline cache</button>
                            </div>
                            <div id="historyList" class="space-y-2"></div>
                        </div>
                    </div>
                </section>
            </main>
        </div>

        <div id="toastContainer" class="fixed top-4 right-4 z-[100] w-11/12 max-w-sm space-y-2"></div>
        <div id="loadingOverlay" class="hidden fixed inset-0 z-[90] bg-slate-900/70 backdrop-blur-sm flex flex-col items-center justify-center">
            <div class="animate-spin rounded-full h-12 w-12 border-4 border-blue-400 border-t-transparent"></div>
            <p id="loadingText" class="mt-4 text-sm font-medium text-white text-center px-6">Working…</p>
        </div>

    <script>
        const tokenKey = 'resumeTailorToken';
        const userKey = 'resumeTailorUser';

        function showToast(message, type = 'info') {
            const palette = {
                success: 'bg-emerald-600 text-white',
                error: 'bg-red-600 text-white',
                warn: 'bg-amber-400 text-slate-900',
                info: 'bg-blue-600 text-white',
            };
            const container = document.getElementById('toastContainer');
            while (container.children.length >= 4) container.firstElementChild.remove();
            const toast = document.createElement('div');
            toast.className = (palette[type] || palette.info) +
                ' text-sm font-medium px-4 py-3 rounded-xl shadow-xl flex items-start justify-between gap-3 opacity-0 transition-opacity duration-300';
            const span = document.createElement('span');
            span.className = 'break-words';
            span.textContent = message;
            const closeBtn = document.createElement('button');
            closeBtn.type = 'button';
            closeBtn.textContent = '×';
            closeBtn.className = 'text-xl leading-none font-bold opacity-70 hover:opacity-100 shrink-0';
            closeBtn.addEventListener('click', () => toast.remove());
            toast.append(span, closeBtn);
            container.appendChild(toast);
            requestAnimationFrame(() => toast.classList.remove('opacity-0'));
            const lifetime = type === 'error' ? 7000 : type === 'warn' ? 6000 : 4000;
            setTimeout(() => {
                toast.classList.add('opacity-0');
                setTimeout(() => toast.remove(), 300);
            }, lifetime);
        }

        function networkErrorMessage() {
            return 'Network error. Please check your internet connection and try again.';
        }

        let loadingTimer = null;
        let loadingSteps = [];
        let loadingStepIndex = 0;

        function showLoading(steps) {
            loadingSteps = steps || [];
            loadingStepIndex = 0;
            document.getElementById('loadingText').textContent = loadingSteps[0] || 'Working…';
            document.getElementById('loadingOverlay').classList.remove('hidden');
            if (loadingTimer) clearInterval(loadingTimer);
            if (loadingSteps.length > 1) {
                loadingTimer = setInterval(() => {
                    loadingStepIndex = Math.min(loadingStepIndex + 1, loadingSteps.length - 1);
                    document.getElementById('loadingText').textContent = loadingSteps[loadingStepIndex];
                }, 6000);
            }
        }

        function hideLoading() {
            if (loadingTimer) { clearInterval(loadingTimer); loadingTimer = null; }
            document.getElementById('loadingOverlay').classList.add('hidden');
        }

        function setAuthState(token, user) {
            const authStatus = document.getElementById('authStatus');
            const logoutBtn = document.getElementById('logoutBtn');
            const dashboardPanel = document.getElementById('dashboardPanel');
            const dashboardUser = document.getElementById('dashboardUser');
            const dashboardActivity = document.getElementById('dashboardActivity');

            if (token && user) {
                localStorage.setItem(tokenKey, token);
                localStorage.setItem(userKey, JSON.stringify(user));
                authStatus.textContent = `Signed in as ${user.name}`;
                authStatus.classList.remove('hidden');
                logoutBtn.classList.remove('hidden');
                dashboardPanel.classList.remove('hidden');
                dashboardUser.textContent = user.name;
                loadUserGadgets();
                fetch('/api/dashboard', {
                    headers: { Authorization: `Bearer ${token}` }
                }).then(async response => {
                    if (!response.ok) return;
                    const data = await response.json();
                    const activity = data.recent_activity || [];
                    dashboardActivity.innerHTML = activity.length ? activity.map(item => `
                        <div class="border border-blue-200 rounded-lg p-2 bg-white">
                            <div class="font-medium">${item.title}</div>
                            <div class="text-xs text-blue-700">ATS: ${item.ats_score}%</div>
                            <div class="text-xs text-gray-600">${new Date(item.created_at).toLocaleString()}</div>
                        </div>
                    `).join('') : '<div class="text-gray-600">No optimizations yet.</div>';
                }).catch(() => {
                    dashboardActivity.innerHTML = '<div class="text-gray-600">No optimizations yet.</div>';
                    showToast('Could not refresh your dashboard. Your saved activity may be out of date.', 'warn');
                });
                return;
            }
            localStorage.removeItem(tokenKey);
            localStorage.removeItem(userKey);
            authStatus.textContent = 'Not signed in';
            authStatus.classList.remove('hidden');
            logoutBtn.classList.add('hidden');
            dashboardPanel.classList.add('hidden');
            dashboardUser.textContent = '';
            dashboardActivity.innerHTML = '';
            document.getElementById('appsList').innerHTML = '';
            renderProfileCard({});
            loadUserGadgets();
        }

        function hydrateAuthState() {
            const token = localStorage.getItem(tokenKey);
            const user = JSON.parse(localStorage.getItem(userKey) || 'null');
            setAuthState(token, user);
        }

        document.getElementById('logoutBtn').addEventListener('click', () => setAuthState(null, null));

        document.getElementById('registerForm').addEventListener('submit', async (event) => {
            event.preventDefault();
            const btn = event.target.querySelector('button[type="submit"]');
            const payload = {
                name: document.getElementById('registerName').value.trim(),
                email: document.getElementById('registerEmail').value.trim(),
                password: document.getElementById('registerPassword').value,
            };

            if (!payload.name || !payload.email || !payload.password) {
                showToast('Please fill in all fields to create your account.', 'warn');
                return;
            }
            if (payload.password.length < 8) {
                showToast('Warning: your password must be at least 8 characters long.', 'warn');
                return;
            }

            btn.disabled = true;
            btn.textContent = 'Creating account…';
            showLoading(['Creating your account…', 'Securing your password…']);

            try {
                const response = await fetch('/api/register', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload),
                });
                let data = {};
                try { data = await response.json(); } catch {}
                if (!response.ok) {
                    const detail = typeof data.detail === 'string' ? data.detail : 'Please check your details and try again.';
                    throw new Error(detail);
                }
                document.getElementById('registerForm').reset();
                showToast('Account created successfully! Log in below to start optimizing.', 'success');
            } catch (error) {
                showToast(error.name === 'TypeError' ? networkErrorMessage() : error.message, 'error');
            } finally {
                btn.disabled = false;
                btn.textContent = 'Create account';
                hideLoading();
            }
        });

        document.getElementById('loginForm').addEventListener('submit', async (event) => {
            event.preventDefault();
            const btn = event.target.querySelector('button[type="submit"]');
            const payload = {
                email: document.getElementById('loginEmail').value.trim(),
                password: document.getElementById('loginPassword').value,
            };

            if (!payload.email || !payload.password) {
                showToast('Please enter both your email and password.', 'warn');
                return;
            }
            if (payload.password.length < 8) {
                showToast('Warning: your password must be at least 8 characters long.', 'warn');
                return;
            }

            btn.disabled = true;
            btn.textContent = 'Signing in…';
            showLoading(['Signing you in…', 'Verifying your credentials…']);

            try {
                const response = await fetch('/api/login', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload),
                });
                let data = {};
                try { data = await response.json(); } catch {}
                if (!response.ok) {
                    const detail = typeof data.detail === 'string' ? data.detail : 'Login failed.';
                    throw new Error(detail);
                }
                setAuthState(data.token, data.user);
                document.getElementById('loginForm').reset();
                showToast('Signed in successfully. Welcome back, ' + data.user.name + '!', 'success');
            } catch (error) {
                showToast(error.name === 'TypeError' ? networkErrorMessage() : error.message, 'error');
            } finally {
                btn.disabled = false;
                btn.textContent = 'Sign in';
                hideLoading();
            }
        });

        function renderHistory(items) {
            const historyList = document.getElementById('historyList');
            if (!items.length) {
                historyList.innerHTML = '<div class="text-sm text-gray-500">No saved optimizations yet.</div>';
                return;
            }

            historyList.innerHTML = items.slice(0, 5).map(item => `
                <button type="button" data-markdown="${encodeURIComponent(item.markdown_resume || '')}" class="history-item w-full text-left rounded border border-gray-200 p-3 hover:bg-gray-50">
                    <div class="flex justify-between gap-4 text-sm">
                        <span class="font-medium text-gray-800">${item.label}</span>
                        <span class="text-blue-700 font-semibold">${item.ats_score}%</span>
                    </div>
                    <div class="text-xs text-gray-500 mt-1">${new Date(item.created_at).toLocaleString()}</div>
                </button>
            `).join('');

            document.querySelectorAll('.history-item').forEach(button => {
                button.addEventListener('click', () => {
                    const markdown = decodeURIComponent(button.dataset.markdown);
                    document.getElementById('markdownOutput').innerHTML = marked.parse(markdown);
                });
            });
        }

        function saveToHistory(label, atsScore, markdownResume) {
            const key = 'resumeTailorHistory';
            const current = JSON.parse(localStorage.getItem(key) || '[]');
            const next = [
                { label, ats_score: atsScore, markdown_resume: markdownResume, created_at: new Date().toISOString() },
                ...current,
            ].slice(0, 5);
            localStorage.setItem(key, JSON.stringify(next));
            renderHistory(next);
        }

        document.getElementById('clearHistoryBtn').addEventListener('click', () => {
            localStorage.removeItem('resumeTailorHistory');
            renderHistory([]);
        });

        hydrateAuthState();
        loadHistory();

        document.getElementById('resumeForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const btn = document.getElementById('submitBtn');
            const results = document.getElementById('results');

            const jobUrl = document.getElementById('jobUrl').value.trim();
            const jobDescriptionText = document.getElementById('jobDescriptionText').value.trim();
            const pastedResume = document.getElementById('resumeTextPaste').value.trim();
            const style = document.getElementById('styleSelect').value;
            const file = document.getElementById('resumeFile').files[0];

            if (!jobUrl && !jobDescriptionText) {
                showToast('Please provide a job posting URL or paste the job description.', 'warn');
                return;
            }
            if (file) {
                const fileName = file.name.toLowerCase();
                const isPdf = file.type === 'application/pdf' || fileName.endsWith('.pdf');
                const isDocx = (file.type || '').includes('wordprocessingml') || fileName.endsWith('.docx');
                const isTxt = file.type === 'text/plain' || fileName.endsWith('.txt');
                if (!isPdf && !isDocx && !isTxt) {
                    showToast('Unsupported file type. Only PDF, DOCX and TXT are accepted (legacy .doc: save as PDF first).', 'warn');
                    return;
                }
                if (file.size > 5 * 1024 * 1024) {
                    showToast('Your resume file is over 5 MB. Please upload a smaller file.', 'warn');
                    return;
                }
            } else if (!pastedResume) {
                showToast('Please attach a resume file (PDF, DOCX, TXT) or paste your resume text.', 'warn');
                return;
            }

            btn.disabled = true;
            results.classList.add('hidden');
            showLoading([
                'Uploading your resume…',
                'Reading the job requirements…',
                'The AI is tailoring your resume. This usually takes 20–30 seconds…',
                'Almost done — polishing your new resume and calculating your ATS score…',
            ]);

            try {
                const formData = new FormData();
                formData.append('job_url', jobUrl);
                formData.append('job_description_text', jobDescriptionText);
                formData.append('style', style);
                if (file) formData.append('resume_file', file);
                if (pastedResume) formData.append('resume_text', pastedResume);

                const headers = {};
                const token = localStorage.getItem('resumeTailorToken');
                if (token) headers['Authorization'] = `Bearer ${token}`;

                const response = await fetch('/api/optimize', { method: 'POST', body: formData, headers });
                if (!response.ok) {
                    const text = await response.text();
                    let msg = `Server error ${response.status}`;
                    try { msg = JSON.parse(text).detail || msg; } catch {}
                    if (response.status === 429) {
                        showToast(msg, 'warn');
                    } else if (response.status === 401) {
                        showToast('Please log in before optimizing your resume.', 'warn');
                    } else {
                        showToast(msg, 'error');
                    }
                    return;
                }
                const data = await response.json();
                renderResult(data);
                saveToHistory(jobUrl || jobDescriptionText.slice(0, 40) || 'Resume optimization', data.ats_score || 0, data.markdown_resume || '');
                showToast('Resume optimized successfully! Scroll up for scores and the tailored result.', 'success');

                const currentToken = localStorage.getItem('resumeTailorToken');
                if (currentToken) {
                    try {
                        const dashboardResponse = await fetch('/api/dashboard', {
                            headers: { Authorization: `Bearer ${currentToken}` }
                        });
                        if (dashboardResponse.ok) {
                            const dashboardData = await dashboardResponse.json();
                            const dashboardActivity = document.getElementById('dashboardActivity');
                            const activity = dashboardData.recent_activity || [];
                            dashboardActivity.innerHTML = activity.length ? activity.map(item => `
                                <div class="border border-blue-200 rounded-lg p-2 bg-white">
                                    <div class="font-medium">${item.title}</div>
                                    <div class="text-xs text-blue-700">ATS: ${item.ats_score}%</div>
                                    <div class="text-xs text-gray-600">${new Date(item.created_at).toLocaleString()}</div>
                                </div>
                            `).join('') : '<div class="text-gray-600">No optimizations yet.</div>';
                        }
                    } catch {
                        showToast('Saved your optimization, but could not refresh the dashboard Activity.', 'warn');
                    }
                }
            } catch (error) {
                if (error.name === 'TypeError') {
                    showToast(networkErrorMessage(), 'error');
                } else {
                    showToast(error.message || 'Something went wrong. Please try again.', 'error');
                }
            } finally {
                btn.disabled = false;
                hideLoading();
            }
        });

        // ==================== Comparison, export, extras, history, profile, tracker ====================

        const STYLE_LABELS = { standard: 'Standard', ats: 'ATS-only', senior: 'Senior', startup: 'Startup', technical: 'Technical', career_switch: 'Career switch' };
        let lastResult = null;

        function escHtml(s) { return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;'); }

        function authHeaders() {
            const token = localStorage.getItem('resumeTailorToken');
            return token ? { Authorization: `Bearer ${token}` } : {};
        }

        function highlightKeywords(mdSource, keywords) {
            let out = mdSource;
            const BS = String.fromCharCode(92); // backslash, kept escape-free on purpose
            keywords.slice(0, 40).forEach(kw => {
                const safe = kw.replace(/[^a-zA-Z0-9]/g, m => BS + m);
                out = out.replace(new RegExp('(^|[^' + BS + 'w>])(' + safe + ')(?=$|[^' + BS + 'w<])', 'gi'), '$1<mark class="bg-amber-200 text-slate-900 rounded px-0.5">$2</mark>');
            });
            return out;
        }

        function renderResult(data) {
            lastResult = data;
            document.getElementById('results').classList.remove('hidden');

            const warn = document.getElementById('scoreWarning');
            if (data.warning) {
                warn.textContent = data.warning;
                warn.classList.remove('hidden');
            } else {
                warn.classList.add('hidden');
            }

            const before = data.ats_score ?? 0;
            const after = data.ats_score_tailored;
            document.getElementById('atsScoreBefore').textContent = `${before}%`;
            document.getElementById('atsScore').textContent = `${after ?? before}%`;
            const delta = document.getElementById('atsDelta');
            if (after !== null && after !== undefined && after > before) {
                delta.textContent = `+${(after - before).toFixed(1)} pts`;
                delta.classList.remove('hidden');
                showToast(`ATS score improved ${before}% → ${after}% (+${(after - before).toFixed(1)} pts).`, 'success');
            } else {
                delta.classList.add('hidden');
            }

            const counts = data.coverage_counts || {};
            document.getElementById('strongCount').textContent = counts.strong ?? 0;
            document.getElementById('weakCount').textContent = counts.weak ?? 0;
            document.getElementById('missingCount').textContent = counts.missing ?? 0;

            document.getElementById('recommendationList').innerHTML =
                (data.recommendations || []).map(r => `<li>${escHtml(r)}</li>`).join('') || '<li>—</li>';

            const strong = (data.matched_keywords || []).filter(k => (k.coverage || 'strong') === 'strong').map(k => k.keyword);
            const weak = (data.matched_keywords || []).filter(k => k.coverage === 'weak').map(k => k.keyword);
            const missing = data.missing_keywords || [];
            const side = (title, items, cls) => items.length
                ? `<div><div class="text-xs font-bold uppercase text-slate-500 mb-1">${title}</div><div class="flex flex-wrap gap-1.5">${items.map(k => `<span class="${cls}">${escHtml(k)}</span>`).join('')}</div></div>`
                : '';
            document.getElementById('keywordsPanel').innerHTML = [
                side('Strong matches', strong, 'bg-emerald-100 text-emerald-800 px-2 py-1 rounded-full text-xs font-medium'),
                side('Weak coverage (related terms)', weak, 'bg-amber-100 text-amber-800 px-2 py-1 rounded-full text-xs font-medium'),
                side('Missing from your resume', missing, 'bg-red-100 text-red-700 px-2 py-1 rounded-full text-xs font-medium'),
            ].join('') || '<span class="text-slate-500">No keyword data for this run.</span>';

            document.getElementById('originalResumeText').textContent = data.original_resume_text || '';
            const badge = document.getElementById('styleBadge');
            if (data.style && data.style !== 'standard') {
                badge.textContent = STYLE_LABELS[data.style] || data.style;
                badge.classList.remove('hidden');
            } else {
                badge.classList.add('hidden');
            }

            const md = data.markdown_resume || '';
            document.getElementById('markdownOutput').innerHTML = marked.parse(highlightKeywords(md, strong));

            document.getElementById('results').scrollIntoView({ behavior: 'smooth' });
        }

        function currentMarkdown() {
            return lastResult ? (lastResult.markdown_resume || '') : (document.getElementById('markdownOutput').textContent || '');
        }

        function downloadBlob(content, filename, mime) {
            const blob = new Blob([content], { type: mime });
            const a = document.createElement('a');
            a.href = URL.createObjectURL(blob);
            a.download = filename;
            a.click();
            URL.revokeObjectURL(a.href);
        }

        document.getElementById('copyBtn').addEventListener('click', async () => {
            try {
                await navigator.clipboard.writeText(currentMarkdown());
                showToast('Tailored resume copied to clipboard.', 'success');
            } catch {
                showToast('Copy failed — your browser blocked clipboard access.', 'error');
            }
        });
        document.getElementById('downloadMdBtn').addEventListener('click', () => downloadBlob(currentMarkdown(), 'tailored_resume.md', 'text/markdown'));
        document.getElementById('downloadTxtBtn').addEventListener('click', () => downloadBlob(currentMarkdown().replace(/[#*]/g, ''), 'tailored_resume.txt', 'text/plain'));

        document.getElementById('downloadDocxBtn').addEventListener('click', async () => {
            if (!currentMarkdown().trim()) { showToast('Nothing to export yet.', 'warn'); return; }
            showLoading(['Generating your DOCX…']);
            try {
                const response = await fetch('/api/export/docx', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json', ...authHeaders() },
                    body: JSON.stringify({ markdown_resume: currentMarkdown() }),
                });
                if (!response.ok) throw new Error('DOCX export failed. Please try again.');
                const blob = await response.blob();
                const a = document.createElement('a');
                a.href = URL.createObjectURL(blob);
                a.download = 'tailored_resume.docx';
                a.click();
                URL.revokeObjectURL(a.href);
                showToast('DOCX downloaded.', 'success');
            } catch (e) {
                showToast(e.name === 'TypeError' ? networkErrorMessage() : e.message, 'error');
            } finally {
                hideLoading();
            }
        });

        document.getElementById('printPdfBtn').addEventListener('click', () => {
            const md = currentMarkdown();
            if (!md.trim()) { showToast('Nothing to print yet.', 'warn'); return; }
            const printWindow = window.open('', '_blank');
            if (!printWindow) { showToast('Your browser blocked the print window. Allow pop-ups for this site and retry.', 'warn'); return; }
            printWindow.document.write('<html><head><title>Tailored Resume</title><style>body{font-family:Georgia,serif;line-height:1.45;max-width:800px;margin:40px auto;color:#111;padding:0 24px}h1{font-size:1.6em;margin:.2em 0}h2{border-bottom:1px solid #ccc;padding-bottom:3px;margin-top:1.2em}li{margin:.25em 0}@media print{body{margin:0}}</style></head><body>' + marked.parse(md) + '</body></html>');
            printWindow.document.close();
            printWindow.focus();
            setTimeout(() => printWindow.print(), 700);
        });

        async function runExtras(kind) {
            const path = kind === 'cover_letter' ? '/api/cover-letter' : '/api/interview-prep';
            const btn = (kind === 'cover_letter') ? document.getElementById('coverLetterBtn') : document.getElementById('interviewPrepBtn');
            if (!localStorage.getItem('resumeTailorToken')) { showToast('Please log in first.', 'warn'); return; }

            let activityId = (lastResult && lastResult.activity_id && lastResult.activity_id !== -1) ? lastResult.activity_id : null;
            if (!activityId) {
                try {
                    const resp = await fetch('/api/history', { headers: authHeaders() });
                    if (resp.ok) {
                        const hist = await resp.json();
                        if (hist.history && hist.history.length) activityId = hist.history[0].id;
                    }
                } catch {}
            }
            if (!activityId) { showToast('Optimize a resume first — extras are generated from your latest saved result.', 'warn'); return; }

            btn.disabled = true;
            showLoading(kind === 'cover_letter'
                ? ['Drafting your tailored cover letter…', 'Weaving in your strongest achievements…']
                : ['Building your interview preparation brief…', 'Composing STAR answers from your resume…']);
            try {
                const response = await fetch(path, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json', ...authHeaders() },
                    body: JSON.stringify({ activity_id: activityId }),
                });
                const data = await response.json().catch(() => ({}));
                if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Generation failed. Please try again.');
                document.getElementById('extrasOutput').classList.remove('hidden');
                document.getElementById('extrasTitle').textContent = kind === 'cover_letter' ? 'Cover letter' : 'Interview preparation brief';
                const content = data.cover_letter || data.interview_prep || '';
                const host = document.getElementById('extrasContent');
                host.innerHTML = marked.parse(content);
                host.dataset.raw = content;
                showToast(kind === 'cover_letter' ? 'Cover letter ready below.' : 'Interview brief ready below.', 'success');
                document.getElementById('extrasOutput').scrollIntoView({ behavior: 'smooth' });
            } catch (e) {
                showToast(e.name === 'TypeError' ? networkErrorMessage() : e.message, 'error');
            } finally {
                btn.disabled = false;
                hideLoading();
            }
        }
        document.getElementById('coverLetterBtn').addEventListener('click', () => runExtras('cover_letter'));
        document.getElementById('interviewPrepBtn').addEventListener('click', () => runExtras('interview_prep'));
        document.getElementById('extrasCopyBtn').addEventListener('click', async () => {
            const host = document.getElementById('extrasContent');
            try {
                await navigator.clipboard.writeText(host.dataset.raw || host.textContent);
                showToast('Copied.', 'success');
            } catch {
                showToast('Copy failed.', 'error');
            }
        });

        async function loadHistory() {
            const token = localStorage.getItem('resumeTailorToken');
            const list = document.getElementById('historyList');
            if (!token) {
                renderHistory(JSON.parse(localStorage.getItem('resumeTailorHistory') || '[]'));
                return;
            }
            try {
                const response = await fetch('/api/history', { headers: authHeaders() });
                if (!response.ok) throw new Error('history');
                const data = await response.json();
                const items = data.history || [];
                list.innerHTML = items.length ? items.map(item => `
                    <div class="border border-slate-200 rounded-lg p-3 bg-white flex items-center justify-between gap-3">
                        <div class="min-w-0">
                            <div class="font-medium text-slate-800 truncate">${escHtml(item.title || 'Optimization')}</div>
                            <div class="text-xs text-slate-500 mt-0.5">${new Date(item.created_at).toLocaleString()} · ${STYLE_LABELS[item.style] || item.style} · ATS ${item.ats_score ?? 0}%${item.ats_score_tailored != null ? ' → ' + item.ats_score_tailored + '%' : ''}</div>
                        </div>
                        <div class="flex gap-2 shrink-0">
                            <button type="button" data-load="${item.id}" class="historyLoad bg-blue-50 text-blue-700 text-xs font-semibold px-2 py-1.5 rounded-md hover:bg-blue-100">Load into form</button>
                            <button type="button" data-del="${item.id}" class="historyDel text-slate-400 hover:text-red-600 text-xs font-semibold px-2 py-1.5">Delete</button>
                        </div>
                    </div>`).join('') : '<div class="text-sm text-gray-500">No saved versions yet — run an optimization.</div>';
                list.querySelectorAll('.historyLoad').forEach(b => b.addEventListener('click', () => loadVersion(b.dataset.load)));
                list.querySelectorAll('.historyDel').forEach(b => b.addEventListener('click', () => deleteVersion(b.dataset.del)));
            } catch {
                renderHistory(JSON.parse(localStorage.getItem('resumeTailorHistory') || '[]'));
            }
        }

        async function loadVersion(id) {
            showLoading(['Loading version…']);
            try {
                const response = await fetch(`/api/history/${id}`, { headers: authHeaders() });
                const data = await response.json();
                if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Could not load this version.');
                document.getElementById('jobUrl').value = '';
                if (data.job_description) document.getElementById('jobDescriptionText').value = data.job_description;
                document.getElementById('resumeTextPaste').value = data.original_resume || '';
                if (data.style && document.getElementById('styleSelect').querySelector(`option[value="${data.style}"]`)) {
                    document.getElementById('styleSelect').value = data.style;
                }
                document.getElementById('optimizer').scrollIntoView({ behavior: 'smooth' });
                showToast('Version loaded into the form — press Optimize Resume to regenerate it.', 'success');
            } catch (e) {
                showToast(e.name === 'TypeError' ? networkErrorMessage() : e.message, 'error');
            } finally {
                hideLoading();
            }
        }

        async function deleteVersion(id) {
            try {
                await fetch(`/api/history/${id}`, { method: 'DELETE', headers: authHeaders() });
                showToast('Version deleted.', 'info');
                loadHistory();
            } catch {
                showToast('Could not delete this version.', 'error');
            }
        }

        function renderProfileCard(profile) {
            const map = [['Target role', profile.target_role], ['Experience', profile.years_experience], ['Industry', profile.industry], ['Locations', profile.locations], ['Emphasize', profile.skills_emphasis], ['Salary', profile.salary_expectations]];
            const rows = map.filter(([, v]) => v).map(([k, v]) =>
                `<span class="inline-flex items-center gap-1 bg-white border border-slate-200 rounded-full px-2.5 py-1 text-xs"><span class="text-slate-400">${k}:</span><span class="font-medium text-slate-800">${escHtml(v)}</span></span>`
            ).join('');
            document.getElementById('profileView').innerHTML = rows || 'Not set yet — click Edit to give the AI your target role, experience and focus skills.';
        }

        async function editProfileOpen() {
            if (!localStorage.getItem('resumeTailorToken')) { showToast('Log in to save a targeting profile.', 'warn'); return; }
            let profile = {};
            try {
                const resp = await fetch('/api/profile', { headers: authHeaders() });
                if (resp.ok) profile = await resp.json();
            } catch {}
            if (profile && typeof profile === 'object' && !profile.detail) {
                document.getElementById('profTargetRole').value = profile.target_role || '';
                document.getElementById('profYears').value = profile.years_experience || '';
                document.getElementById('profIndustry').value = profile.industry || '';
                document.getElementById('profLocations').value = profile.locations || '';
                document.getElementById('profSkills').value = profile.skills_emphasis || '';
                document.getElementById('profSalary').value = profile.salary_expectations || '';
            }
            document.getElementById('profileForm').classList.remove('hidden');
        }

        async function saveTargetingProfile() {
            const btn = document.getElementById('profileSaveBtn');
            const payload = {
                target_role: document.getElementById('profTargetRole').value.trim(),
                years_experience: document.getElementById('profYears').value,
                industry: document.getElementById('profIndustry').value.trim(),
                locations: document.getElementById('profLocations').value.trim(),
                skills_emphasis: document.getElementById('profSkills').value.trim(),
                salary_expectations: document.getElementById('profSalary').value.trim(),
            };
            btn.disabled = true;
            try {
                const response = await fetch('/api/profile', {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json', ...authHeaders() },
                    body: JSON.stringify(payload),
                });
                const data = await response.json().catch(() => ({}));
                if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Could not save your profile.');
                renderProfileCard(payload);
                document.getElementById('profileForm').classList.add('hidden');
                showToast('Targeting profile saved. It will be applied to your next optimization.', 'success');
            } catch (e) {
                showToast(e.name === 'TypeError' ? networkErrorMessage() : e.message, 'error');
            } finally {
                btn.disabled = false;
            }
        }

        async function loadProfileCard() {
            if (!localStorage.getItem('resumeTailorToken')) { renderProfileCard({}); return; }
            try {
                const resp = await fetch('/api/profile', { headers: authHeaders() });
                const profile = resp.ok ? await resp.json() : {};
                renderProfileCard(profile && typeof profile === 'object' && !profile.detail ? profile : {});
            } catch {
                renderProfileCard({});
            }
        }

        document.getElementById('editProfileBtn').addEventListener('click', editProfileOpen);
        document.getElementById('profileCancelBtn').addEventListener('click', () => document.getElementById('profileForm').classList.add('hidden'));
        document.getElementById('profileSaveBtn').addEventListener('click', saveTargetingProfile);

        async function loadApplications() {
            const token = localStorage.getItem('resumeTailorToken');
            const list = document.getElementById('appsList');
            list.innerHTML = '';
            if (!token) return;
            try {
                const response = await fetch('/api/applications', { headers: authHeaders() });
                if (!response.ok) return;
                const data = await response.json();
                const apps = data.applications || [];
                list.innerHTML = apps.length ? apps.map(item => {
                    let followUp = '';
                    if (item.status === 'applied' && item.applied_at) {
                        const days = Math.floor((Date.now() - new Date(item.applied_at).getTime()) / 86400000);
                        if (days >= 14) followUp = `<div class="text-amber-600 font-medium text-xs mt-2">Consider following up — ${days} days since you applied.</div>`;
                    }
                    const statusSel = `<select data-app-id="${item.id}" class="appStatus text-xs border border-blue-200 rounded-md px-1.5 py-1 bg-white text-gray-900 outline-none">${["saved", "applied", "interview", "offer", "rejected"].map(s => `<option value="${s}"${s === item.status ? ' selected' : ''}>${s}</option>`).join('')}</select>`;
                    return `<div class="border border-blue-200 rounded-lg p-3 bg-white">
                        <div class="flex flex-wrap items-center justify-between gap-2">
                            <div class="min-w-0">
                                <div class="font-semibold text-slate-800 truncate">${escHtml(item.title)}${item.company ? ' · ' + escHtml(item.company) : ''}</div>
                                ${item.job_url ? `<a href="${escHtml(item.job_url)}" target="_blank" rel="noopener" class="text-xs text-blue-600 hover:underline break-all">job link ↗</a>` : ''}
                                <div class="text-xs text-slate-500 mt-1">Updated ${new Date(item.applied_at).toLocaleString()}</div>
                            </div>
                            <div class="flex items-center gap-2 shrink-0">${statusSel}
                                <button type="button" data-app-del="${item.id}" class="appDel text-slate-400 hover:text-red-600 text-xs font-semibold px-1">Delete</button></div>
                        </div>
                        ${item.notes ? `<div class="mt-2 text-xs text-slate-600 bg-slate-50 border border-slate-100 rounded p-2">${escHtml(item.notes)}</div>` : ''}
                        ${followUp}
                    </div>`;
                }).join('') : '<div class="text-sm text-slate-500">No tracked applications yet — add one above.</div>';
                list.querySelectorAll('.appStatus').forEach(sel => sel.addEventListener('change', () => updateApplicationStatus(sel.dataset.appId, sel.value)));
                list.querySelectorAll('.appDel').forEach(b => b.addEventListener('click', () => deleteApplication(b.dataset.appDel)));
            } catch {}
        }

        async function updateApplicationStatus(id, status) {
            try {
                await fetch(`/api/applications/${id}`, { method: 'PATCH', headers: { 'Content-Type': 'application/json', ...authHeaders() }, body: JSON.stringify({ status }) });
                showToast('Status updated.', 'success');
                loadApplications();
            } catch {
                showToast('Could not update this application.', 'error');
            }
        }

        async function deleteApplication(id) {
            try {
                await fetch(`/api/applications/${id}`, { method: 'DELETE', headers: authHeaders() });
                showToast('Application deleted.', 'info');
                loadApplications();
            } catch {
                showToast('Could not delete this application.', 'error');
            }
        }

        document.getElementById('showAppFormBtn').addEventListener('click', () => document.getElementById('appForm').classList.toggle('hidden'));
        document.getElementById('cancelAppBtn').addEventListener('click', () => document.getElementById('appForm').classList.add('hidden'));
        document.getElementById('appForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            if (!localStorage.getItem('resumeTailorToken')) { showToast('Please log in to track applications.', 'warn'); return; }
            const payload = {
                title: document.getElementById('appTitle').value.trim(),
                company: document.getElementById('appCompany').value.trim(),
                job_url: document.getElementById('appUrl').value.trim(),
                status: document.getElementById('appStatus').value,
                notes: document.getElementById('appNotes').value.trim(),
            };
            if (!payload.title) { showToast('Job title is required.', 'warn'); return; }
            try {
                const response = await fetch('/api/applications', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json', ...authHeaders() },
                    body: JSON.stringify(payload),
                });
                const data = await response.json().catch(() => ({}));
                if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Could not save this application.');
                document.getElementById('appForm').reset();
                document.getElementById('appForm').classList.add('hidden');
                showToast('Application tracked.', 'success');
                loadApplications();
            } catch (err) {
                showToast(err.name === 'TypeError' ? networkErrorMessage() : err.message, 'error');
            }
        });

        function loadUserGadgets() {
            loadProfileCard();
            loadHistory();
            loadApplications();
        }
    </script>
</body>
</html>"""
    return HTMLResponse(
        content=html_content,
        headers={"Cache-Control": "no-store, must-revalidate"},
    )
