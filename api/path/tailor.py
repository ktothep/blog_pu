import io
import os
import re

import PyPDF2
import anthropic
import requests
from bs4 import BeautifulSoup
from fastapi import HTTPException, APIRouter, Form, Request, UploadFile, File
from starlette.responses import HTMLResponse

from api.limiter import limiter
from api.database import log_interaction
from api.auth import get_authenticated_user, save_user_activity

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
- Keep contact info on separate lines: Email | Phone | LinkedIn | Location
- Avoid text inside parentheses for job titles — ATS may strip them

### FORMAT RULES
- Start directly with the candidate's name as a top-level heading (`#`)
- Use `##` for section headings, `-` for all bullet points
- Standard section order: Name & Contact → Summary → Experience → Skills → Education → Certifications/Projects
- One bullet point = one achievement, max 2 lines, starts with action verb, ends with quantified impact where possible"""


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


def calculate_ats_score(job_description: str, resume_text: str) -> float:
    jd_terms = _tokenize_keywords(job_description)
    matched = extract_keyword_hits(job_description, resume_text)
    if not jd_terms:
        return 0.0
    return round((len(matched) / len(jd_terms)) * 100, 1)


def analyze_resume_for_ats(job_description: str, resume_text: str) -> dict:
    jd_terms = _tokenize_keywords(job_description)
    matched = extract_keyword_hits(job_description, resume_text)
    missing = sorted(jd_terms - matched)
    return {
        "score": calculate_ats_score(job_description, resume_text),
        "matched_keywords": sorted(matched),
        "missing_keywords": missing[:10],
    }


def scrape_url(url: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    resp = requests.get(url, headers=headers, timeout=15)
    if resp.status_code != 200:
        return ""
    soup = BeautifulSoup(resp.content, "html.parser")
    for tag in soup(["script", "style", "meta", "noscript", "head"]):
        tag.decompose()
    lines = [l for l in soup.get_text(separator="\n", strip=True).splitlines() if l.strip()]
    return "\n".join(lines)


def parse_file(file_bytes: bytes, filename: str) -> str:
    if filename.endswith(".pdf"):
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        return "\n".join(p.extract_text() or "" for p in reader.pages)
    elif filename.endswith(".txt"):
        return file_bytes.decode("utf-8")
    else:
        raise ValueError("Only PDF and TXT files are supported.")


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
@limiter.limit("2/hour")
async def optimize_resume(
    request: Request,
    job_url: str = Form(default=""),
    job_description_text: str = Form(default=""),
    resume_file: UploadFile = File(...)
):
    auth_user = get_authenticated_user(request)

    resume_text = None
    job_description_for_analysis = ""
    try:
        if not job_url.strip() and not job_description_text.strip():
            raise HTTPException(status_code=400, detail="Please provide either a job posting URL or paste the job description.")

        file_bytes = await resume_file.read()
        try:
            resume_text = parse_file(file_bytes, resume_file.filename.lower())
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        if not resume_text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from the uploaded file.")

        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        if job_description_text.strip():
            job_description_for_analysis = job_description_text.strip()
            user_content = f"Here is the job description:\n\n{job_description_for_analysis}\n\nHere is my current resume:\n\n{resume_text}"
            if job_url.strip():
                user_content = f"Job posting URL: {job_url}\n\n" + user_content
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_content}]
            )
            final_text = next(b.text for b in response.content if hasattr(b, "text"))
            ats_summary = analyze_resume_for_ats(job_description_for_analysis, resume_text)
            save_user_activity(auth_user["sub"], job_url or "Resume optimization", ats_summary["score"])
            log_interaction(job_url or "text-input", resume_file.filename, resume_text, "success")
            return {
                "status": "success",
                "markdown_resume": final_text,
                "original_resume_text": resume_text,
                "ats_score": ats_summary["score"],
                "matched_keywords": ats_summary["matched_keywords"],
                "missing_keywords": ats_summary["missing_keywords"],
            }

        messages = [
            {
                "role": "user",
                "content": f"Please tailor my resume for this job posting: {job_url}\n\nHere is my current resume:\n\n{resume_text}"
            }
        ]
        final_job_description = ""

        while True:
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages
            )

            if response.stop_reason == "tool_use":
                tool_use_block = next(b for b in response.content if b.type == "tool_use")
                final_job_description = scrape_url(tool_use_block.input["url"])

                if not final_job_description:
                    final_job_description = "Could not retrieve the job description from this URL. Please tailor the resume based on the URL context and any visible keywords."

                messages.append({"role": "assistant", "content": response.content})
                messages.append({
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": tool_use_block.id, "content": final_job_description}]
                })

            elif response.stop_reason == "end_turn":
                final_text = next(b.text for b in response.content if hasattr(b, "text"))
                ats_summary = analyze_resume_for_ats(final_job_description or job_url, resume_text)
                save_user_activity(auth_user["sub"], job_url or "Resume optimization", ats_summary["score"])
                log_interaction(job_url, resume_file.filename, resume_text, "success")
                return {
                    "status": "success",
                    "markdown_resume": final_text,
                    "original_resume_text": resume_text,
                    "ats_score": ats_summary["score"],
                    "matched_keywords": ats_summary["matched_keywords"],
                    "missing_keywords": ats_summary["missing_keywords"],
                }

            else:
                raise HTTPException(status_code=500, detail=f"Unexpected stop reason: {response.stop_reason}")

    except HTTPException as e:
        log_interaction(job_url or "text-input", resume_file.filename, resume_text, "error", e.detail)
        raise
    except Exception as e:
        if "529" in str(e) or "overloaded" in str(e).lower():
            msg = "The AI service is temporarily overloaded. Please try again in a moment."
            log_interaction(job_url or "text-input", resume_file.filename, resume_text, "error", msg)
            raise HTTPException(status_code=503, detail=msg)
        log_interaction(job_url or "text-input", resume_file.filename, resume_text, "error", f"{type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


@route_tailor.get("/")
async def serve_ui():
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Resume Tailor</title>
    <script src="https://cdn.tailwindcss.com"></script>
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
                            <input id="registerName" type="text" placeholder="Full name" class="w-full px-4 py-3 border border-slate-300 rounded-xl focus:ring-2 focus:ring-blue-500 outline-none" required>
                            <input id="registerEmail" type="email" placeholder="Email" class="w-full px-4 py-3 border border-slate-300 rounded-xl focus:ring-2 focus:ring-blue-500 outline-none" required>
                            <input id="registerPassword" type="password" placeholder="Password" class="w-full px-4 py-3 border border-slate-300 rounded-xl focus:ring-2 focus:ring-blue-500 outline-none" required>
                            <button type="submit" class="w-full bg-blue-600 text-white font-semibold py-3 rounded-xl hover:bg-blue-700 transition-colors">Create account</button>
                        </form>
                    </div>

                    <div class="border border-slate-200 rounded-2xl p-6 bg-white shadow-sm">
                        <h2 class="text-2xl font-bold text-slate-900 mb-5">Welcome back</h2>
                        <form id="loginForm" class="space-y-3">
                            <input id="loginEmail" type="email" placeholder="Email" class="w-full px-4 py-3 border border-slate-300 rounded-xl focus:ring-2 focus:ring-blue-500 outline-none" required>
                            <input id="loginPassword" type="password" placeholder="Password" class="w-full px-4 py-3 border border-slate-300 rounded-xl focus:ring-2 focus:ring-blue-500 outline-none" required>
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
                </section>

                <section id="optimizer" class="rounded-3xl border border-slate-200 bg-white p-6 md:p-8 shadow-sm">
                    <div class="mb-6">
                        <p class="text-sm uppercase tracking-[0.18em] text-blue-600 font-semibold">Resume optimizer</p>
                        <h2 class="mt-2 text-3xl font-black text-slate-900">Optimize your resume for the next role</h2>
                    </div>

                    <form id="resumeForm" class="space-y-6">
                        <div>
                            <label class="block text-sm font-medium text-slate-700 mb-2">Job Posting URL <span class="text-slate-400 font-normal">(optional if you paste below)</span></label>
                            <input type="url" id="jobUrl" placeholder="https://linkedin.com/jobs/..."
                                   class="w-full px-4 py-3 border border-slate-300 rounded-xl focus:ring-2 focus:ring-blue-500 outline-none">
                        </div>
                        <div>
                            <label class="block text-sm font-medium text-slate-700 mb-2">
                                Job Description
                                <span class="text-slate-400 font-normal">(paste here if URL can't be parsed)</span>
                            </label>
                            <textarea id="jobDescriptionText" rows="6" placeholder="Paste the full job description here..."
                                      class="w-full px-4 py-3 border border-slate-300 rounded-xl focus:ring-2 focus:ring-blue-500 outline-none resize-y text-sm"></textarea>
                        </div>
                        <div>
                            <label class="block text-sm font-medium text-slate-700 mb-2">Your Current Resume (PDF/TXT)</label>
                            <input type="file" id="resumeFile" required accept=".pdf,.txt"
                                   class="w-full px-4 py-3 border border-slate-300 rounded-xl file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700">
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
                        <div class="grid md:grid-cols-2 gap-6">
                            <div class="bg-blue-50 border border-blue-100 rounded-xl p-5">
                                <div class="text-sm text-blue-700 font-semibold uppercase tracking-wide">ATS Match Score</div>
                                <div id="atsScore" class="text-3xl font-bold text-blue-900 mt-2">0%</div>
                                <div id="atsTags" class="mt-4 flex flex-wrap gap-2"></div>
                            </div>
                            <div class="bg-slate-50 border border-slate-200 rounded-xl p-5">
                                <div class="text-sm text-slate-700 font-semibold uppercase tracking-wide">Missing Keywords</div>
                                <div id="missingKeywords" class="mt-2 text-sm text-slate-700"></div>
                            </div>
                        </div>

                        <div class="grid md:grid-cols-2 gap-6">
                            <div class="bg-slate-50 p-4 rounded-xl border border-slate-200">
                                <h2 class="text-lg font-semibold text-slate-800 mb-3">Original Resume</h2>
                                <pre id="originalResumeText" class="whitespace-pre-wrap text-sm text-slate-700 max-h-80 overflow-auto"></pre>
                            </div>
                            <div class="bg-slate-50 p-4 rounded-xl border border-slate-200">
                                <h2 class="text-lg font-semibold text-slate-800 mb-3">Tailored Resume</h2>
                                <div id="markdownOutput" class="prose max-w-none text-sm"></div>
                            </div>
                        </div>

                        <div class="bg-white border border-slate-200 rounded-xl p-4">
                            <div class="flex items-center justify-between mb-3">
                                <h2 class="text-lg font-semibold text-slate-800">Recent Optimizations</h2>
                                <button type="button" id="clearHistoryBtn" class="text-sm text-slate-500 hover:text-slate-700">Clear</button>
                            </div>
                            <div id="historyList" class="space-y-2"></div>
                        </div>
                    </div>
                </section>
            </main>
        </div>

    <script>
        const tokenKey = 'resumeTailorToken';
        const userKey = 'resumeTailorUser';

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
        }

        function hydrateAuthState() {
            const token = localStorage.getItem(tokenKey);
            const user = JSON.parse(localStorage.getItem(userKey) || 'null');
            setAuthState(token, user);
        }

        document.getElementById('logoutBtn').addEventListener('click', () => setAuthState(null, null));

        document.getElementById('registerForm').addEventListener('submit', async (event) => {
            event.preventDefault();
            const payload = {
                name: document.getElementById('registerName').value,
                email: document.getElementById('registerEmail').value,
                password: document.getElementById('registerPassword').value,
            };

            try {
                const response = await fetch('/api/register', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload),
                });
                const data = await response.json();
                if (!response.ok) throw new Error(data.detail || 'Registration failed.');
                alert('Registration successful. Please log in.');
                document.getElementById('registerForm').reset();
            } catch (error) {
                alert('Registration error: ' + error.message);
            }
        });

        document.getElementById('loginForm').addEventListener('submit', async (event) => {
            event.preventDefault();
            const payload = {
                email: document.getElementById('loginEmail').value,
                password: document.getElementById('loginPassword').value,
            };

            try {
                const response = await fetch('/api/login', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload),
                });
                const data = await response.json();
                if (!response.ok) throw new Error(data.detail || 'Login failed.');
                setAuthState(data.token, data.user);
                document.getElementById('loginForm').reset();
                alert('Signed in successfully.');
            } catch (error) {
                alert('Login error: ' + error.message);
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
        renderHistory(JSON.parse(localStorage.getItem('resumeTailorHistory') || '[]'));

        document.getElementById('resumeForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const btn = document.getElementById('submitBtn');
            const loading = document.getElementById('loading');
            const results = document.getElementById('results');
            const markdownOutput = document.getElementById('markdownOutput');
            const atsScore = document.getElementById('atsScore');
            const atsTags = document.getElementById('atsTags');
            const missingKeywords = document.getElementById('missingKeywords');
            const originalResumeText = document.getElementById('originalResumeText');

            btn.disabled = true;
            loading.classList.remove('hidden');
            results.classList.add('hidden');

            try {
                const jobUrl = document.getElementById('jobUrl').value.trim();
                const jobDescriptionText = document.getElementById('jobDescriptionText').value.trim();

                if (!jobUrl && !jobDescriptionText) {
                    throw new Error('Please provide a job posting URL or paste the job description.');
                }

                const formData = new FormData();
                formData.append('job_url', jobUrl);
                formData.append('job_description_text', jobDescriptionText);
                formData.append('resume_file', document.getElementById('resumeFile').files[0]);

                const headers = {};
                const token = localStorage.getItem('resumeTailorToken');
                if (token) headers['Authorization'] = `Bearer ${token}`;

                const response = await fetch('/api/optimize', { method: 'POST', body: formData, headers });
                if (!response.ok) {
                    const text = await response.text();
                    let msg = `Server error ${response.status}`;
                    try { msg = JSON.parse(text).detail || msg; } catch {}
                    if (response.status === 429) msg = "You've used your 2 free optimizations for this hour. Please try again later.";
                    throw new Error(msg);
                }
                const data = await response.json();

                atsScore.textContent = `${data.ats_score || 0}%`;
                atsTags.innerHTML = (data.matched_keywords || []).slice(0, 12).map(keyword => `<span class="bg-blue-100 text-blue-800 px-2 py-1 rounded-full text-xs font-medium">${keyword}</span>`).join('');
                missingKeywords.textContent = (data.missing_keywords || []).length ? (data.missing_keywords || []).join(', ') : 'No major keyword gaps detected.';
                originalResumeText.textContent = data.original_resume_text || 'No original resume text available.';

                results.classList.remove('hidden');
                markdownOutput.innerHTML = marked.parse(data.markdown_resume);
                saveToHistory(jobUrl || jobDescriptionText.slice(0, 40) || 'Resume optimization', data.ats_score || 0, data.markdown_resume || '');

                const currentToken = localStorage.getItem('resumeTailorToken');
                if (currentToken) {
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
                }
            } catch (error) {
                alert('Error: ' + error.message);
            } finally {
                btn.disabled = false;
                loading.classList.add('hidden');
            }
        });
    </script>
</body>
</html>"""
    return HTMLResponse(content=html_content)
