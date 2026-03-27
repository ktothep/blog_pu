import io
import os

import PyPDF2
import anthropic
import requests
from bs4 import BeautifulSoup
from fastapi import HTTPException, APIRouter, Form, Request, UploadFile, File
from starlette.responses import HTMLResponse

from api.limiter import limiter
from api.database import log_interaction

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
    job_url: str = Form(...),
    resume_file: UploadFile = File(...)
):
    resume_text = None
    try:
        file_bytes = await resume_file.read()
        try:
            resume_text = parse_file(file_bytes, resume_file.filename.lower())
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        if not resume_text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from the uploaded file.")

        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        messages = [
            {
                "role": "user",
                "content": f"Please tailor my resume for this job posting: {job_url}\n\nHere is my current resume:\n\n{resume_text}"
            }
        ]

        # Agentic loop — Claude calls scrape_job_url tool, we execute it, then Claude writes the resume
        while True:
            response = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages
            )

            if response.stop_reason == "tool_use":
                tool_use_block = next(b for b in response.content if b.type == "tool_use")
                url_to_scrape = tool_use_block.input["url"]
                job_description = scrape_url(url_to_scrape)

                if not job_description:
                    job_description = "Could not retrieve the job description from this URL. Please tailor the resume based on the URL context and any visible keywords."

                messages.append({"role": "assistant", "content": response.content})
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use_block.id,
                            "content": job_description
                        }
                    ]
                })

            elif response.stop_reason == "end_turn":
                final_text = next(b.text for b in response.content if hasattr(b, "text"))
                log_interaction(job_url, resume_file.filename, resume_text, "success")
                return {"status": "success", "markdown_resume": final_text}

            else:
                raise HTTPException(status_code=500, detail=f"Unexpected stop reason: {response.stop_reason}")

    except HTTPException as e:
        log_interaction(job_url, resume_file.filename, resume_text, "error", e.detail)
        raise
    except Exception as e:
        if "529" in str(e) or "overloaded" in str(e).lower():
            msg = "The AI service is temporarily overloaded. Please try again in a moment."
            log_interaction(job_url, resume_file.filename, resume_text, "error", msg)
            raise HTTPException(status_code=503, detail=msg)
        log_interaction(job_url, resume_file.filename, resume_text, "error", f"{type(e).__name__}: {e}")
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
</head>
<body class="bg-gray-50 min-h-screen p-8 font-sans">
    <div class="max-w-3xl mx-auto bg-white p-8 rounded-xl shadow-lg border border-gray-100">
        <h1 class="text-3xl font-bold text-gray-800 mb-2">AI Resume Tailor</h1>
        <p class="text-gray-500 mb-8">Upload your resume and a job URL to get a perfectly optimized match.</p>

        <form id="resumeForm" class="space-y-6">
            <div>
                <label class="block text-sm font-medium text-gray-700 mb-2">Job Posting URL</label>
                <input type="url" id="jobUrl" required placeholder="https://linkedin.com/jobs/..."
                       class="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 outline-none">
            </div>
            <div>
                <label class="block text-sm font-medium text-gray-700 mb-2">Your Current Resume (PDF/TXT)</label>
                <input type="file" id="resumeFile" required accept=".pdf,.txt"
                       class="w-full px-4 py-2 border border-gray-300 rounded-lg file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700">
            </div>
            <button type="submit" id="submitBtn"
                    class="w-full bg-blue-600 text-white font-bold py-3 px-4 rounded-lg hover:bg-blue-700 transition-colors">
                Optimize Resume
            </button>
        </form>

        <div id="loading" class="hidden mt-8 text-center text-blue-600 font-medium animate-pulse">
            Processing document and generating tailored resume... (20-30s)
        </div>

        <div id="results" class="hidden mt-10 pt-8 border-t border-gray-200">
            <div id="markdownOutput" class="prose max-w-none bg-gray-50 p-6 rounded-lg border border-gray-200"></div>
        </div>
    </div>

    <script>
        document.getElementById('resumeForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const btn = document.getElementById('submitBtn');
            const loading = document.getElementById('loading');
            const results = document.getElementById('results');
            const markdownOutput = document.getElementById('markdownOutput');

            btn.disabled = true;
            loading.classList.remove('hidden');
            results.classList.add('hidden');

            try {
                const formData = new FormData();
                formData.append('job_url', document.getElementById('jobUrl').value);
                formData.append('resume_file', document.getElementById('resumeFile').files[0]);

                const response = await fetch('/api/optimize', { method: 'POST', body: formData });
                if (!response.ok) {
                    const text = await response.text();
                    let msg = `Server error ${response.status}`;
                    try { msg = JSON.parse(text).detail || msg; } catch {}
                    if (response.status === 429) msg = "You've used your 2 free optimizations for this hour. Please try again later.";
                    throw new Error(msg);
                }
                const data = await response.json();

                results.classList.remove('hidden');
                markdownOutput.innerHTML = marked.parse(data.markdown_resume);
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
