import io
import os

import PyPDF2
import anthropic
import requests
from bs4 import BeautifulSoup
from fastapi import HTTPException, APIRouter, Form, Request, UploadFile, File
from starlette.responses import HTMLResponse

from api.limiter import limiter

route_tailor = APIRouter()

SYSTEM_PROMPT = """You are an expert ATS resume writer. Your ONLY job is to output a fully rewritten, tailored resume in Markdown format.

RULES:
1. Rewrite the user's resume to highlight matching skills and keywords from the job description.
2. CRITICAL: Do NOT hallucinate or invent experience the user does not have.
3. CRITICAL: Do NOT output advice, tips, cover letters, or a summary of the job description.
4. Your entire response MUST be the final Markdown resume, starting directly with the user's name and contact info.
5. If the job description requires a specific technology (e.g., Async Python) and the user has used a directly related framework (e.g., FastAPI, which is inherently async), you MAY explicitly name the required technology to optimize for ATS algorithms. However, do not invent entire roles or completely unrelated skills."""


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


@route_tailor.post("/api/optimize")
@limiter.limit("2/hour")
async def optimize_resume(
    request: Request,
    job_url: str = Form(...),
    resume_file: UploadFile = File(...)
):
    try:
        file_bytes = await resume_file.read()
        try:
            resume_text = parse_file(file_bytes, resume_file.filename.lower())
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        if not resume_text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from the uploaded file.")

        job_description = scrape_url(job_url)
        if not job_description:
            raise HTTPException(status_code=400, detail="Could not fetch job description from the provided URL.")

        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        message = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": f"Job Description:\n{job_description}\n\nMy Current Resume:\n{resume_text}"
            }]
        )

        return {"status": "success", "markdown_resume": message.content[0].text}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
                const data = await response.json();
                if (!response.ok) throw new Error(data.detail || 'Optimization failed.');

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
