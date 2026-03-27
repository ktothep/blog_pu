import uuid

from fastapi import HTTPException, APIRouter, Form, UploadFile, File
from pydantic import BaseModel
from google.genai import types
from starlette.responses import HTMLResponse

from api.index import limiter
from api.runner import runner, session_service
from api.tools.tools import extract_text_from_file

# 1. Initialize FastAPI


route_tailor=APIRouter()
# 2. Define the exact data structure we expect from the frontend
class OptimizationRequest(BaseModel):
    job_url: str
    resume_text: str


# 5. Create the API Endpoint
@route_tailor.post("/path/v1/optimize-resume")
@limiter.limit("2/hour")
async def optimize_resume(job_url: str = Form(...),resume_file: UploadFile = File(...)):
    try:
        # 1. Read the uploaded file into memory
        file_bytes = await resume_file.read()

        # 2. Extract the text based on file type
        try:
            resume_text = extract_text_from_file(file_bytes, resume_file.filename)
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))

        if not resume_text.strip():
            raise HTTPException(status_code=400, detail="Could not extract any text from the document.")

        # 3. Construct the prompt
        prompt = f"""
            Please tailor my resume for this job: {job_url}

            Here is my current resume:
            {resume_text}
            """

        # 4. Execute the ADK agent
        content = types.Content(
            role='user',
            parts=[types.Part.from_text(text=prompt)]
        )
        user_id = "api_user"
        # 2. Generate a unique session ID per request so users don't overwrite each other
        session_id = str(uuid.uuid4())
        session_service.create_session(user_id=user_id, session_id=session_id,app_name=runner.app_name)
        final_markdown = ""

        async for event in runner.run_async(
                user_id="api_user",
                session_id=session_id,
                new_message=content
        ):
            # The runner emits various events (tool calls, reasoning, etc.)
            # We only care about the final text output for this specific API response.
            if event.is_final_response():
                if event.content and event.content.parts:
                    final_markdown = event.content.parts[0].text
                elif event.actions and event.actions.escalate:
                    raise Exception(f"Agent escalated: {event.error_message}")
                break

        return {
            "status": "success",
            "filename_processed": resume_file.filename,
            "markdown_resume": final_markdown
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@route_tailor.get("/")
async def serve_ui():
    html_content = """
    <!DOCTYPE html>
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
            <p class="text-gray-500 mb-8">Upload your base resume and a job URL to get a perfectly optimized ATS match.</p>

            <form id="resumeForm" class="space-y-6">
                <div>
                    <label class="block text-sm font-medium text-gray-700 mb-2">Job Posting URL</label>
                    <input type="url" id="jobUrl" required placeholder="https://linkedin.com/jobs/..." 
                           class="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition-all">
                </div>

                <div>
                    <label class="block text-sm font-medium text-gray-700 mb-2">Your Current Resume (PDF/TXT/DOCX)</label>
                    <input type="file" id="resumeFile" required accept=".pdf,.txt,.docx"
                           class="w-full px-4 py-2 border border-gray-300 rounded-lg file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100 transition-all">
                </div>

                <button type="submit" id="submitBtn" 
                        class="w-full bg-blue-600 text-white font-bold py-3 px-4 rounded-lg hover:bg-blue-700 transition-colors flex justify-center items-center">
                    <span>Optimize Resume</span>
                </button>
            </form>

            <div id="loading" class="hidden mt-8 text-center text-blue-600 font-medium animate-pulse">
                Analyzing requirements and tailoring your resume... This takes about 15-20 seconds.
            </div>

            <div id="results" class="hidden mt-10 pt-8 border-t border-gray-200">
                <div id="feedbackBox" class="p-4 rounded-lg mb-6"></div>
                <div id="markdownOutput" class="prose max-w-none bg-gray-50 p-6 rounded-lg border border-gray-200"></div>
            </div>
        </div>

        <script>
            document.getElementById('resumeForm').addEventListener('submit', async (e) => {
                e.preventDefault();

                const btn = document.getElementById('submitBtn');
                const loading = document.getElementById('loading');
                const results = document.getElementById('results');
                const feedbackBox = document.getElementById('feedbackBox');
                const markdownOutput = document.getElementById('markdownOutput');

                // UI State: Loading
                btn.disabled = true;
                btn.classList.add('opacity-50');
                loading.classList.remove('hidden');
                results.classList.add('hidden');

                // Prepare the data
                const formData = new FormData();
                formData.append('job_url', document.getElementById('jobUrl').value);
                formData.append('resume_file', document.getElementById('resumeFile').files[0]);

                try {
                    // Send it to your FastAPI backend
                    const response = await fetch('/api/v1/optimize-resume', {
                        method: 'POST',
                        body: formData
                    });

                    const data = await response.json();

                    if (!response.ok) throw new Error(data.detail || 'Something went wrong');

                    // UI State: Success / Render Results
                    results.classList.remove('hidden');

                    if (data.is_match) {
                        feedbackBox.className = "p-4 rounded-lg mb-6 bg-green-50 text-green-800 border border-green-200";
                        feedbackBox.innerHTML = `<strong>Great Fit!</strong> ${data.feedback}`;
                        markdownOutput.innerHTML = marked.parse(data.markdown_resume);
                    } else {
                        feedbackBox.className = "p-4 rounded-lg mb-6 bg-yellow-50 text-yellow-800 border border-yellow-200";
                        feedbackBox.innerHTML = `<strong>Missing Requirements:</strong> ${data.feedback}`;
                        markdownOutput.innerHTML = "<p class='text-gray-500 italic'>No resume generated due to skill mismatch.</p>";
                    }

                } catch (error) {
                    alert('Error: ' + error.message);
                } finally {
                    // UI State: Reset
                    btn.disabled = false;
                    btn.classList.remove('opacity-50');
                    loading.classList.add('hidden');
                }
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)