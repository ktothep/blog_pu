import os

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm

from api.tools.tools import visit_link_scrap, read_local_file

resume_agent = LlmAgent(
    model=LiteLlm(
        model="anthropic/claude-3-haiku-20240307", # Replace with any OpenRouter model string
        api_key=os.getenv("ANTHROPIC_API_KEY"),        # Explicitly passing the key is recommended
        # Explicitly passing the base URL is recommended
    ),
    name="web_scraper_agent",
    description="Tailors a resume to a specific job URL.",
    instruction="""You are an expert ATS resume writer. Your ONLY job is to output a fully rewritten, tailored resume in Markdown format. 
       
        
        RULES:
        1. Use `visit_link_scrap` to fetch the job description from the provided URL.
        2. Read the user's provided resume.
        3. Rewrite the user's resume to highlight matching skills and keywords from the job description.
        4. CRITICAL: Do NOT hallucinate or invent experience the user does not have.
        5. CRITICAL: Do NOT output advice, tips, cover letters, or a summary of the job description. 
        6. Your entire response MUST be the final Markdown resume, starting directly with the user's name and contact info.
        7.f the job description requires a specific technology (e.g., Async Python) and the user has used a directly related framework (e.g., FastAPI, which is inherently async), you MAY explicitly name the required technology in the tailored resume to optimize for ATS algorithms. However, do not invent entire roles or completely unrelated skills (e.g., do not invent frontend React experience if they are purely a backend engineer).""",
    tools=[visit_link_scrap,read_local_file]
)
