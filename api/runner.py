import os

from google.adk import Runner
from google.adk.sessions import InMemorySessionService, DatabaseSessionService

from agent.agent import resume_agent

session_service = DatabaseSessionService(db_url=os.getenv("DB_URL"))

# 3. Initialize the Runner
runner = Runner(
    agent=resume_agent,
    app_name="resume_optimizer_api",
    session_service=session_service
)