import io

import PyPDF2

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = lambda: None

load_dotenv()

from fastapi import FastAPI, UploadFile, File, HTTPException
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from api.limiter import limiter
from api.path.tailor import route_tailor

app = FastAPI(title="Resume Optimizer API", version="1.0")

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)
app.include_router(route_tailor)


@app.post("/api/parser")
async def parse_document(resume_file: UploadFile = File(...)):
    try:
        file_bytes = await resume_file.read()
        filename = resume_file.filename.lower()

        if filename.endswith('.pdf'):
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
            text = "\n".join([page.extract_text() for page in pdf_reader.pages])
        elif filename.endswith('.txt'):
            text = file_bytes.decode('utf-8')
        else:
            raise HTTPException(status_code=400, detail="Only PDF and TXT supported.")

        return {"resume_text": text}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
