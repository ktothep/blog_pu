# api/parser.py
from fastapi import FastAPI, UploadFile, File, HTTPException
import io
import pypdf  # Make sure this is in requirements.txt
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()



@app.post("/api/parser")
async def parse_document(resume_file: UploadFile = File(...)):
    try:
        file_bytes = await resume_file.read()
        filename = resume_file.filename.lower()

        if filename.endswith('.pdf'):
            pdf_reader = pypdf.PdfReader(io.BytesIO(file_bytes))
            text = "\n".join([page.extract_text() for page in pdf_reader.pages])
        elif filename.endswith('.txt'):
            text = file_bytes.decode('utf-8')
        else:
            raise HTTPException(status_code=400, detail="Only PDF and TXT supported.")

        return {"resume_text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))