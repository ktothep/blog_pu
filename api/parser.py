import io

import PyPDF2
from fastapi import FastAPI, UploadFile, File, HTTPException

app = FastAPI()


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