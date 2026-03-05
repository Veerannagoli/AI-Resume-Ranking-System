from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from engine import ResumeRankingEngine
import uvicorn, io
import os


app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
engine = ResumeRankingEngine(groq_api_key=GROQ_API_KEY)

@app.post("/rank-resumes")
async def api_rank_resumes(jd_text: str = Form(...), candidate_level: str = Form(...), files: list[UploadFile] = File(...)):
    streams = []
    for file in files:
        content = await file.read()
        s = io.BytesIO(content); s.filename = file.filename; streams.append(s)
    return {"results": engine.rank_resumes(streams, jd_text, candidate_level)}

@app.post("/generate-report")
async def api_generate_report(jd_text: str = Form(...), filename: str = Form(...), files: list[UploadFile] = File(...)):
    for file in files:
        if file.filename == filename:
            content = await file.read()
            s = io.BytesIO(content); s.filename = file.filename
            return engine.generate_deep_report(s, jd_text)
    raise HTTPException(status_code=404, detail="File not found")
