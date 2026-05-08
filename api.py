import os, uuid, shutil
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import main as interview
from config import Config

app = FastAPI(title="AI Interview System")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root(): return FileResponse("static/index.html")

@app.post("/start")
async def start(
    file: UploadFile = File(...),
    interview_mode: str = Form("text"),
    job_description: str = Form(""),
):
    os.makedirs(Config.UPLOAD_DIR, exist_ok=True)
    path = f"{Config.UPLOAD_DIR}/{uuid.uuid4()}_{file.filename}"
    with open(path, "wb") as f: shutil.copyfileobj(file.file, f)
    result = interview.start_session(path, interview_mode=interview_mode, job_description=job_description)
    if result["status"] == "error": raise HTTPException(400, result["message"])
    return result

@app.post("/answer/text")
async def answer_text(user_id: str = Form(...), answer: str = Form(...)):
    result = interview.submit_answer(user_id, answer=answer)
    if result["status"] == "error": raise HTTPException(400, result["message"])
    return result

@app.post("/answer/audio")
async def answer_audio(user_id: str = Form(...), file: UploadFile = File(...)):
    os.makedirs(Config.UPLOAD_DIR, exist_ok=True)
    ext  = os.path.splitext(file.filename)[-1] or ".webm"
    path = f"{Config.UPLOAD_DIR}/audio_{uuid.uuid4()}{ext}"
    with open(path, "wb") as f: shutil.copyfileobj(file.file, f)
    try:
        result = interview.submit_answer(user_id, media_file_path=path)
    finally:
        if os.path.exists(path): os.remove(path)
    if result["status"] == "error": raise HTTPException(400, result["message"])
    return result

@app.post("/answer/video")
async def answer_video(user_id: str = Form(...), file: UploadFile = File(...)):
    os.makedirs(Config.UPLOAD_DIR, exist_ok=True)
    ext  = os.path.splitext(file.filename)[-1] or ".webm"
    path = f"{Config.UPLOAD_DIR}/video_{uuid.uuid4()}{ext}"
    with open(path, "wb") as f: shutil.copyfileobj(file.file, f)
    try:
        result = interview.submit_answer(user_id, media_file_path=path)
    finally:
        if os.path.exists(path): os.remove(path)
    if result["status"] == "error": raise HTTPException(400, result["message"])
    return result
