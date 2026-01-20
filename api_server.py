import os
import uuid
import tempfile
import subprocess
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware

from inference import SpeechScoringSystem  

API_KEY = os.environ.get("TOEFL_API_KEY", "")  

app = FastAPI(title="TOEFL Speech Scoring API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

system = SpeechScoringSystem(
    wav2vec_model_path="/root/autodl-tmp/models/checkpoints/best_model.pth",
    whisper_model_path="/root/autodl-tmp/models/whisper-base.en",
    ollama_host="http://localhost:11434",
    ollama_model="qwen3:8b"
)

@app.get("/health")
def health():
    return {"ok": True}

def _ensure_wav(input_path: str) -> str:
    ext = os.path.splitext(input_path)[1].lower()
    if ext == ".wav":
        return input_path
    out_path = os.path.splitext(input_path)[0] + ".wav"
    cmd = ["ffmpeg", "-y", "-i", input_path, "-ac", "1", "-ar", "16000", out_path]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except Exception:
        raise HTTPException(status_code=400, detail="Audio format unsupported or ffmpeg not installed.")
    return out_path

@app.post("/api/evaluate")
async def evaluate(
    audio: UploadFile = File(...),
    transcript: UploadFile | None = File(None),
    task: UploadFile | None = File(None),
    transcript_text: str | None = Form(None),
    task_text: str | None = Form(None),
    authorization: str | None = Header(None),
):
    if API_KEY:
        if not authorization or authorization != f"Bearer {API_KEY}":
            raise HTTPException(status_code=401, detail="Unauthorized")

    with tempfile.TemporaryDirectory() as td:
        audio_path = os.path.join(td, f"{uuid.uuid4()}_{audio.filename}")
        with open(audio_path, "wb") as f:
            f.write(await audio.read())

        audio_path = _ensure_wav(audio_path)

        transcript_path = os.path.join(td, "transcript.txt")
        task_path = os.path.join(td, "task.txt")

        if transcript is not None:
            with open(transcript_path, "wb") as f:
                f.write(await transcript.read())
        elif transcript_text is not None:
            with open(transcript_path, "w", encoding="utf-8") as f:
                f.write(transcript_text)
        else:
            raise HTTPException(status_code=400, detail="Provide transcript file or transcript_text.")

        if task is not None:
            with open(task_path, "wb") as f:
                f.write(await task.read())
        elif task_text is not None:
            with open(task_path, "w", encoding="utf-8") as f:
                f.write(task_text)
        else:
            raise HTTPException(status_code=400, detail="Provide task file or task_text.")

        result = system.evaluate(
            audio_path=audio_path,
            transcript_path=transcript_path,
            task_path=task_path
        )

        return {
            "audio_score": float(result.audio_score),
            "transcribed_text": result.transcribed_text,
            "llm_score": float(result.llm_score),
            "audio_score_detail": getattr(result, "audio_score_detail", None),
            "llm_score_detail": getattr(result, "llm_score_detail", None),
        }
