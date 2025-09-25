from fastapi import APIRouter, UploadFile, BackgroundTasks

router = APIRouter()

@router.post("/audio/upload")
async def upload_audio(file: UploadFile, bg: BackgroundTasks):
    # TODO: save file and enqueue ASR job
    return {"job_id": "asr:placeholder"}

