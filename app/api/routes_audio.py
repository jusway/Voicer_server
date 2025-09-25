from datetime import datetime
from pathlib import Path
import os
import shutil
import uuid

from fastapi import APIRouter, UploadFile, HTTPException

router = APIRouter()

ALLOWED_EXTS = {
    ".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus",
    ".mp4", ".mkv", ".mov", ".avi", ".webm",
}


def _is_media(file: UploadFile) -> bool:
    ct = (file.content_type or "").lower()
    if ct.startswith("audio/") or ct.startswith("video/"):
        return True
    ext = os.path.splitext(file.filename or "")[1].lower()
    return ext in ALLOWED_EXTS


@router.post("/audio/upload")
async def upload_audio(file: UploadFile):
    if not file or not (file.filename):
        raise HTTPException(status_code=400, detail="未收到文件")
    if not _is_media(file):
        raise HTTPException(status_code=400, detail="仅支持音频/视频文件")

    day = datetime.now().strftime("%Y%m%d")
    base_dir = Path("storage/uploads") / day
    base_dir.mkdir(parents=True, exist_ok=True)

    safe_name = f"{uuid.uuid4().hex}_" + os.path.basename(file.filename)
    dest_path = base_dir / safe_name

    # 保存到磁盘（阻塞式写入足够简单可靠）
    with dest_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    return {
        "ok": True,
        "filename": file.filename,
        "saved_path": str(dest_path),
        "url_hint": None,  # 如使用对象存储可返回可访问 URL
    }
