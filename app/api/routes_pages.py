from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

@router.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@router.get("/upload", response_class=HTMLResponse)
def upload_page(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})

@router.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@router.get("/jobs/{job_id}", response_class=HTMLResponse)
def job_detail_page(job_id: str, request: Request):
    return templates.TemplateResponse("job_detail.html", {"request": request, "job_id": job_id})

@router.get("/transcripts/{transcript_id}", response_class=HTMLResponse)
def transcript_detail_page(transcript_id: str, request: Request):
    return templates.TemplateResponse("transcript_detail.html", {"request": request, "transcript_id": transcript_id})

@router.get("/polish", response_class=HTMLResponse)
def polish_page(request: Request):
    return templates.TemplateResponse("polish.html", {"request": request})

