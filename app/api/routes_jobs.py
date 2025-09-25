from fastapi import APIRouter

router = APIRouter()

@router.get("/jobs/{job_id}")
def get_job(job_id: str):
    return {"job_id": job_id, "status": "pending", "progress": 0}

