from fastapi import APIRouter

router = APIRouter()

@router.post("/polish")
def create_polish():
    return {"job_id": "polish:placeholder"}

@router.get("/polishes/{polish_id}")
def get_polish(polish_id: str):
    return {"polish_id": polish_id, "status": "pending"}

