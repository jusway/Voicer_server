from fastapi import APIRouter

router = APIRouter()

@router.get("/providers")
def list_providers():
    return {"providers": []}

@router.post("/providers")
def set_provider():
    return {"ok": True}

