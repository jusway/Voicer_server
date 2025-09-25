import os
import uvicorn

from app.main import app


def main():
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload_flag = os.getenv("RELOAD", "false").lower() == "true"
    uvicorn.run(app, host=host, port=port, reload=reload_flag)


if __name__ == "__main__":
    main()
