from dotenv import load_dotenv
import os

from fastapi import FastAPI

load_dotenv()
os.environ.get("GOOGLE_API_KEY")


from contextlib import asynccontextmanager

PREFIX_V1 = "/study_buddy"


app = FastAPI(
    title = "Study Buddy",
    docs_url = f"{PREFIX_V1}/docs",
    redoc_url = None,
    swagger_ui_parameters={"defaultModelsExpandDepth": -1}
)

