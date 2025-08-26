from dotenv import load_dotenv
import os

from fastapi import FastAPI

from routes import chat_routes, vector_store_routes



load_dotenv()
os.environ.get("GOOGLE_API_KEY")


from contextlib import asynccontextmanager



app = FastAPI(
    title = "Study Buddy",
    docs_url = f"/docs",
    redoc_url = None,
    swagger_ui_parameters={"defaultModelsExpandDepth": -1}
)

app.include_router(chat_routes.router)
app.include_router(vector_store_routes.router)
