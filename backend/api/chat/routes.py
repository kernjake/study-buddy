from fastapi import Request as Request1
from fastapi import APIRouter, HTTPException


from backend.models.chat_models import (
    QueryRequest,
    QueryResponse
)
router = APIRouter()

from backend.services.vector_store_services import VectorStoreManager
from backend.services.chat_services import ChatManager


@router.post(
    "/query_vector_store"
    status_code = 200,
    summary = "Querys documents based on the vectore store"
)

async def query_vector_store(
    req: QueryRequest
    ):
