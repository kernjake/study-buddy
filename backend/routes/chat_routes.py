from fastapi import Request as Request1
from fastapi import APIRouter, HTTPException


from models.chat_models import (
    QueryRequest,
    QueryResponse
)
router = APIRouter(prefix = "/chat",
                   tags = ["chat"])

from services.chat_services import ChatManager


@router.post(
    "/rag_response",
    status_code = 200,
    summary = "Querys documents based on the vectore store"
)

async def rag_response(
    req: QueryRequest
    ):
    try:
        message = ChatManager.generate_rag_response(
            vector_store = req.vector_store,
            user_question = req.user_query)
        
        return QueryResponse(
            status = "success",
            message = message
        )
        
    except Exception as e:
        raise HTTPException(
            status_code = 400,
            detail = f"error: {str(e)}"
        )