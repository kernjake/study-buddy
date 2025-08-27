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
    summary = "Returns response based on documents in the vector store"
)

async def rag_response(
    req: QueryRequest
    ):
    try:
        message = ChatManager.generate_rag_response(
            vector_store_name = req.vector_store_name,
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