from fastapi import Request as Request1
from fastapi import APIRouter, HTTPException


from services.vector_store_services import VectorStoreManager
from services.document_processing_services import DocumentProcessingServices

from models.vector_store_models import (
    CreateVectorStoreRequest,
    CreateVectorStoreResponse,
    IngestDocsRequest,
    IngestDocsResponse
)


router = APIRouter(prefix = "/vector_store",
                   tags = ["vector_store"])

@router.post(
        "/create_vector_store",
        status_code = 200,
        summary = "Creates a vector store for documents to be stored to."
    )

async def create_vector_store(
    req: CreateVectorStoreRequest,
    ): 
    try: 
        ner_model = req.ner_model
        if ner_model is not None:
            ner_model = {"model_name": ner_model.model_name,
                         "model_type": ner_model.model_type}
    
        if req.embedding_model is not None:
            messsage = VectorStoreManager.create_vector_store(
                vector_store_name= req.name,
                embedding_model = req.embedding_model,
                ner_model = ner_model
             )
            
        else:
            message = VectorStoreManager.create_vector_store(
                vector_store_name = req.name,
                ner_model = ner_model
            )
        
        
        return CreateVectorStoreResponse(
            status = "success",
            message = message
        )
    
    except Exception as e:
        raise HTTPException(
            status_code = 400,
            detail = f"error: {str(e)}"
        )

# @router.get(
#         "/get_vector_stores",
#         status_code = 200,
#         summary = "Fetch all previously created Vector Stores."
# )

# async def get_available_vector_stores(
#     req:
# )
#     print("hi")


@router.post(
    "/ingest_documents",
    status_code = 200,
    summary = "Loads and ingests given documents from a directory to vector DB."
    )

async def ingest_documents(
    req: IngestDocsRequest
    ):
    try:
        message = VectorStoreManager.ingest_documents(
            path = req.file_dir,
            vector_store_name = req.vector_store
        )
        return IngestDocsResponse(
            status = "success",
            message = message
        )
    
    except Exception as e:
        raise HTTPException(
            status_code = 400,
            detail = f"error: {str(e)}"
        )
    
