from pydantic import BaseModel
from typing import List, Optional
import enum

from langchain_core.documents import Document

class Status(str, enum):
    failure = "failure"
    success = "success"

class CreateVectorStoreRequest(BaseModel):
    name: str
    embedding_model:Optional[str] = None

class CreateVectorStoreResponse(BaseModel):
    status: Status
    message: str

class IngestDocsRequest(BaseModel):
    file_dir: str
    vector_store: str

class IngestDocsResponse(BaseModel):
    status: Status
    message: str

