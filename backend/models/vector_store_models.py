from pydantic import BaseModel
from typing import List, Optional
from enum import Enum

class Status(str, Enum):
    failure = "failure"
    success = "success"

class NERModel(BaseModel):
    model_name: str
    model_type: str

class CreateVectorStoreRequest(BaseModel):
    name: str
    embedding_model:Optional[str] = None
    ner_model: Optional[NERModel] = None

class CreateVectorStoreResponse(BaseModel):
    status: Status
    message: str

class IngestDocsRequest(BaseModel):
    file_dir: str
    vector_store: str

class IngestDocsResponse(BaseModel):
    status: Status
    message: str