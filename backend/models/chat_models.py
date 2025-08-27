from pydantic import BaseModel
from typing import List, Optional
from enum import Enum

class Status(str, Enum):
    failure = "failure"
    success = "success"

class QueryRequest(BaseModel):
    vector_store_name: str
    user_query: str

class QueryResponse(BaseModel):
    status: Status
    message: str