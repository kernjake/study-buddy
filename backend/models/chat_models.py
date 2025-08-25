from pydantic import BaseModel
from typing import List, Optional
import enum


class QueryRequest(BaseModel):
    vector_store: str
    user_query: str

class QueryResponse(BaseModel):
    response: str