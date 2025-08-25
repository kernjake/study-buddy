from typing_extensions import List, TypedDict

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate


from backend.prompts.prompts import *
from backend.services.vector_store_services import VectorStoreManager

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


class ChatManager:
    @staticmethod
    def retrieve(state: State, vector_store):
        retrieved_docs = vector_store.similarity_search(state["question"])
        return{"context": retrieved_docs}

    @staticmethod
    def generate(state: State):
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])

        prompt_template = ChatPromptTemplate([
             ("system", rag_prompt_system),
             ("user", rag_prompt_user)])
        messages = prompt.invoke({"question": state["question"], "context": docs_content})
