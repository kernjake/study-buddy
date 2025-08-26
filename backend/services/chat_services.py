from typing_extensions import List, TypedDict

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model

from langgraph.graph import START, StateGraph

from prompts.prompts import *
from services.vector_store_services import VectorStoreManager


class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


class ChatManager:
    _model = None
    _rag_graph = None

    @classmethod
    def get_llm(cls):
        if cls._model is None:
            cls._model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
        return cls._model

    @classmethod
    def get_rag_graph(cls, vector_store_name):
        if cls._rag_graph is None:
            vector_store = VectorStoreManager.get_vector_store(vector_store_name)
            retrieve = ChatManager.make_retrieve_node(vector_store)
            generate = ChatManager.make_generate_node()
            graph_builder = StateGraph(State).add_sequence([
                retrieve,
                generate
            ])
            graph_builder.add_edge(START, "retrieve")
            cls._rag_graph = graph_builder.compile()

        return cls._rag_graph
    
    @staticmethod
    def make_retrieve_node(vector_store) -> callable:
        def retrieve(state: State):
            retrieved_docs = vector_store.similarity_search(state["question"])
            return{"context": retrieved_docs}
        return retrieve

    @staticmethod
    def make_generate_node() -> callable:
        def generate(state: State):
            docs_content = "\n\n".join(doc.page_content for doc in state["context"])
            prompt_template = ChatPromptTemplate([
                ("system", rag_prompt_system),
                ("user", rag_prompt_user)])
            messages = prompt_template.invoke({"question": state["question"], "context": docs_content})
            llm = ChatManager.get_llm()
            response = llm.invoke(messages)
            return {"answer": response.content}
        return generate
    
    @staticmethod
    def generate_rag_response(vector_store, user_question):
        graph = ChatManager.get_rag_graph(vector_store)
        question = user_question
        response = graph.invoke({"question": question})
        return response["answer"]
        

