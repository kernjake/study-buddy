import faiss
import getpass
import os
import json
from uuid import uuid4

from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from services.document_processing_services import DocumentProcessingServices

class VectorStoreManager:
    _stores = {} #cache of vector stores
    _metadata = {} #chace for vector store metadata

    @staticmethod
    def get_embeddings_model(embedding_model_name: str = "models/gemini-embedding-001"):
        embeddings = GoogleGenerativeAIEmbeddings(model = embedding_model_name)
        return embeddings
    
    @staticmethod
    def save_metadata(path: str, 
                      metadata: dict):
        with open(path, "w", encoding = "utf-8") as f:
            json.dump(metadata, f)

    @staticmethod
    def load_metadata(vector_store_name: str):
        with open(f"vector_stores/{vector_store_name}/metadata.json", "r", encoding = "utf-8") as f:
            return json.load(f)
        
    @staticmethod
    def create_vector_store(vector_store_name: str, 
                            embedding_model: str = "models/gemini-embedding-001",
                            ner_model: dict = None
                            ):
        if embedding_model == "models/gemini-embedding-001":
            embeddings = VectorStoreManager.get_embeddings_model()

        else:
            print("Oh no! No huggingface support yet.")
            #huggingface model support
        
        index = faiss.IndexFlatL2(len(embeddings.embed_query("Getting embeddings!")))

        vector_store = FAISS(
            embedding_function = embeddings,
            index = index,
            docstore = InMemoryDocstore(),
            index_to_docstore_id={}
        )

        vector_store_path = f"vector_stores/{vector_store_name}"
        os.makedirs(os.path.dirname(vector_store_path), exist_ok = True)
        vector_store.save_local(vector_store_path)
        
        metadata = {
            "name": vector_store_name,
            "embedding_model": embedding_model,
            "ner_model": ner_model,
            "doc_names": []
        }

        VectorStoreManager.save_metadata(vector_store_path + "/metadata.json", 
                                         metadata)

        return "Vector store created succesfully!"
    
    @classmethod
    def get_vector_store(cls, vector_store_name):
        if vector_store_name not in cls._stores:
            vector_store_metadata = VectorStoreManager.load_metadata(vector_store_name)
            embedding_model = vector_store_metadata["embedding_model"]
            embeddings = VectorStoreManager.get_embeddings_model(embedding_model)
            vector_store = FAISS.load_local(
                folder_path = f"vector_stores/{vector_store_name}",
                embeddings = embeddings,
                allow_dangerous_deserialization = True
            )
            cls._stores[vector_store_name] = vector_store
            cls._metadata[vector_store_name] = vector_store_metadata
        return cls._stores[vector_store_name]


    #revet to static with seperate class method for fetching metadata
    @classmethod
    def ingest_documents(cls,
                         path: str, 
                         vector_store_name: str
                         ):
        
        vector_store = VectorStoreManager.get_vector_store(vector_store_name)
        ner_model_info = cls._metadata[vector_store_name]["ner_model"]

        documents = DocumentProcessingServices.process_files(directory_path = path,
                                                             ner_model_info = ner_model_info)

        uuids = [str(uuid4()) for _ in range(len(documents))]

        vector_store.add_documents(
            documents = documents,
            ids = uuids)
        
        vector_store.save_local(f"vector_stores/{vector_store_name}")

        return "Ingested documents and saved store succesfully!"