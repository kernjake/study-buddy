import os

from services.document_loader import DocLoader
from services.ocr_service import OCRService
from services.ner_service import NERService

from langchain_core.documents import Document

class DocumentProcessingServices:

    @staticmethod
    def get_files(directory_path:str, extensions:list = None):
        all_files = []

        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if extensions:
                    if any(file.lower().endswith(ext.lower()) for ext in extensions):
                        all_files.append(os.path.join(root, file))
                else:
                    all_files.append(os.path.join(root, file))

        return all_files

    @staticmethod
    def load_documents(file_names: list):
        docs = []

        for file in file_names:
            doc = DocLoader.layout(file)
            docs.append((doc, file))

        return docs
    
    @staticmethod
    def prepare_document(doc_info: tuple,
                         ner_model: NERService):
        #modify for entities detection into metadata later
        doc, file_name = doc_info
        
        pages = []
        text = doc[0].text if doc and len(doc) > 0 else ""

        if not text.strip():
            ocr_engine = OCRService()
            ocr_pages = ocr_engine.extract_from_pdf(file_name)

            for page in ocr_pages:
                page_content = page["content"]
                entities = []
                if ner_model is not None:
                    entities.extend(NERService.extract_entities())
                document = Document(
                    page_content = page_content,
                    metadata = {
                        "file_name": file_name,
                        "page_no": page["page_no"],
                        "processing_method": "trOCR",
                        "entites": entities
                    }
                )
                pages.append(document)
            return pages

        for page in doc._.pages:
            page_no = page[0].page_no
            page_spans = page[1]
            page_start = page_spans[0].start_char
            page_end = page_spans[-1].end_char
            page_text = text[page_start:page_end]

            entities = []
            if ner_model is not None:
                entities.extend(NERService.extract_entities())
            
            document = Document(
                page_content = page_text,
                metadata = {
                    "file_name": doc_info[1],
                    "page_no": page_no,
                    "processing_method": "spaCyLayout",
                    "entities": entities

                }
            )
            pages.append(document)

        return pages

    @staticmethod
    def process_files(directory_path:str, 
                      extensions:list = None,
                      ner_model_info: tuple = None):
        if ner_model_info is not None:
            model_type, model_name = ner_model_info
            ner_model = NERService(model_type = model_type,
                                   model_name = model_name)

        files = DocumentProcessingServices.get_files(directory_path, extensions)
        loaded_files = DocumentProcessingServices.load_documents(files)

        documents = []
        for file in loaded_files:
            documents.extend(DocumentProcessingServices.prepare_document(file, 
                                                                         ner_model))

        return documents

        