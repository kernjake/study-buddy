import os
from services.document_loader import DocLoader
from services.ocr_service import OCRService
from services.ner_service import NERService

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

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
        if DocLoader.layout is None:
            DocLoader()

        for file in file_names:
            doc = DocLoader.layout(file)
            docs.append((doc, file))

        return docs
    
    @staticmethod
    def prepare_document(doc_info: tuple,
                         ner_model_name: str):
        #modify for entities detection into metadata later
        doc, file_name = doc_info
        root, extension = os.path.splitext(file_name)
        
        pages = []
        text = doc.text if doc and len(doc) > 0 else ""

        #check for handwritten pdfs for notes
        if not text.strip():
            ocr_engine = OCRService()
            ocr_pages = ocr_engine.extract_from_pdf(file_name)

            for page in ocr_pages:
                page_content = page["content"]

                metadata = {
                    "file_name": file_name,
                    "page_no": page["page_no"],
                    "processing_method": "trOCR"
                }
                entities = []
                if ner_model_name is not None:
                    entites = NERService.extract_entities(text = text,
                                                          model_name = ner_model_name)
                    metadata.update(entities)

                document = Document(
                    page_content = page_content,
                    metadata = metadata
                )
                pages.append(document)
            return pages
        
        if extension == ".pdf":
            for page in doc._.pages:
                page_no = page[0].page_no
                page_spans = page[1]
                page_start = page_spans[0].start_char
                page_end = page_spans[-1].end_char
                page_text = text[page_start:page_end]

                metadata = {
                    "file_name": doc_info[1],
                    "page_no": page_no,
                    "processing_method": "spaCyLayout",
                }
                if ner_model_name is not None:
                    entities = NERService.extract_entities(text = text,
                                                            model_name = ner_model_name)
                    metadata.update(entities)
                
                document = Document(
                    page_content = page_text,
                    metadata = metadata
                )
                pages.append(document)
            return pages
        
            #need to refactor to handle page-wise extractin for docx and pptx
        else:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size = 1000,
                chunk_overlap = 200,
                length_function = len,
                is_separator_regex = False
            )
            text = doc.text
            chunks = text_splitter.split_text(text)

            for chunk in chunks:
                metadata = {
                    "file_name": doc_info[1],
                    "processing_method": "Recursive Text Splitter"
                }
                if ner_model_name is not None:
                    entities = NERService.extract_entities(text = chunk,
                                                           model_name = ner_model_name)
                    metadata.update(entities)

                document = Document(page_content = chunk,
                                    metadata = metadata
                )
                pages.append(document)
            return pages


    @staticmethod
    def process_files(directory_path:str, 
                      extensions:list = None,
                      ner_model_info: dict = None):
        
        ner_model_name = None
        if ner_model_info is not None:
            NERService.get_ner_model(model_type = ner_model_info["model_type"],
                                     model_name = ner_model_info["model_name"])
            ner_model_name = ner_model_info["model_name"]

        extensions = [".pdf", ".docx", ".pptx"]
        files = DocumentProcessingServices.get_files(directory_path, extensions)
        loaded_files = DocumentProcessingServices.load_documents(files)

        documents = []
        print("Docs loaded!")
        for file in loaded_files:
            documents.extend(DocumentProcessingServices.prepare_document(file, 
                                                                         ner_model_name))

        return documents