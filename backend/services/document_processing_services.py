import os

from services.document_loader import DocLoader

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
    def prepare_document(doc_info: tuple):
        #modify for entities detection into metadata later
        doc = doc_info[0]
        
        pages = []
        text = doc[0].text

        for page in doc._.pages:
            page_no = page[0].page_no
            page_spans = page[1]
            page_start = page_spans[0].start_char
            page_end = page_spans[-1].end_char
            page_text = text[page_start:page_end]

            document = Document(
                page_content = page_text,
                metadata = {
                    "doc_title": doc_info[1],
                    "page_no": page_no
                }
            )
            pages.append(document)

        return pages

    @staticmethod
    def process_files(directory_path:str, extensions:list = None):
        files = DocumentProcessingServices.get_files(directory_path, extensions)
        loaded_files = DocumentProcessingServices.load_documents(files)

        documents = []
        for file in loaded_files:
            documents.extend(DocumentProcessingServices.prepare_document(file))

        return documents


        