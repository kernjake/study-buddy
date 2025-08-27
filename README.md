# study-buddy
*Note: In active development*
Goal: Chatbot capable of interacting with various document types common to classes including
- Microsoft Office documents (spaCyLayout)
- Printed PDFs (spaCyLayout)
- Scanned PDFs (trOCR)
- Handwritten notes (trOCR)

On top of this, the project aims to implement a hybrid rag approach to document search and retrieval.
Upon creation of a vector DB, ner models (spacy or huggingface) can be provided to enable metadata tagging of documents.
Search will utilize entities tagged in a user query based on the chosen model to filter document pieces to increase accuracy

Current limitations:
- Basic backend created
    - Vector store creation
    - Document ingestion
        - spaCyLayout
        - OCR
        - NER Tagging
        - Various embedding model support
    - Rag based answer generation
- Still developing and testing hybrid rag functionality
- No frontend

To use:
- Install requirements
- Initiate backend using uvicorn
