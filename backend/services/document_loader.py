import spacy
from spacy_layout import spaCyLayout
import pandas as pd

class DocLoader:
    """
    Pipeline to handle various document type ingestion and parsing by section. 
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DocLoader, cls).__new__(cls)

            print("Loading DocLoader!")

            #modify later to handle different chunking/parsing
            nlp = spacy.blank("en")
            layout = spaCyLayout(nlp, display_table = cls.render_table) 

            cls.layout = layout
        return cls._instance
    
    def render_table(cls, df:pd.DataFrame):
        return df.to_markdown