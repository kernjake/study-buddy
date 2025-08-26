import spacy
from transformers import pipeline

class NERService:
    _instance = None

    def __new__(cls, model_type, model_name):
        if cls._instance is None:
            cls._instance = super(NERService, cls).__new__(cls)
            cls._instance.model_type = model_type

            if model_type == "spacy":
                cls._instance.model = spacy.load(model_name)
            elif model_type == "hf":
                cls._instance.model = pipeline(
                    "ner", model=model_name, aggregation_strategy="simple"
                )
            else:
                raise ValueError(f"Unsupported NER model_type: {model_type}")

        return cls._instance

    def extract_entities(self, text: str):
        if self.model_type == "spacy":
            doc = self.model(text)
            ner_results = [ent.text.lower() for ent in doc.ents]
            return ner_results

        elif self.model_type == "hf":
            ner_results = [ent["word"].lower() for ent in self.model(text)]
            return ner_results
