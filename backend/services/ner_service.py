import spacy
from transformers import pipeline

class NERService:
    _ner_models = {} #cache for models
    _model_metadata = {} #cache for metadata

    @classmethod
    def get_ner_model(cls, 
                      model_type: str, 
                      model_name: str):
        if model_name not in cls._ner_models:
            if model_type == "spacy":
                model = spacy.load(model_name)
                cls._ner_models[model_name] = model
                cls._model_metadata[model_name] ={"model_name": model_name,
                                                  "model_type": model_type}
            elif model_type == "hf":
                model = pipeline("ner", model = model_name, aggregation_strategy = "simple")
                
                cls._ner_models[model_name] = model
                cls._model_metadata[model_name] = {"model_name": model_name,
                                                   "model_type": model_type}
            else:
                raise ValueError(f"Unsupported NER model_type: {model_type}")

    @classmethod
    def extract_entities(cls, 
                         text: str, 
                         model_name: str):
        #need to refactor and clean up
        model_type = cls._model_metadata[model_name]["model_type"]
        model = cls._ner_models[model_name]

        entities = {}
        if model_type == "spacy":
            doc = model(text)
            for ent in doc.ents:
                ent_label = ent.label_
                ent_text = ent.text.strip().lower()
                if ent_label not in entities:
                    entities[ent_label] = []
                entities[ent_label].append(ent_text)

        elif model_type == "hf":
            results = model(text)
            for ent in results:
                ent_label = ent["entity_group"]
                ent_text = ent["word"].strip().lower()
                if ent_label not in entities:
                    entities[ent_label] = []
                entities[ent_label].append(ent_text)
        
        else:
                raise ValueError(f"Unsupported NER model_type: {model_type}")
        
        return entities