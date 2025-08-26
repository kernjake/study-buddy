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
        return cls._ner_models[model_name]


    def extract_entities(self, 
                         text: str, 
                         model_name: str):
        #need to refactor and clean up
        model_type = self._model_metadata[model_name]["model_type"]
        model = self._ner_models["model_name"]

        if model_type == "spacy":
            doc = model(text)
            ner_results = [ent.text.lower() for ent in doc.ents]
            return ner_results

        elif model_type == "hf":
            ner_results = [ent["word"].lower() for ent in model(text)]
            return ner_results
        
        else:
                raise ValueError(f"Unsupported NER model_type: {model_type}")
