from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from pdf2image import convert_from_path
from PIL import Image

class OCRService:

    _instance = None

    def __new__(cls, model_name="microsoft/trocr-base-handwritten"):
        if cls._instance is None:
            cls._instance = super(OCRService, cls).__new__(cls)
            print("Loading TrOCR")

            cls.processor = TrOCRProcessor.from_pretrained(model_name)
            cls.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        return cls._instance

    def extract_text_from_image(self, image: Image.Image) -> str:
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
        generated_ids = self.model.generate(pixel_values)
        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return text.strip()

    def extract_from_pdf(self, pdf_path: str):
        """Convert PDF to images and run OCR page by page."""
        pages = convert_from_path(pdf_path)
        results = []
        for i, page in enumerate(pages, start=1):
            text = self.extract_text_from_image(page)
            results.append({"page_no": i, "content": text})
        return results
