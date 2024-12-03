from transformers import ElectraForSequenceClassification, ElectraTokenizer
import torch
import os

class Generator:
    def __init__(self, model_path: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if not os.path.exists(model_path):
            raise ValueError(f"Model path does not exist: {model_path}")
        
        self.model = ElectraForSequenceClassification.from_pretrained(model_path).to(self.device)
        self.tokenizer = ElectraTokenizer.from_pretrained(model_path)

    def generate(self, question: str, context: str) -> str:
        # Combine question and context
        input_text = f"Question: {question} Context: {context}"
        
        inputs = self.tokenizer.encode_plus(
            input_text,
            add_special_tokens=True,
            return_tensors='pt',
            max_length=512,
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Get the predicted class (assuming binary classification)
        predicted_class = torch.argmax(outputs.logits, dim=1).item()
        
        # You might want to adjust this based on your specific classification task
        return "Yes" if predicted_class == 1 else "No"