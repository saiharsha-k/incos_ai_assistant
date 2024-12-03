import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from typing import List, Dict

class QAModel:
    def __init__(self, model_name: str = "t5-small"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)

    def train(self, train_data: List[Dict[str, str]], epochs: int = 100, batch_size: int = 16):
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5)

        for epoch in range(epochs):
            total_loss = 0
            for i in range(0, len(train_data), batch_size):
                batch = train_data[i:i+batch_size]
                
                input_texts = [f"question: {item['question']} context: {item['context']}" for item in batch]
                target_texts = [item['answer'] for item in batch]

                inputs = self.tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
                targets = self.tokenizer(target_texts, return_tensors="pt", padding=True, truncation=True, max_length=64).to(self.device)

                loss = self.model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, labels=targets.input_ids).loss
                total_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_data)}")

    def generate_answer(self, question: str, context: str) -> str:
        self.model.eval()
        input_text = f"question: {question} context: {context}"
        inputs = self.tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, max_length=64)

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def save_model(self, path: str):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    @classmethod
    def load_model(cls, path: str):
        model = cls(model_name=path)
        return model