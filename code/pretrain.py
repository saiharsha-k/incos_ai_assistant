import json
import torch
from model import QAModel
from transformers import T5Tokenizer
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class LongTextDataset(Dataset):
    def __init__(self, json_file, tokenizer, max_length=512):
        with open(json_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Check if data is a list or a dictionary
        if isinstance(self.data, dict):
            self.keys = list(self.data.keys())
        elif isinstance(self.data, list):
            self.keys = range(len(self.data))
        else:
            raise ValueError("Unsupported JSON structure")

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        item = self.data[key] if isinstance(self.data, dict) else self.data[idx]
        
        # Try to find text content in the item
        text = None
        if isinstance(item, str):
            text = item
        elif isinstance(item, dict):
            # Look for common keys that might contain text
            for possible_key in ['text', 'content', 'body', 'chunk']:
                if possible_key in item:
                    text = item[possible_key]
                    break
        
        if text is None:
            raise ValueError(f"Could not find text content in item at index {idx}")

        inputs = self.tokenizer.encode_plus(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze()
        }

def pre_train_model(model, dataset, epochs=100, batch_size=16, learning_rate=5e-5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")

    return model

'''if __name__ == "__main__":
    # Initialize the model and tokenizer
    model = QAModel("t5-small")
    tokenizer = T5Tokenizer.from_pretrained("t5-small")

    # Load and prepare the dataset
    #json_file_path = "E:\Sai Harsha\From_23\MSc Data Science\DeepLearning_Applications\coursework_assistant\code\processed_data.json"
    dataset = LongTextDataset(json_file_path, tokenizer)

    # Pre-train the model
    pre_trained_model = pre_train_model(model.model, dataset, epochs=10, batch_size=8)

    # Save the pre-trained model
    #output_dir = "E:\Sai Harsha\From_23\MSc Data Science\DeepLearning_Applications\coursework_assistant\code\models"
    pre_trained_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Pre-trained model saved to {output_dir}")'''