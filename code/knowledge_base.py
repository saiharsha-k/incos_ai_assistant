<<<<<<< HEAD
import json
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

class DataProcessor:
    def __init__(self, json_directory, output_file):
        self.json_directory = json_directory
        self.output_file = output_file
        self.stop_words = set(stopwords.words('english'))

    def load_json_files(self):
        """Load all JSON files from the specified directory."""
        data = {}
        for filename in os.listdir(self.json_directory):
            if filename.endswith('.json'):
                file_path = os.path.join(self.json_directory, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    data[filename] = json.load(file)
        return data

    def extract_text(self, data):
        """Recursively extract text from nested JSON structures."""
        text = []
        if isinstance(data, dict):
            for value in data.values():
                text.extend(self.extract_text(value))
        elif isinstance(data, list):
            for item in data:
                text.extend(self.extract_text(item))
        elif isinstance(data, str):
            text.append(data)
        return text

    def clean_text(self, text):
        """Clean the text by removing special characters and extra spaces."""
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text.lower()

    def preprocess_text(self, text):
        """Preprocess the text: clean, tokenize, remove stopwords."""
        cleaned_text = self.clean_text(text)
        tokens = word_tokenize(cleaned_text)
        tokens = [token for token in tokens if token not in self.stop_words]
        return ' '.join(tokens)

    def chunk_text(self, text, chunk_size=100, overlap=20):
        """Split text into overlapping chunks."""
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
        return chunks

    def process_and_save(self):
        """Process all data and save to a JSON file."""
        json_data = self.load_json_files()
        all_chunks = []

        for filename, data in json_data.items():
            raw_text = ' '.join(self.extract_text(data))
            preprocessed_text = self.preprocess_text(raw_text)
            chunks = self.chunk_text(preprocessed_text)
            all_chunks.extend([{"text": chunk, "source": filename} for chunk in chunks])

        # Save to JSON file
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(all_chunks, f, ensure_ascii=False, indent=2)

        print(f"Processed {len(all_chunks)} chunks and saved to {self.output_file}")
        return all_chunks

# Usage
if __name__ == "__main__":
    json_directory = "E:\Sai Harsha\From_23\MSc Data Science\DeepLearning_Applications\coursework_assistant\coursework_data"
    output_file = "processed_chunks.json"
    processor = DataProcessor(json_directory, output_file)
    processed_chunks = processor.process_and_save()

    print(f"Total chunks processed: {len(processed_chunks)}")
    print("Sample chunks:")
    for chunk, source in processed_chunks[:5]:
        print(f"Source: {source}")
        print(f"Chunk: {chunk[:100]}...")  # Print first 100 characters of each chunk
        print()
=======
class KnowledgeBase:
    def __init__(self) -> None:
        pass


>>>>>>> 2e89973b853e0a6caa099bd696848c32a04cd763
