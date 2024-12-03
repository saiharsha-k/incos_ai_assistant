import json
from sentence_transformers import SentenceTransformer
import numpy as np

class CorpusEmbedder:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def load_chunks(self, json_file_path):
        with open(json_file_path, 'r', encoding='utf-8') as file:
            return json.load(file)

    def embed_chunks(self, chunks):
        embeddings = self.model.encode(chunks, show_progress_bar=True)
        return embeddings

    def process(self, json_file_path, output_file_path):
        print("Loading chunks...")
        data = self.load_chunks(json_file_path)
        
        chunks = [item['text'] for item in data]
        metadata = [{'source': item['source']} for item in data]

        print("Generating embeddings...")
        embeddings = self.embed_chunks(chunks)

        print("Saving embeddings and metadata...")
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump({
                'embeddings': embeddings.tolist(),
                'metadata': metadata,
                'chunks': chunks
            }, f)

        print(f"Embeddings and metadata saved to {output_file_path}")

# Usage
# json_file_path = ''
# output_file_path = ''

# Create an instance of CorpusEmbedder and process the data
#embedder = CorpusEmbedder()
#embedder.process(json_file_path, output_file_path)

