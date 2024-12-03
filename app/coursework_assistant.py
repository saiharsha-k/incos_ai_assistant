# rag_system.py

import faiss
import numpy as np
import json
from transformers import T5ForConditionalGeneration, T5Tokenizer
from sentence_transformers import SentenceTransformer

class RAGSystem:
    def __init__(self, model_path, faiss_index_path, embeddings_path, corpus_path):
        self.model, self.tokenizer = self.load_model(model_path)
        self.index = faiss.read_index(faiss_index_path)
        with open(embeddings_path, 'r') as f:
            self.embeddings = json.load(f)
        with open(corpus_path, 'r') as f:
            self.corpus = json.load(f)
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')

    def load_model(self, model_path):
        model = T5ForConditionalGeneration.from_pretrained(model_path)
        tokenizer = T5Tokenizer.from_pretrained(model_path)
        return model, tokenizer

    def retrieve(self, query, k=3):
        query_vector = self.sentence_transformer.encode([query])[0]
        _, indices = self.index.search(np.array([query_vector], dtype=np.float32), k)
        contexts = [self.corpus[str(i)] for i in indices[0]]
        return contexts

    def generate_answer(self, question, contexts, max_length=100):
        combined_context = " ".join(contexts)
        input_text = f"question: {question} context: {combined_context}"
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
        
        outputs = self.model.generate(input_ids, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2)
        
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer

    def answer_question(self, question):
        contexts = self.retrieve(question)
        answer = self.generate_answer(question, contexts)
        return answer