from typing import List, Dict
import json

def load_embeddings(file_path: str) -> Dict[str, List[float]]:
    """Load embeddings from a JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def main():
    # Load your embeddings
    embeddings = load_embeddings('E:\Sai Harsha\From_23\MSc Data Science\DeepLearning_Applications\coursework_assistant\code\corpus_embeddings.json')

    # Print detailed debug information
    print(f"Number of embeddings: {len(embeddings)}")
    print(f"Sample embedding keys: {list(embeddings.keys())[:5]}")
    
    # Print the structure of the first embedding
    first_key = list(embeddings.keys())[0]
    first_embedding = embeddings[first_key]
    print(f"Structure of first embedding:")
    print(f"Type: {type(first_embedding)}")
    print(f"Length: {len(first_embedding)}")
    print(f"Type of first element: {type(first_embedding[0])}")
    print(f"First few elements: {first_embedding[:2]}")
    
    if isinstance(first_embedding[0], list):
        print(f"Length of first sub-element: {len(first_embedding[0])}")
        print(f"Type of first sub-element's first item: {type(first_embedding[0][0])}")
        print(f"First few items of first sub-element: {first_embedding[0][:5]}")

    # Don't proceed with index creation for now
    print("Debug information printed. Index creation skipped.")

if __name__ == "__main__":
    main()