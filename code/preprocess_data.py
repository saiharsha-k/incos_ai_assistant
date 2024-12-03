from transformers import BertTokenizer
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re
import nltk
import json
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Download NLTK stopwords, punkt tokenizer, and WordNet data for the first time
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

class PreprocessData:
    def __init__(self, model_name="bert-base-uncased", json_path="path_to_your_json_file.json"):
        """Initialize BERT tokenizer, stopwords, lemmatizer, and stemmer."""
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.json_path = json_path

    def extract_text(self):
        """Extract text from JSON recursively, and organize it by section."""
        sections = {}
        with open(self.json_path, 'r') as file:
            data = json.load(file)
            sections = self._extract_from_data(data)
        return sections

    def _extract_from_data(self, data):
        """Recursively extract text from nested data (list/dict) and categorize by sections."""
        sections = {}
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, str):  # If value is a string, it's content
                    sections[key] = value
                else:
                    sections.update(self._extract_from_data(value))
        elif isinstance(data, list):
            for item in data:
                sections.update(self._extract_from_data(item))
        return sections

    def clean_text(self, text):
        """Clean the text by removing special characters and extra spaces."""
        text = re.sub(r"[^\w\s]", "", text)  # Remove special characters
        text = re.sub(r"\s+", " ", text)  # Remove extra spaces
        return text.strip().lower()

    def tokenize(self, text):
        """Tokenize the text using BERT tokenizer."""
        return self.tokenizer.tokenize(text)

    def lemmatize_and_stem(self, word):
        """Lemmatize and stem a word."""
        # Lemmatize the word
        lemmatized_word = self.lemmatizer.lemmatize(word)
        # Stem the word
        stemmed_word = self.stemmer.stem(lemmatized_word)
        return stemmed_word

    def preprocess(self, sections):
        """Preprocess text data: Clean, remove stopwords, lemmatize, stem, and tokenize."""
        preprocessed_data = {}
        all_words = []  # To gather all words for word cloud
        
        for section, text in sections.items():
            cleaned_text = self.clean_text(text)
            tokens = self.tokenize(cleaned_text)
            filtered_tokens = [word for word in tokens if word not in self.stop_words]
            
            # Lemmatize and stem the filtered tokens
            lemmatized_stemmed_tokens = [self.lemmatize_and_stem(word) for word in filtered_tokens]
            preprocessed_data[section] = lemmatized_stemmed_tokens
            
            # Add to word cloud word list
            all_words.extend(lemmatized_stemmed_tokens)

        # Generate and display word cloud
        self.create_wordcloud(all_words)
        self.save_processed_data(preprocessed_data)
        
        return preprocessed_data
    
    def save_processed_data(self, processed_data, output_file="processed_data_for_training.json"):
        """Save preprocessed data to a JSON file."""
        try:
            with open(output_file, 'w') as file:
                json.dump(processed_data, file, indent=4)
            print(f"Preprocessed data saved to {output_file}")
        except Exception as e:
            print(f"Error saving preprocessed data: {e}")

    def create_wordcloud(self, words):
        """Generate and display a word cloud."""
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(" ".join(words))
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()


# Example usage
if __name__ == "__main__":
    # Specify the path to your JSON file
    json_path = "E:/Sai Harsha/From_23/MSc Data Science/DeepLearning_Applications/coursework_assistant/code/processed_data.json"  # Replace with the actual path
    
    # Initialize the PreprocessData object
    extractor = PreprocessData(json_path=json_path)
    
    # Extract the text from JSON
    sections = extractor.extract_text()

    # Preprocess the extracted text (tokenization, stopwords removal, lemmatization, stemming)
    preprocessed_data = extractor.preprocess(sections)
    
    # Display the preprocessed sections
    print("Preprocessed Data:")
    for section, tokens in preprocessed_data.items():
        print(f"Section: {section} | Processed Tokens: {tokens[:5]}...")  # Display the first 5 tokens for each section
