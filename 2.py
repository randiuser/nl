import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import matplotlib.pyplot as plt
import re

def plot_word_frequency(text):
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token.isalnum()]
    word_freq = Counter(tokens)
    most_common = word_freq.most_common(10)
    words, frequencies = zip(*most_common)
    
    plt.figure(figsize=(12, 6))
    plt.bar(words, frequencies)
    plt.title('Most Frequent Words Distribution')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def compare_text_splitting(text):
    nltk_tokens = word_tokenize(text)
    python_tokens = text.split()
    regex_tokens = re.findall(r'\b\w+\b', text)
    
    return {
        'nltk': nltk_tokens,
        'python': python_tokens,
        'regex': regex_tokens
    }

if __name__ == "__main__":
    sample_text = """Natural Language Processing (NLP) is a branch of artificial 
    intelligence that helps computers understand, interpret, and manipulate human language.
    NLP combines computational linguistics, machine learning, and deep learning models."""
    
    plot_word_frequency(sample_text)
    
    results = compare_text_splitting(sample_text)
    print("\nComparison of tokenization methods:")
    for method, tokens in results.items():
        print(f"\n{method.upper()} tokenization:")
        print(tokens)
