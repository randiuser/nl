from nltk.util import ngrams
from collections import Counter
import nltk

nltk.download('punkt')

def analyze_bigrams(documents):
    all_bigrams = []
    
    for doc in documents:
        tokens = nltk.word_tokenize(doc.lower())
        doc_bigrams = list(ngrams(tokens, 2))
        all_bigrams.extend(doc_bigrams)
    
    bigram_freq = Counter(all_bigrams)
    total_bigrams = len(bigram_freq)
    top_5 = bigram_freq.most_common(5)
    
    return total_bigrams, top_5

if __name__ == "__main__":
    documents = [
        "Natural language processing is a field of artificial intelligence.",
        "Artificial intelligence helps computers understand human language.",
        "Machine learning is a subset of artificial intelligence."
    ]
    
    total, top_bigrams = analyze_bigrams(documents)
    
    print(f"Total unique bigrams: {total}")
    print("\nTop 5 most common bigrams:")
    for bigram, freq in top_bigrams:
        print(f"{bigram}: {freq}")
