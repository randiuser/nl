import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import matplotlib.pyplot as plt
from nltk.util import ngrams

nltk.download('punkt')
nltk.download('stopwords')

def create_ngrams(text, n=2):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token.isalnum() and token not in stop_words]
    n_grams = list(ngrams(tokens, n))
    return n_grams

def plot_word_frequency(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token.isalnum() and token not in stop_words]
    
    word_freq = Counter(tokens)
    most_common = word_freq.most_common(10)
    words, frequencies = zip(*most_common)
    
    plt.figure(figsize=(12, 6))
    plt.bar(words, frequencies)
    plt.title('Most Frequent Words Distribution (Excluding Stop Words)')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    text = "Artificial intelligence has made significant advancements, but it still struggles to understand human emotions and context, limiting its ability to interact naturally."
    
    print("N-grams (excluding stop words):")
    ngrams_list = create_ngrams(text)
    for gram in ngrams_list:
        print(gram)
        
    plot_word_frequency(text)
