import nltk
from collections import defaultdict
import numpy as np

nltk.download('punkt')

class BigramModel:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.unigram_counts = defaultdict(int)
        self.bigram_counts = defaultdict(lambda: defaultdict(int))
        self.vocab = set()
        
    def train(self, text):
        tokens = nltk.word_tokenize(text.lower())
        tokens = ['<s>'] + tokens + ['</s>']
        self.vocab.update(tokens)
        
        for i in range(len(tokens)-1):
            self.unigram_counts[tokens[i]] += 1
            self.bigram_counts[tokens[i]][tokens[i+1]] += 1
            
    def get_smoothed_probability(self, word1, word2):
        V = len(self.vocab)
        numerator = self.bigram_counts[word1][word2] + self.alpha
        denominator = self.unigram_counts[word1] + (self.alpha * V)
        return numerator / denominator
    
    def score_sequence(self, sequence):
        tokens = ['<s>'] + nltk.word_tokenize(sequence.lower()) + ['</s>']
        score = 0
        
        for i in range(len(tokens)-1):
            prob = self.get_smoothed_probability(tokens[i], tokens[i+1])
            score += np.log2(prob)
        
        return score

if __name__ == "__main__":
    training_text = """Natural language processing helps computers understand 
    and analyze human language. The field of NLP combines linguistics and 
    artificial intelligence."""
    
    model = BigramModel(alpha=1.0)
    model.train(training_text)
    
    test_sequences = [
        "language processing helps",
        "artificial language helps",
        "natural artificial field"
    ]
    
    print("Sequence Probabilities (log2):")
    for seq in test_sequences:
        score = model.score_sequence(seq)
        print(f"{seq}: {score:.4f}")
        
    print("\nSample Bigram Probabilities:")
    word_pairs = [("natural", "language"), ("artificial", "intelligence")]
    for w1, w2 in word_pairs:
        prob = model.get_smoothed_probability(w1, w2)
        print(f"P({w2}|{w1}) = {prob:.4f}")
