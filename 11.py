import nltk
from nltk import pos_tag, word_tokenize
from nltk.chunk import RegexpParser
from textblob import TextBlob

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def analyze_sentiment_and_pos(text):
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    
    grammar = """
    NP: {<DT>?<JJ.*>*<NN.*>+}
    ADJ: {<JJ.*>}
    """
    
    chunk_parser = RegexpParser(grammar)
    tree = chunk_parser.parse(tagged)
    
    nouns = []
    adjectives = []
    
    for subtree in tree.subtrees():
        if subtree.label() == 'NP':
            nouns.extend([token for token, pos in subtree.leaves() if pos.startswith('NN')])
        elif subtree.label() == 'ADJ':
            adjectives.extend([token for token, pos in subtree.leaves()])
    
    sentiment_scores = []
    for adj in adjectives:
        blob = TextBlob(adj)
        sentiment_scores.append((adj, blob.sentiment.polarity))
    
    overall_sentiment = sum(score for _, score in sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
    
    return {
        'nouns': nouns,
        'adjectives': adjectives,
        'sentiment_scores': sentiment_scores,
        'overall_sentiment': overall_sentiment
    }

if __name__ == "__main__":
    text = "The beautiful sunset created a peaceful and serene atmosphere in the busy city."
    results = analyze_sentiment_and_pos(text)
    
    print("Nouns:", results['nouns'])
    print("Adjectives:", results['adjectives'])
    print("\nAdjective Sentiment Scores:")
    for adj, score in results['sentiment_scores']:
        print(f"{adj}: {score:.2f}")
    print(f"\nOverall Sentiment Score: {results['overall_sentiment']:.2f}")
