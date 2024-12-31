import nltk
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer
from nltk.stem import WordNetLemmatizer
import spacy

nltk.download('wordnet')

def compare_stemmers_and_lemmatizer(words):
    porter = PorterStemmer()
    lancaster = LancasterStemmer()
    snowball = SnowballStemmer('english')
    lemmatizer = WordNetLemmatizer()
    
    results = []
    for word in words:
        result = {
            'word': word,
            'porter': porter.stem(word),
            'lancaster': lancaster.stem(word),
            'snowball': snowball.stem(word),
            'lemmatizer': lemmatizer.lemmatize(word)
        }
        results.append(result)
    return results

def analyze_dependencies(text):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    
    dependencies = []
    for token in doc:
        dep = {
            'text': token.text,
            'dep': token.dep_,
            'head': token.head.text,
            'head_pos': token.head.pos_,
            'children': [child.text for child in token.children]
        }
        dependencies.append(dep)
    return dependencies

if __name__ == "__main__":
    words = ['running', 'flies', 'argued', 'happily', 'flying']
    results = compare_stemmers_and_lemmatizer(words)
    
    print("Stemmer and Lemmatizer Comparison:")
    for result in results:
        print(f"\nWord: {result['word']}")
        print(f"Porter: {result['porter']}")
        print(f"Lancaster: {result['lancaster']}")
        print(f"Snowball: {result['snowball']}")
        print(f"Lemmatizer: {result['lemmatizer']}")
    
    text = "The quick brown fox jumps over the lazy dog."
    dependencies = analyze_dependencies(text)
    
    print("\nDependency Analysis:")
    for dep in dependencies:
        print(f"\nToken: {dep['text']}")
        print(f"Dependency: {dep['dep']}")
        print(f"Head: {dep['head']} ({dep['head_pos']})")
        print(f"Children: {', '.join(dep['children'])}")
