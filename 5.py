import nltk
from nltk import pos_tag, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def perform_morphological_analysis(text):
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    
    lemmatizer = WordNetLemmatizer()
    lemmatized = []
    
    for word, pos in pos_tags:
        wordnet_pos = get_wordnet_pos(pos)
        lemma = lemmatizer.lemmatize(word, wordnet_pos)
        lemmatized.append((word, pos, lemma))
    
    return lemmatized

def pos_tag_with_perceptron(text):
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    return tagged

if __name__ == "__main__":
    text = "The quick brown foxes are jumping over the lazy dogs and running through fields"
    
    print("Morphological Analysis and Lemmatization:")
    results = perform_morphological_analysis(text)
    for word, pos, lemma in results:
        print(f"Word: {word:15} POS: {pos:6} Lemma: {lemma}")
        
    print("\nPerceptron POS Tagging:")
    perceptron_tags = pos_tag_with_perceptron(text)
    for word, tag in perceptron_tags:
        print(f"Word: {word:15} Tag: {tag}")
