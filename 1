import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


nltk.download('punkt_tab')
nltk.download('stopwords')

def extract_twitter_handles(text):
    """
    Extract Twitter handles from text.
    A Twitter handle comes after https://twitter.com/ and contains only
    alphanumeric characters and underscores.
    """
    pattern = r'https://twitter\.com/([A-Za-z0-9_]+)'

    handles = re.findall(pattern, text)
    
    return handles

def preprocess_text(text):
    """
    Perform text preprocessing including:
    1. Tokenization
    2. Stop word removal
    3. Stemming
    """
    tokens = word_tokenize(text)

    tokens = [token.lower() for token in tokens]

    tokens = [token for token in tokens if token.isalnum()]

    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    
    return {
        'original_tokens': tokens,
        'stemmed_tokens': stemmed_tokens
    }

if __name__ == "__main__":
    sample_text = """
    Check out our profile at https://twitter.com/PythonDev and follow
    our team members at https://twitter.com/NLP_Expert and
    https://twitter.com/AI_Researcher123
    """

    handles = extract_twitter_handles(sample_text)
    print("Found Twitter handles:")
    for handle in handles:
        print(f"- {handle}")

    sample_text_2 = """
    Natural Language Processing (NLP) is a branch of artificial intelligence
    that helps computers understand, interpret, and manipulate human language.
    """

    result = preprocess_text(sample_text_2)
    
    print("\nOriginal tokens:")
    print(result['original_tokens'])
    
    print("\nStemmed tokens:")
    print(result['stemmed_tokens'])
