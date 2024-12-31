import nltk
from nltk import pos_tag, word_tokenize
import re

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('averaged_perceptron_tagger_eng')
def pos_tag_and_parse(text):
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    return tagged

def extract_personal_info(text):
    info = {
        'name': '',
        'age': '',
        'date_of_birth': '',
        'education': '',
        'place_of_birth': ''
    }
    
    lines = text.split('\n')
    
    for line in lines:
        if line.startswith('Born'):
            info['name'] = line.replace('Born', '').strip()
        elif '(age' in line:
            age_match = re.search(r'\(age (\d+)\)', line)
            if age_match:
                info['age'] = age_match.group(1)
            date_match = re.search(r'([A-Za-z]+ \d+, \d{4})', line)
            if date_match:
                info['date_of_birth'] = date_match.group(1)
        elif 'Education' in line:
            info['education'] = line.replace('Education', '').strip()
        elif 'Pretoria' in line:
            info['place_of_birth'] = line.split('Citizenship')[0].strip()

    return info

if __name__ == "__main__":
    text = """Born Elon Reeve Musk
June 28, 1971 (age 50)
Pretoria, Transvaal, South Africa Citizenship
South Africa 
Education University of Pennsylvania (BS, BA)"""
    
    print("Part-of-Speech Tags:")
    tagged = pos_tag_and_parse(text)
    for token, tag in tagged:
        print(f"{token}: {tag}")
        
    print("\nExtracted Personal Information:")
    info = extract_personal_info(text)
    for key, value in info.items():
        print(f"{key}: {value}")
