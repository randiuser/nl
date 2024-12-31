import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

nltk.download('vader_lexicon')

def analyze_vaccine_tweets():
    df = pd.read_csv('vaccination_tweets.csv')
    sia = SentimentIntensityAnalyzer()
    
    df['sentiment'] = df['text'].apply(lambda x: sia.polarity_scores(str(x))['compound'])
    df['sentiment_class'] = df['sentiment'].apply(lambda x: 'Positive' if x > 0 else 'Negative' if x < 0 else 'Neutral')
    
    sentiment_counts = df['sentiment_class'].value_counts()
    
    plt.figure(figsize=(8, 6))
    sentiment_counts.plot(kind='bar')
    plt.title('Sentiment Distribution in Vaccination Tweets')
    plt.ylabel('Number of Tweets')
    plt.xticks(rotation=45)
    plt.show()
    
    return df[['text', 'sentiment', 'sentiment_class']].head()

print(analyze_vaccine_tweets())
