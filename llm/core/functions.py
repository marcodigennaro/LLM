import nltk
from textblob import TextBlob
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import string

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')


def read_text_file(filepath):
    """Read the text file from the given filepath and return its contents as a string."""
    with open(filepath, 'r', encoding='utf-8') as file:
        text = file.read()
    return text


def clean_text(text):
    """Remove punctuation and stopwords from the text."""
    # Lowercase the text to standardize it
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = nltk.word_tokenize(text)
    cleaned_text = ' '.join(word for word in words if word not in stop_words)
    return cleaned_text


def analyze_sentiment(text):
    """Use TextBlob to compute the sentiment polarity of each sentence and return the results."""
    blob = TextBlob(text)
    sentiment_scores = [sentence.sentiment.polarity for sentence in blob.sentences]
    return sentiment_scores


def plot_sentiment(sentiment_scores):
    """Plot the sentiment scores of each sentence over time."""
    plt.figure(figsize=(10, 5))
    plt.plot(sentiment_scores, color='blue')
    plt.title('Sentiment Analysis over Text')
    plt.xlabel('Sentence Number')
    plt.ylabel('Sentiment Polarity')
    plt.axhline(y=0, color='r', linestyle='-')  # Red line at y=0 for neutrality reference
    plt.show()
