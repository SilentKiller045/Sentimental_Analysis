from flask import Flask, render_template, request
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('punkt')



app = Flask(__name__)


@app.route('/', methods=["GET", "POST"])
def main():
    filename = 'SA.html'
    if request.method == "POST":
        z = request.form.get("inp")
        score = sentiment_analysis(z)
        if score == 1:
            return render_template(filename, message="Positive😊😊😊")
        elif score == 0:
            return render_template(filename, message="Negative😢😢😢")
    return render_template(filename)


def sentiment_analysis(text):
    text = preprocess_text(text)
    return get_sentiment(text)

def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(text.lower())
    # Remove stop words
    filtered_tokens = [token for token in tokens if token not in stopwords.words('english')]
    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    processed_text = ' '.join(lemmatized_tokens)
    return processed_text

def get_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    sentiment = 1 if scores['pos'] > 0 else 0
    return sentiment