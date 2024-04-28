import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download NLTK resources (only required once)
nltk.download('vader_lexicon')

# Sample reviews
reviews = [
    "This product is amazing! I love it.",
    "The product was good, but the packaging was damaged.",
    "Very disappointing experience. Would not recommend.",
    "Neutral feedback on the product.",
]

# Initialize Sentiment Intensity Analyzer
sid = SentimentIntensityAnalyzer()

# Analyze sentiment for each review
for review in reviews:
    print("Review:", review)
    scores = sid.polarity_scores(review)
    print("Sentiment:", end=' ')
    if scores['compound'] > 0.05:
        print("Positive")
    elif scores['compound'] < -0.05:
        print("Negative")
    else:
        print("Neutral")
    print()
