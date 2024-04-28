import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import movie_reviews # Sample dataset from NLTK
# Download NLTK resources (run only once if not downloaded)
import nltk
nltk.download('movie_reviews')
# Load the movie_reviews dataset
documents = [(list(movie_reviews.words(fileid)), category)
for category in movie_reviews.categories()
for fileid in movie_reviews.fileids(category)]
# Convert data to DataFrame
df = pd.DataFrame(documents, columns=['text', 'sentiment'])
# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['sentiment'], test_size=0.2,
random_state=42)
# Initialize TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()
# Fit and transform the training data
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train.apply(' '.join))
# Initialize SVM classifier
svm_classifier = SVC(kernel='linear')
# Train the classifier
svm_classifier.fit(X_train_tfidf, y_train)
# Transform the test data
X_test_tfidf = tfidf_vectorizer.transform(X_test.apply(' '.join))
# Predict on the test data
y_pred = svm_classifier.predict(X_test_tfidf)
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
# Display classification report
print(classification_report(y_test, y_pred))
 
   
