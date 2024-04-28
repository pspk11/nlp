import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Initialize NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Function to perform semantic analysis
def semantic_analysis(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    synonyms = set()
    for token in lemmatized_tokens:
        for syn in wordnet.synsets(token):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name())
    return list(synonyms)

# Example customer queries
customer_queries = [
    "I received a damaged product. Can I get a refund?",
    "I'm having trouble accessing my account.",
    "How can I track my order status?",
    "The item I received doesn't match the description.",
    "Is there a discount available for bulk orders?"
]

# Semantic analysis for each query
for query in customer_queries:
    print("Customer Query:", query)
    synonyms = semantic_analysis(query)
    print("Semantic Analysis (Synonyms):", synonyms)
    print("\n")
