import gensim.downloader as api
from nltk.tokenize import word_tokenize
# Download pre-trained word vectors (Word2Vec)
word_vectors = api.load("word2vec-google-news-300")
# Sample sentences
sentences = [
"Natural language processing is a challenging but fascinating field.",
"Word embeddings capture semantic meanings of words in a vector space."
]
# Tokenize sentences
tokenized_sentences = [word_tokenize(sentence.lower()) for sentence in sentences]
# Perform semantic analysis using pre-trained word vectors
for tokenized_sentence in tokenized_sentences:
    for word in tokenized_sentence:
        if word in word_vectors:
            similar_words = word_vectors.most_similar(word)
            print(f"Words similar to '{word}': {similar_words}")
        else:
            print(f"'{word}' is not in the pre-trained Word2Vec model.")
