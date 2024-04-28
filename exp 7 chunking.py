import nltk
from nltk import RegexpParser
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

# Download NLTK resources (run only once if not downloaded)
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Sample sentence
sentence = "The quick brown fox jumps over the lazy dog"

# Tokenize the sentence
tokens = word_tokenize(sentence)

# POS tagging
tagged = pos_tag(tokens)

# Define a chunk grammar using regular expressions
# NP (noun phrase) chunking: "NP: {<DT>?<JJ>*<NN>}"
# This grammar captures optional determiner (DT), adjectives (JJ), and nouns (NN) as a noun phrase
chunk_grammar = r"""
NP: {<DT>?<JJ>*<NN>}
"""

# Create a chunk parser with the defined grammar
chunk_parser = RegexpParser(chunk_grammar)

# Parse the tagged sentence to extract chunks
chunks = chunk_parser.parse(tagged)

# Display the chunks
for subtree in chunks.subtrees():
    if subtree.label() == 'NP':
        print(subtree)
