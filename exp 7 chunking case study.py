import nltk
import os

# Set NLTK data path
nltk.data.path.append("/usr/local/share/nltk_data")

# Download the 'punkt' tokenizer model
nltk.download('punkt')

# Download the 'averaged_perceptron_tagger' model
nltk.download('averaged_perceptron_tagger')

# Sample text
text = "The quick brown fox jumps over the lazy dog."

# Tokenize the text into words
words = nltk.word_tokenize(text)

# Perform part-of-speech tagging
pos_tags = nltk.pos_tag(words)

# Define chunk grammar
chunk_grammar = r"""
NP: {<DT>?<JJ>*<NN>} # Chunk sequences of DT, JJ, NN
"""

# Create chunk parser
chunk_parser = nltk.RegexpParser(chunk_grammar)

# Apply chunking
chunked_text = chunk_parser.parse(pos_tags)

# Extract noun phrases
noun_phrases = []
for subtree in chunked_text.subtrees(filter=lambda t: t.label() == 'NP'):
    noun_phrases.append(' '.join(word for word, tag in subtree.leaves()))

# Output
print("Original Text:", text)
print("Noun Phrases:")
for phrase in noun_phrases:
    print("-", phrase)
