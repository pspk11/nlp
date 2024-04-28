import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
# Sample text for POS tagging
text = "Parts of speech tagging helps to understand the function of each word in a sentence."
# Tokenize the text into words
tokens = nltk.word_tokenize(text)
# Perform POS tagging
pos_tags = nltk.pos_tag(tokens)
# Display the POS tags
print("POS tags:", pos_tags)
