import spacy

# Load English tokenizer, tagger, parser, NER, and word vectors
nlp = spacy.load("en_core_web_sm")

# Sample text for analysis
text = "Natural Language Processing is a fascinating field of study."

# Process the text with spaCy
doc = nlp(text)

# Extracting tokens and lemmatization
tokens = [token.text for token in doc]
lemmas = [token.lemma_ for token in doc]
print("Tokens:", tokens)
print("Lemmas:", lemmas)

# Dependency parsing
print("\nDependency Parsing:")
for token in doc:
    print(token.text, token.dep_, token.head.text, token.head.pos_,
          [child for child in token.children])
