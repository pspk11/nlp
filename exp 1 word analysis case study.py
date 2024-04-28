import spacy

# Load English tokenizer, tagger, parser, NER, and word vectors
nlp = spacy.load("en_core_web_sm")

# Sample customer feedback data
customer_feedback = [
    "The product is amazing! I love the quality.",
    "The customer service was terrible, very disappointed.",
    "Great experience overall, highly recommended.",
    "The delivery was late, very frustrating."
]

def analyze_feedback(feedback):
    for idx, text in enumerate(feedback, start=1):
        print(f"\nAnalyzing Feedback {idx}: '{text}'")
        doc = nlp(text)
        tokens = [token.text for token in doc]
        lemmas = [token.lemma_ for token in doc]
        print("Tokens:", tokens)
        print("Lemmas:", lemmas)
        print("\nDependency Parsing:")
        for token in doc:
            print(token.text, token.dep_, token.head.text, token.head.pos_,
                  [child for child in token.children])

if __name__ == "__main__":
    analyze_feedback(customer_feedback)
