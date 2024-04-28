import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

# Download NLTK resources (if not already downloaded)
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def pos_tagging(text):
    sentences = sent_tokenize(text)
    tagged_tokens = []
    for sentence in sentences:
        tokens = word_tokenize(sentence)
        tagged_tokens.extend(nltk.pos_tag(tokens))
    return tagged_tokens

def main():
    article_text = """Manchester United secured a 3-1 victory over Chelsea in yesterday's match.
    Goals from Rashford, Greenwood, and Fernandes sealed the win for United.
    Chelsea's only goal came from Pulisic in the first half.
    The victory boosts United's chances in the Premier League title race.
    """
    tagged_tokens = pos_tagging(article_text)
    print("Original Article Text:\n", article_text)
    print("\nParts of Speech Tagging:")
    for token, pos_tag in tagged_tokens:
        print(f"{token}: {pos_tag}")

if __name__ == "__main__":
    main()
