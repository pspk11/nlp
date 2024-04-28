import nltk
import random
nltk.download('punkt')
nltk.download('gutenberg')
words = nltk.corpus.gutenberg.words()
bigrams = list(nltk.bigrams(words))
starting_word = "the"
generated_text = [starting_word]
for _ in range(20):
  possible_words = [word2 for (word1, word2) in bigrams if word1.lower() == generated_text[-1].lower()]
  next_word = random.choice(possible_words)
  generated_text.append(next_word)
  print(' '.join(generated_text))
