
#vectorizing text and making vectors danse at the same time 
def vectorized():
  from sklearn.feature_extraction.text import CountVectorizer
  corpus = [
    'UNC played Duke in basketball',
    'Duke losr the basketball game',
    'I ate the sandwich'
  ]

  vectorizer = CountVectorizer(stop_words="english")
  print vectorizer.fit_transform(corpus).todense()
  print vectorizer.vocabulary_

#stemming recognizing words on their 'core'
def simpleLemmatizer():
  from nltk.stem.wordnet import WordNetLemmatizer
  lemmatizer = WordNetLemmatizer()
  print lemmatizer.lemmatize('gathering','v')
  print lemmatizer.lemmatize('gathering','n')


# vectorized()
simpleLemmatizer()
