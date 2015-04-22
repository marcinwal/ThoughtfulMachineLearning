
#vectorizing text and making vectors danse at the same time 
corpus = [
    'He ate the sandwiches',
    'Every sandwich was eaten by him'
]

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

def lemmaComapredWithStemming():
  from nltk.stem import PorterStemmer
  stemmer = PorterStemmer()
  print stemmer.stem('gathering')

def lemmatize(token,tag,lemmatizer):
  if tag[0].lower() in ['n','v']:
    return lemmatizer.lemmatize(token,tag[0].lower())
  return token

def lemmaFull():
  from nltk import word_tokenize
  from nltk.stem import PorterStemmer
  from nltk.stem.wordnet import WordNetLemmatizer
  from nltk import pos_tag
  wordnet_tags = ['n','v']
  stemmer = PorterStemmer()
  lemmatizer = WordNetLemmatizer()

  print 'Stemmed:',[[stemmer.stem(token) for token in word_tokenize(document)] for 
                   document in corpus]
  tagged_corpus = [pos_tag(word_tokenize(document)) for document in corpus]
  print 'Lemmatized:' ,[[lemmatize(token,tag,lemmatizer) for token,tag in document] for 
                        document in tagged_corpus]

# vectorized()
# simpleLemmatizer()
lemmaFull();