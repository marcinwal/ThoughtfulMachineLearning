def readCollection():
  import pandas as pd
  df = pd.read_csv('data/SMSSpamCollection',delimiter='\t',header = None)
  print df.head()
  print 'Spam:', df[df[0] == 'spam'][0].count()
  print 'Ham:',df[df[0] == 'ham'][0].count()


def predictions():
  import numpy as np 
  import pandas as pd 
  from sklearn.feature_extraction.text import TfidfVectorizer
  from sklearn.linear_model.logistic import LogisticRegression
  from sklearn.cross_validation import train_test_split,cross_val_score

  df = pd.read_csv('data/SMSSpamCollection',delimiter='\t',header = None)
  X_train_raw, X_test_raw, y_train, y_test = train_test_split(df[1],df[0])
  #75% goes to training set and 25% goes to test set

  vactorizer = TfidfVectorizer()
  X_train = vactorizer.fit_transform(X_train_raw)
  X_test = vactorizer.transform(X_test_raw)

  #the model
  classifier = LogisticRegression()
  classifier.fit(X_train, y_train)
  predictions = classifier.predict(X_test)
  for i,prediction in enumerate(predictions[:5]):
    print 'Prediction: %s. Message: %s' % (prediction, X_test_raw[i])
        
#ham treated as spam and spam classifed as ham   
#4 true negative, 3 true positive
#2 false negative, 1 false positive     
def confusionCheck():
  from sklearn.metrics import confusion_matrix
  import matplotlib.pyplot as plt 

  y_test = [0,0,0,0,0,1,1,1,1,1]
  y_pred = [0,1,0,0,0,0,0,1,1,1]

  confusion_matrix = confusion_matrix(y_test,y_pred)
  print(confusion_matrix)
  plt.matshow(confusion_matrix)
  plt.grid(True)
  plt.title('Confusion matrix')
  plt.colorbar()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.show()

# readCollection()
predictions()
confusionCheck()
