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
  from sklearn.metrics import accuracy_score
  import matplotlib.pyplot as plt 

  y_test = [0,0,0,0,0,1,1,1,1,1]
  y_pred = [0,1,0,0,0,0,0,1,1,1]

  confusion_matrix = confusion_matrix(y_test,y_pred)
  print(confusion_matrix)
  print 'Accuracy is:',accuracy_score(y_pred,y_test)

  plt.matshow(confusion_matrix)
  plt.grid(True)
  plt.title('Confusion matrix')
  plt.colorbar()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.show()

def evaluationOfClassifier():
  import numpy as np
  import pandas as pd 
  from sklearn.feature_extraction.text import TfidfVectorizer
  from sklearn.linear_model.logistic import LogisticRegression
  from sklearn.cross_validation import train_test_split, cross_val_score

  df = pd.read_csv('data/sms.csv')
  X_train_raw, X_test_raw, y_train, y_test = \
                    train_test_split(df['message'],df['label'])
  vectorizer = TfidfVectorizer()
  X_train = vectorizer.fit_transform(X_train_raw)
  X_test = vectorizer.transform(X_test_raw)
  classifier = LogisticRegression()
  classifier.fit(X_train, y_train)
  scores = cross_val_score(classifier,X_train,y_train,cv=5)
  print np.mean(scores),scores

def classifierPrecission():
  import numpy as np 
  import pandas as pd 
  import matplotlib.pyplot as plt
  from sklearn.feature_extraction.text import TfidfVectorizer
  from sklearn.linear_model.logistic import LogisticRegression
  from sklearn.cross_validation import train_test_split, cross_val_score
  from sklearn.metrics import roc_curve, auc


  df = pd.read_csv('data/sms.csv')
  X_train_raw, X_test_raw, y_train, y_test = train_test_split \
                                      (df['message'],df['label'])
  vectorizer = TfidfVectorizer()
  X_train = vectorizer.fit_transform(X_train_raw)
  X_test = vectorizer.transform(X_test_raw)
  classifier = LogisticRegression()
  classifier.fit(X_train,y_train)
  precisions = cross_val_score(classifier,X_train,y_train,cv=5,scoring='precision')
  print 'precission:', np.mean(precisions),precisions

  recalls = cross_val_score(classifier,X_train,y_train,cv=5,scoring='recall')
  print 'recalls:',np.mean(recalls),recalls

  #f1 = 2*PR/(P+R) for perfect should be 1 
  f1s = cross_val_score(classifier,X_train,y_train,cv=5,scoring='f1')
  print 'f1s:',np.mean(f1s),f1s

  #ROC  Receiver operating characteristic ROC Currve clasisfier performance 
  #its classifier recall against its fall-out 
  #F = FP /(TN + FP)
  predictions = classifier.predict_proba(X_test)
  false_positive_rate,recall,thresholds = roc_curve(y_test,predictions[:,1])
  roc_auc = auc(false_positive_rate,recall)
  plt.title('ROC')
  plt.plot(false_positive_rate,recall,'b',label='AUC = %0.2f' %roc_auc)
  plt.legend(loc='lower right')
  plt.plot([0,1],[0,1],'r--')
  plt.xlim([0.0,1.0])
  plt.ylim([0.0,1.0])
  plt.ylabel('Recall')
  plt.xlabel('fall-out')
  plt.show()

# readCollection()
# predictions()
# confusionCheck()
# evaluationOfClassifier() 
classifierPrecission()
