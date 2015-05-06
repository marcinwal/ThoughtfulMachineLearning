# Principal Component Extraction or Karhunen-Loeve Transform
#search for patterns in high-dimensional data
#idea of projecting 3d into 2d plane etc..
#it is used if variance it unevenly distriputed accross the dimensions
def simpleCov():
  import numpy as np
  X = [[2,0.0,-1.4],
       [2.2,0,2,-1.5],
       [2.4,0.1,-1.],
       [1.9,0.0,-1.2]]

  print np.cov(np.array(X).T)

def eigenCheck():
  import numpy as np 
  w,v = np.linalg.eig(np.array([[1, -2],[2,-3]]))
  print w,v   

def iris():
  import matplotlib.pyplot as plt 
  from sklearn.decomposition import PCA
  from sklearn.datasets import load_iris

  data = load_iris()
  y = data.target
  X = data.data 
  pca = PCA(n_components = 2)
  reduced_X = pca.fit_transform(X)
  red_x,red_y = [],[]
  blue_x,blue_y = [],[]
  green_x,green_y = [],[]
  for i in range(len(reduced_X)):
    if y[i]== 0:
      red_x.append(reduced_X[i][0])
      red_y.append(reduced_X[i][1])
    elif y[i]== 1:  
      blue_x.append(reduced_X[i][0])
      blue_y.append(reduced_X[i][1])
    else:
      green_x.append(reduced_X[i][0])
      green_y.append(reduced_X[i][1])     

  plt.scatter(red_x,red_y,c='r',marker='x') 
  plt.scatter(blue_x,blue_y,c='b',marker='D') 
  plt.scatter(green_x,green_y,c='g',marker='.') 
  plt.show()

def faces():
  from os import walk,path
  import numpy as np
  import mahotas as mh 
  from sklearn.cross_validation import train_test_split
  from sklearn.cross_validation import cross_val_score
  from sklearn.preprocessing import scale 
  from sklearn.decomposition import PCA 
  from sklearn.linear_model import LogisticRegression
  from sklearn.metrics import classification_report
  X = []
  y = []
  for dir_path,dir_names,file_names in walk('./data/att_faces'):
    for fn in file_names:
      if fn[-3:] == 'pgm':
        image_filename = path.join(dir_path,fn)
        X.append(scale(mh.imread(image_filename,as_grey=True).reshape(10304).astype('float32')))
        y.append(dir_path)
  X = np.array(X)
  X_train,X_test,y_train,y_test = train_test_split(X,y)
  pca = PCA(n_components = 150)
  X_train_reduced = pca.fit_transform(X_train)
  X_test_reduced = pca.transform(X_test)
  print 'original data were',X_train.shape
  print 'reduced is ',X_train_reduced.shape

  classifier = LogisticRegression()
  accuracies = cross_val_score(classifier,X_train_reduced,y_train)

  print 'cross val: ',np.mean(accuracies),accuracies
  classifier.fit(X_train_reduced,y_train)
  predictions = classifier.predict(X_test_reduced)
  print classification_report(y_test,predictions)



# simpleCov()
# eigenCheck()
# iris()
faces()