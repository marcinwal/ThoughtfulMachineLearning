#chapter 6 
#clustering with K-means

from sklearn.metrics import *

def Kmean():
  import numpy as np
  from sklearn.cluster import KMeans 
  from scipy.spatial.distance import cdist
  import matplotlib.pyplot as plt 

  cluster1 = np.random.uniform(0.5,1.5,(2,10))
  cluster2 = np.random.uniform(3.5,4.5,(2,10))

  X = np.hstack((cluster1,cluster2)).T
  X = np.vstack((cluster1,cluster2)).T

  K = range(1,10)
  meandistortions = []
  for k in K:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    meandistortions.append(sum(np.min(cdist(X,kmeans.cluster_centers_,'euclidean'),axis=1))/X.shape[0])

    plt.plot(K,meandistortions,'bx-')
    plt.xlabel('k')
    plt.ylabel('avg distorion')
    plt.title('Selecing k with Elbow method')
    plt.show()

# Kmean()

def clustersEval():
  import numpy as np 
  from sklearn.cluster import KMeans
  from sklearn import metrics 
  import matplotlib.pyplot as plt 

  plt.subplot(3,2,1)
  x1 = np.array([1,2,3,1,5,6,5,5,6,7,8,9,7,9])
  x2 = np.array([1,3,2,2,8,6,7,6,7,1,2,1,1,3])
  X = np.array(zip(x1,x2)).reshape(len(x1),2)
  plt.xlim([0,10])
  plt.ylim([0,10])
  colors = ['b','g','r','c','m','y','k','b']
  markers = ['o','s','D','v','^','p','*','+']
  tests = [2,3,4,5,8]
  subplot_counter = 1
  for t in tests:
    subplot_counter += 1
    plt.subplot(3,2,subplot_counter)
    kmeans_model = KMeans(n_clusters=t).fit(X)

    for i,l in enumerate(kmeans_model.labels_):
      plt.plot(x1[i],x2[i],color=colors[l],marker=markers[l])
      plt.xlim([0,10])
      plt.ylim([0,10])
      plt.title('K = %s, silhoutte coefficient = %.03f' % (t,metrics.silhouette_score(X,kmeans_model.labels_,metric='euclidean')))
  plt.show()

def imageQuant():
  import numpy as np 
  import matplotlib.pyplot as plt 
  from sklearn.cluster import KMeans
  from sklearn.utils import shuffle
  import mahotas as mh 


  original_img = np.array(mh.imread('./data/malpa.png'),dtype=np.float64)/255
  original_dimension = tuple(original_img.shape)
  width,height,depth = tuple(original_img.shape)
  image_flattened = np.reshape(original_img,(width*height,depth))
  #K-means culsters from sample of 1000 selected colors; 
  image_array_sample = shuffle(image_flattened,random_state=0)[:1000]
  estimator = KMeans(n_clusters=64,random_state=0)
  estimator.fit(image_array_sample)
  #predictions of clsters for each pixels of orig image
  cluster_assignments = estimator.predict(image_flattened)
  compressed_palette = estimator.cluster_centers_
  compressed_img = np.zeros((width,height,compressed_palette.shape[1]))
  label_idx = 0
  for i in range(width):
    for j in range(height):
      compressed_img[i][j] = compressed_palette[cluster_assignments[label_idx]]
      label_idx += 1
  plt.subplot(122)
  plt.title('original_img')
  plt.imshow(original_img)
  plt.axis('off')
  plt.subplot(121)
  plt.title('compressed_img')
  plt.imshow(compressed_img)
  plt.axis('off')
  plt.show()


def catsAnddogs():
  import numpy as np 
  import mahotas as mh
  from mahotas.features import surf 
  from sklearn.linear_model import LogisticRegression
  import glob
  from sklearn.cluster import MiniBatchKMeans


  all_instance_filenames = []
  all_instance_targets = []

  for f in glob.glob('./data/train/*.jpg'):
    target = 1 if 'cat' in f else 0
    all_instance_filenames.append(f)
    all_instance_targets.append(target)

  surf_features = []
  counter = 0
  for f in all_instance_filenames:
    print 'reading image:',f
    image = mh.imread(f,as_grey=True)
    surf_features.append(surf.surf(image)[:,5:])

  train_len = int(len(all_instance_filenames)*.6)
  X_train_surf_features = np.concatenate(surf_features[:train_len])
  X_test_surf_features = np.concatenate(surf_features[train_len:])
  y_train = all_instance_targets[:train_len]
  y_test = all_instance_targets[train_len:]

  n_clusters = 300
  print 'Clustering', len(X_train_surf_features), 'features'
  estimator = MiniBatchKMeans(n_clusters=n_clusters)
  estimator.fit_transform(X_train_surf_features)

  X_train = []
  for instance in surf_features[:train_len]:
    clusters = estimator.predict(instance)
    features = np.bincount(clustes)
    if len(features) < n_clusters:
      features = np.append(features,np.zeros((1,n_clusters-len(features))))
    X_train.append(features)

  X_test = []
  for instance in surf_features[train_len:]:
    clusters = estimator.predict(instance)
    features = np.bincount(clustes)
    if len(features) < n_clusters:
      features = np.append(features,np.zeros((1,n_clusters-len(features))))
    X_test.append(features)    

  clf = LogisticRegression(C=0.001,penalty='l2')
  clf.fit_transform(X_train,y_train)
  predictions = clf.predict(X_test)
  print classification_report(y_test,predictions)
  print 'precision:', precision_score(y_test,predictions)
  print 'recall:', recall_score(y_test,predictions)
  print 'accuracy:', accuracy_score(y_test,predictions)


# clustersEval()
# imageQuant()
catsAnddogs()
