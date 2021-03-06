import numpy as nps
from skimage.feature import corner_harris,corner_peaks
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import skimage.io as io 
from skimage.exposure import equalize_hist


def imageExtraction():
  from sklearn import datasets
  digits = datasets.load_digits()
  print 'Digit:',digits.target[0]
  print digits.images[0]
  print 'Feature vector:\n',digits.images[0].reshape(-1,64)

def show_corners(corners,image):
  fig = plt.figure()
  plt.gray()
  plt.imshow(image)
  y_corner, x_corner = zip(*corners)
  plt.plot(x_corner,y_corner,'or')
  plt.xlim(0,image.shape[1])
  plt.ylim(image.shape[0],0)
  fig.set_size_inches(nps.array(fig.get_size_inches()) * 1.5)
  plt.show()

def featureExtractImage():
  malpa = io.imread('data/malpa.png')
  malpa = equalize_hist(rgb2gray(malpa))
  corners = corner_peaks(corner_harris(malpa),min_distance=2)
  show_corners(corners,malpa)

#scale invariant feature transform and spedup robust features
#they are in mahotas library !!!!
def SIFT_SURF():
  import mahotas as mh 
  from mahotas.features import surf 
  image = mh.imread('data/malpa.png',as_grey=True)
  print 'The 1st SURF descriptor is:\n',surf.surf(image)[0]
  print 'Extracted %s SURF descriptors' %len(surf.surf([image]))

def normalizingExample():
  from sklearn import preprocessing 
  import numpy as np 
  X = np.array([
    [0.0,4.,13.,9.,-1.,2.],
    [0.,1.,13.,15.,10.,15.],
    [0.,3.,15.,2.,0.,11.]
    ])
  print preprocessing.scale(X)

# imageExtraction()
# featureExtractImage()
# SIFT_SURF()
normalizingExample()