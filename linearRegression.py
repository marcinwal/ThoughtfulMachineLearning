
def simpleRegressionWithChart():

    import matplotlib.pyplot as plt 
    import numpy as np
    from sklearn.linear_model import LinearRegression

    x=[[10],[20],[30],[27]]
    y=[[12],[35],[55],[44]]

    plt.figure()
    plt.title('points')
    plt.grid(True)
    plt.xlabel("X pizza diameer")
    plt.ylabel("y price")
    plt.plot(x,y,'k.')
    plt.axis([0,50,0,50])
    plt.show()

    model = LinearRegression()
    model.fit(x,y)
    print "Predcition of price of 34 diameteer is %.2f" % model.predict([29])[0]

    print "Residulal sum of squares %.2f" % np.mean((model.predict(x)-y)**2)

    print "variance of x is %.2f" %np.var(x)
    # print "cov of x and y is %.2f" %np.cov(x,y)[0][1]

def multiLinearExample():
  from numpy.linalg import inv
  from numpy import dot,transpose

  X = [[1,6,2],[1,8,1],[1,10,0],[1,14,2],[1,18,0]]
  y =[[7],[9],[13],[17.5],[18]]
  print dot(inv(dot(transpose(X),X)),dot(transpose(X),y))

def multiLinearExampleWithLeastSQR():
  from numpy.linalg import lstsq
  X = [[1,6,2],[1,8,1],[1,10,0],[1,14,2],[1,18,0]]
  y =[[7],[9],[13],[17.5],[18]]  
  print "\n Least sqr:"
  print lstsq(X,y)[0]
  print lstsq(X,y)[0]  

def regressionWithPredictions():
  from sklearn.linear_model import LinearRegression

  X = [[6,2],[8,1],[10,0],[14,2],[18,0]]
  y =[[7],[9],[13],[17.5],[18]]  

  model = LinearRegression()
  model.fit(X,y)

  X_test = [[8,2],[9,0],[11,2],[16,2],[12,0]]
  y_test = [[11],[8.5],[15],[18],[11]]

  predictions = model.predict(X_test)

  for i,prediction in enumerate(predictions):
    print "predicted: %s,Target: %s" % (prediction,y_test[i])

  print "R-squared: %.2f" % model.score(X_test,y_test)

# simpleRegressionWithChart()
# multiLinearExample()
# multiLinearExampleWithLeastSQR()
regressionWithPredictions()

