
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

def polynomialRegression():
  import numpy as np
  import matplotlib.pyplot as plt 
  from sklearn.linear_model import LinearRegression
  from sklearn.preprocessing import PolynomialFeatures

  X_train = [[6],[8],[10],[14],[18]]
  y_train = [[7],[9],[13],[17.5],[18]]
  X_test = [[6],[8],[11],[16]]
  y_test = [[8],[12],[15],[18]]

  regressor = LinearRegression()
  regressor.fit(X_train,y_train)

  xx = np.linspace(0,26,100)
  yy = regressor.predict(xx.reshape(xx.shape[0],1))
  plt.plot(xx,yy)

  quadratic_featurizer = PolynomialFeatures(degree=2)
  X_train_quadratic = quadratic_featurizer.fit_transform(X_train)
  X_test_quadratic = quadratic_featurizer.transform(X_test)

  regressor_quadratic = LinearRegression()
  regressor_quadratic.fit(X_train_quadratic,y_train)

  xx_quadratic = quadratic_featurizer.transform(xx.reshape(xx.shape[0],1))

  plt.plot(xx,regressor_quadratic.predict(xx_quadratic),c='r',linestyle='--')
  plt.title("pizza on diameter")
  plt.xlabel("pizza in inch")
  plt.ylabel("px in usd")
  plt.axis([0,25,0,25])
  plt.grid(True)
  plt.scatter(X_train,y_train)
  plt.show()

  print X_train
  print X_train_quadratic
  print X_test
  print X_test_quadratic
  print "simple reg r-squared", regressor.score(X_test,y_test)
  print "Quadratic regression r-squared", regressor_quadratic.score(X_test_quadratic, y_test)

# using Stochastic gradient decent
def SGDDemo():
  import numpy as np
  from sklearn.datasets import load_boston
  from sklearn.linear_model import SGDRegressor
  from sklearn.cross_validation import cross_val_score
  from sklearn.preprocessing import StandardScaler
  from sklearn.cross_validation import train_test_split

  data = load_boston()
  X_train,X_test,y_train,y_test = train_test_split(data.data,data.target)

  X_scaler = StandardScaler()
  y_scaler = StandardScaler()
  X_train = X_scaler.fit_transform(X_train)
  y_train = y_scaler.fit_transform(y_train)
  X_test = X_scaler.transform(X_test)
  y_test = y_scaler.transform(y_test)

  regressor = SGDRegressor(loss='squared_loss')
  scores = cross_val_score(regressor,X_train,y_train,cv=5)
  print "Cross validation r-sqr ",np.mean(scores)
  regressor.fit_transform(X_train,y_train)
  print "TEST score :",regressor.score(X_test,y_test)

# simpleRegressionWithChart()
# multiLinearExample()
# multiLinearExampleWithLeastSQR()
# regressionWithPredictions()
# polynomialRegression()
SGDDemo()
