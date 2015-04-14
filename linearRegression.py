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

multiLinearExample()
