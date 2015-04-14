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
print "cov of x and y is %.2f" %np.cov(x,y)[0][1]
