import matplotlib.pyplot as plt 
from sklearn.linear_model import LiearRegression

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

model = LiearRegression()
model.fit(x,y)
print "Predcition of price of 34 diameteer is %.2f" % model.predict([29])[0]

