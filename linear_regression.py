import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense

X=np.linspace(1,10,100)
Y=2*X+20+np.random.randn(100)
print(X)
print(Y)

X.reshape(-1,1)
print(X.shape)

model=Sequential()
model.add(Dense(1,input_dim=1,activation='linear'))
model.compile(optimizer='sgd',loss='mse')
model.fit(X,Y,epochs=100)

pred=model.predict(X)

plt.scatter(X,Y,label='original data')
plt.plot(X,pred,label='predicted data')
plt.show()

#test