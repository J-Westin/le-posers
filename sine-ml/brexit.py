## Basic machine learning script that trains a neural network
##  to replicate a sine function

from keras.models import Sequential
from keras.layers import Dense

import numpy as np
import matplotlib.pyplot as plt



model = Sequential()

model.add(Dense(units=10, activation='tanh', input_dim=1))

for k in range(10):
	model.add(Dense(units=10, activation='tanh'))

model.add(Dense(units=1, activation='linear'))

model.compile(loss='mean_squared_error', optimizer='sgd')



x_train = 20.*(np.random.rand(10000000)-0.5)
y_train = np.sin(x_train)

model.fit(x_train,y_train, batch_size=100)

x_test = np.linspace(-20,20, 1000)
y_test = model.predict(x_test, batch_size=100)

plt.plot(x_test, np.sin(x_test), "k,")
plt.plot(x_test, y_test, "r-")
plt.show()
