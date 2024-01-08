import numpy as np
import time
import matplotlib.pyplot as plt

NUM_SAMPLES = 1000

np.random.seed(int(time.time()))

xs = np.random.uniform(0.1, 5, NUM_SAMPLES)
np.random.shuffle(xs)

ys = 1.0/xs

ys += 0.1*np.random.randn(NUM_SAMPLES)

NUM_SPLIT = int(0.8*NUM_SAMPLES)

x_train, x_test = np.split(xs, [NUM_SPLIT])
y_train, y_test = np.split(ys, [NUM_SPLIT])

import tensorflow as tf

model_f = tf.keras.Sequential(
    [
     tf.keras.Input(shape=(1,)),
     tf.keras.layers.Dense(16, activation='relu'),
     tf.keras.layers.Dense(16, activation='relu'),
     tf.keras.layers.Dense(1)
    ]
)

model_f.compile(optimizer='sgd',
                     loss='mean_squared_error')

model_f.fit(x_train, y_train, epochs=600)

p_test = model_f.predict(x_test)

plt.plot(x_test, y_test, 'b.', label='actual')
plt.plot(x_test, p_test, 'r.', label='predicted')
plt.legend()
plt.show()