import tensorflow as tf

mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
print("x_train:%s y_train:%s x_test:%s y_test:%s "%( 
      x_train.shape, y_train.shape, x_test.shape, y_test.shape))

import matplotlib.pyplot as plt

plt.figure()
plt.imshow(x_train[0])
plt.show()

for y in range(28):
  for x in range(28):
    print("%4s"%x_train[0][y][x],end='')
  print()

plt.figure(figsize=(10,10))
for i in range(25):
  plt.subplot(5,5,i+1)
  plt.xticks([])
  plt.yticks([])
  plt.imshow(x_train[i], cmap=plt.cm.binary)
  plt.xlabel(y_train[i])
plt.show()

x_train, x_test = x_train/255.0, x_test/255.0
x_train, x_test = x_train.reshape(60000,784), x_test.reshape(10000,784)

model = tf.keras.models.Sequential([
  tf.keras.Input(shape=(784,)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test, y_test)

p_test = model.predict(x_test)
print('p_test[0] :', p_test[0])

import numpy as np

print('p_test[0] :', np.argmax(p_test[0]), 
      'y_test[0] :',y_test[0])

x_test = x_test.reshape(10000,28,28)

plt.figure()
plt.imshow(x_test[0])
plt.show()

plt.figure(figsize=(10,10))
for i in range(25):
  plt.subplot(5,5,i+1)
  plt.xticks([])
  plt.yticks([])
  plt.imshow(x_test[i], cmap=plt.cm.binary)
  plt.xlabel(np.argmax(p_test[i]))
plt.show()