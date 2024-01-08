import tensorflow as tf
import numpy as np

np.set_printoptions(formatter={'float_kind':lambda x: "{0:6.3f}".format(x)})

X = np.array([[2, 3, 4]])
T = np.array([[27, -30, 179]])

W = np.array([[3, 5, 8],
              [4, 6, 9],
              [5, 7, 10]])
B = np.array([1, 2, 3])

model = tf.keras.Sequential([
	tf.keras.layers.Dense(3, input_shape=(3,)),
])

model.layers[0].set_weights([W, B])
  
model.compile(
		optimizer=tf.keras.optimizers.SGD(learning_rate=0.01), # ➐ ➑
		loss=tf.keras.losses.MeanSquaredError()) # ➎
    
for epoch in range(2):

	print('epoch = %d' %epoch)
	
	Y = model.predict(X) # ➊
	print(' Y  =', Y)

	model.fit(X, T, epochs=1)
	print(' W =\n', model.layers[0].get_weights()[0])
	print(' B =', model.layers[0].get_weights()[1])