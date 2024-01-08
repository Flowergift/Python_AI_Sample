import tensorflow as tf
import numpy as np

np.set_printoptions(formatter={'float_kind':lambda x: "{0:6.3f}".format(x)})

I = np.array([[.05, .10]])
T = np.array([[.01, .99]])

WH = np.array([[.15, .25],
		[.20, .30]])
BH = np.array([.35, .35])
WO = np.array([[.40, .50],
		[.45, .55]])
BO = np.array([.60, .60])

model = tf.keras.Sequential([
	tf.keras.layers.Dense(2, input_shape=(2,), activation='relu'),
	tf.keras.layers.Dense(2, activation='relu')
])

model.layers[0].set_weights([WH, BH])
model.layers[1].set_weights([WO, BO])
  
model.compile(
		optimizer=tf.keras.optimizers.SGD(learning_rate=0.01), 
		loss=tf.keras.losses.MeanSquaredError())
    
for epoch in range(2):

	print('epoch = %d' %epoch)
	
	O = model.predict(I)
	print(' O  =', O)

	model.fit(I, T, epochs=1)
	print(' WH =\n', model.layers[0].get_weights()[0])
	print(' BH =\n', model.layers[0].get_weights()[1])
	print(' WO =\n', model.layers[1].get_weights()[0])
	print(' BO =\n', model.layers[1].get_weights()[1])