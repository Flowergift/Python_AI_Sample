import tensorflow as tf
import numpy as np

np.set_printoptions(formatter={'float_kind':lambda x: "{0:6.3f}".format(x)})

IMG_SIZE = 4
HALF_SIZE = int(IMG_SIZE/2)

I_DEPTH = 1
C_DEPTH = 2
K_SIZE = 3
NUM_OUT = 3

np.random.seed(1)

I = np.random.rand(1,IMG_SIZE,IMG_SIZE,I_DEPTH)
WC = np.random.rand(K_SIZE,K_SIZE,I_DEPTH,C_DEPTH)
BC = np.random.rand(C_DEPTH,)

WO = np.random.rand(HALF_SIZE*HALF_SIZE*C_DEPTH,NUM_OUT)
BO = np.random.rand(NUM_OUT,)

T = np.zeros((1, NUM_OUT))

model = tf.keras.Sequential([
	tf.keras.layers.InputLayer(input_shape=(IMG_SIZE, IMG_SIZE, I_DEPTH)),
	tf.keras.layers.Conv2D(C_DEPTH, (K_SIZE, K_SIZE), padding='same'),
	tf.keras.layers.MaxPooling2D((2, 2)),
])

model.layers[0].set_weights([WC, BC])
  
optimizer=tf.keras.optimizers.SGD(learning_rate=0.01)
loss=tf.keras.losses.MeanSquaredError()
    
for epoch in range(1):
	
	with tf.GradientTape() as tape:
	
		O = model(I)
		print('O_0  =\n', O.numpy()[0,:,:,0])
		print('O_1  =\n', O.numpy()[0,:,:,1])
		print('O.shape =', O.shape)