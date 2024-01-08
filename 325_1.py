import numpy as np
import random
import time
import tensorflow as tf

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.reshape((60000, 1, 784))
y_train = np.array(tf.one_hot(y_train, depth=10))
y_train = y_train.reshape((60000, 1, 10))

np.set_printoptions(formatter={'float_kind':lambda x: "{0:6.4f}".format(x)})

NUM_PATTERN = 60000
NUM_IN = 784
NUM_HID = 64
NUM_OUT = 10

I = x_train
T = y_train
O = np.zeros((NUM_PATTERN, 1, NUM_OUT))
WH = np.random.randn(NUM_IN, NUM_HID)/np.sqrt(NUM_IN/2) # He
BH = np.zeros((1, NUM_HID))
WO = np.random.randn(NUM_HID, NUM_OUT)/np.sqrt(NUM_HID) # Lecun
BO = np.zeros((1, NUM_OUT))

shuffled_pattern = [pc for pc in range(NUM_PATTERN)] #정수로!

random.seed(int(time.time()))

begin = time.time()

for epoch in range(1, 4):
	
	tmp_a = 0;
	tmp_b = 0;
	for pc in range(NUM_PATTERN) :
		tmp_a = random.randrange(0,NUM_PATTERN)
		tmp_b = shuffled_pattern[pc]
		shuffled_pattern[pc] = shuffled_pattern[tmp_a]
		shuffled_pattern[tmp_a] = tmp_b
		
	sumError = 0.

	hit, miss = 0, 0
	
	for rc in range(NUM_PATTERN) : 
	
		pc = shuffled_pattern[rc]
 	
		H = I[pc] @ WH + BH
		H = (H>0)*H # ReLU
		
		O[pc] = H @ WO + BO
		O[pc] = 1/(1+np.exp(-O[pc])) #sigmoid
		if np.argmax(O[pc][0])==np.argmax(T[pc][0]) :
			hit+=1
		else :
			miss+=1
		
		E = np.sum((O[pc]-T[pc])**2/2) #mean squared error
		
		sumError += E
			
		Ob = O[pc] - T[pc]
		Ob = Ob*O[pc]*(1-O[pc]) #sigmoid
		
		Hb = Ob @ WO.T
		Hb = Hb*(H>0)*1 # ReLU
		
		WHb = I[pc].T @ Hb
		BHb = 1 * Hb
		WOb = H.T @ Ob
		BOb = 1 * Ob

		lr = 0.01	
		WH = WH - lr * WHb
		BH = BH - lr * BHb
		WO = WO - lr * WOb
		BO = BO - lr * BOb

		if rc%10000==9999 :
			print("epoch: %2d rc: %6d " %(epoch, rc+1), end='')
			print("hit: %6d miss: %6d " %(hit, miss), end='')
			print("loss: %f accuracy: %f" \
			%(sumError/10000, hit/(hit+miss)))
			sumError = 0

end = time.time()

time_taken = end - begin

print("\nTime taken (in seconds) = {}".format(time_taken))