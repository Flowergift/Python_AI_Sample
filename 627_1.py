import numpy as np
from my_cnn3D import *
from my_cnn2D import *
import tensorflow as tf
import time #
import random #

mnist = tf.keras.datasets.fashion_mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.reshape((60000, 28, 28, 1))
y_train = np.array(tf.one_hot(y_train, depth=10))
x_test = x_test.reshape((10000, 28, 28, 1))

np.set_printoptions(formatter={'float_kind':lambda x: "{0:6.3f}".format(x)})

IMG_SIZE = 28
HALF_SIZE = int(IMG_SIZE/2)

NUM_PATTERN = 60000 #
I_DEPTH = 1
C_DEPTH = 32
K_SIZE = 3
P_SIZE = 2
NUM_OUT = 10

I = x_train 
WC = np.random.randn(K_SIZE,K_SIZE,I_DEPTH,C_DEPTH)
WC /= np.sqrt(K_SIZE*K_SIZE*I_DEPTH/2) # He
BC = np.zeros((C_DEPTH,))

WO = np.random.randn(HALF_SIZE*HALF_SIZE*C_DEPTH,NUM_OUT)
WO /= np.sqrt(HALF_SIZE*HALF_SIZE*C_DEPTH) # Lecun
BO = np.zeros((1, NUM_OUT))

T = y_train

shuffled_pattern = [0 for node in range(NUM_PATTERN)] #

random.seed(int(time.time())) #

for pc in range(NUM_PATTERN) : #
	shuffled_pattern[pc] = pc #

begin = time.time() #

for epoch in range(1, 4): #

	tmp_a = 0; #
	tmp_b = 0; #
	for pc in range(NUM_PATTERN) : #
		tmp_a = random.randrange(0,NUM_PATTERN) #
		tmp_b = shuffled_pattern[pc] #
		shuffled_pattern[pc] = shuffled_pattern[tmp_a] #
		shuffled_pattern[tmp_a] = tmp_b #

	sumError = 0. #

	hit, miss = 0, 0 #
  
	t_prev = time.time() #

	for rc in range(NUM_PATTERN) : #

		n = shuffled_pattern[rc] #

		C, cache = conv_2D_f(I[n], WC, BC)
		C = (C>0)*C # relu

		P = max_pool_3D_f(C)

		(p_size, _, p_depth) = P.shape
		F = P.reshape((1,p_size*p_size*p_depth))

		O = F@WO + BO
		OM = O - np.max(O)
		O = np.exp(OM)/np.sum(np.exp(OM)) # softmax
		if np.argmax(O)==np.argmax(T[n]) : #
			hit+=1 #
		else : #
			miss+=1 #

		E = np.sum(-T[n]*np.log(O))

		sumError += E #

		Ob = O - T[n]
		# nothing for softmax

		WOb = F.T@Ob
		BOb = Ob

		Fb = Ob@WO.T

		Pb = Fb.reshape(P.shape)

		Cb = max_pool_3D_b(Pb, C)
		Cb = Cb*(C>0)*1 # relu

		WCb, BCb, Ib = conv_2D_b(Cb, I[n], WC, BC, cache, input_back=False)

		lr = 0.01
		WO -= lr*WOb
		BO -= lr*BOb
		WC -= lr*WCb
		BC -= lr*BCb
    
		if rc%100 == 99 : #
			print("epoch: %2d rc: %6d " %(epoch, rc+1), end='') #
			print("hit: %6d miss: %6d " %(hit, miss), end='') #
			print("loss: %f accuracy: %f" %(sumError/100, hit/(hit+miss)), end='') #
			sumError = 0 #
			
			t_now = time.time() #
			time_taken = t_now - t_prev #
			t_prev = t_now #
			print(" {:10.3f}sec".format(time_taken)) #
      
end = time.time()
time_taken = end - begin
print("\nTotal time taken (in seconds) = {}".format(time_taken))