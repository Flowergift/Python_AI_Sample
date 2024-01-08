import numpy as np
import random
import time

np.set_printoptions(formatter={'float_kind':lambda x: "{0:6.4f}".format(x)})

NUM_PATTERN = 60000
NUM_IN = 784
NUM_HID = 64
NUM_OUT = 10

I = np.random.randn(NUM_PATTERN, 1, NUM_IN)
T = np.random.randn(NUM_PATTERN, 1, NUM_OUT)
O = np.zeros((NUM_PATTERN, 1, NUM_OUT))
WH = np.random.randn(NUM_IN, NUM_HID)/np.sqrt(NUM_IN/2) # He
BH = np.zeros((1, NUM_HID))
WO = np.random.randn(NUM_HID, NUM_OUT)/np.sqrt(NUM_HID) # Lecun
BO = np.zeros((1, NUM_OUT))

shuffled_pattern = [pc for pc in range(NUM_PATTERN)] #정수로!

random.seed(int(time.time()))

begin = time.time()

for epoch in range(1, 2):
	
	tmp_a = 0;
	tmp_b = 0;
	for pc in range(NUM_PATTERN) :
		tmp_a = random.randrange(0,NUM_PATTERN)
		tmp_b = shuffled_pattern[pc]
		shuffled_pattern[pc] = shuffled_pattern[tmp_a]
		shuffled_pattern[tmp_a] = tmp_b
		
	sumError = 0.
	
	for rc in range(NUM_PATTERN) : 
	
		pc = shuffled_pattern[rc]
 	
		H = I[pc] @ WH + BH
		H = (H>0)*H # ReLU
		
		O[pc] = H @ WO + BO
		O[pc] = 1/(1+np.exp(-O[pc])) #sigmoid
		
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

		if rc%1000==999 :
			print(".", end='', flush=True)

end = time.time()

time_taken = end - begin

print("\nTime taken (in seconds) = {}".format(time_taken))