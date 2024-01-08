import numpy as np
import random
import time

np.set_printoptions(formatter={'float_kind':lambda x: "{0:6.4f}".format(x)})

NUM_PATTERN = 10
NUM_IN = 7
NUM_HID = 8
NUM_OUT = 4

I = np.array([
	[[ 1, 1, 1, 1, 1, 1, 0 ]],  # 0
	[[ 0, 1, 1, 0, 0, 0, 0 ]],  # 1
	[[ 1, 1, 0, 1, 1, 0, 1 ]],  # 2
	[[ 1, 1, 1, 1, 0, 0, 1 ]],  # 3
	[[ 0, 1, 1, 0, 0, 1, 1 ]],  # 4
	[[ 1, 0, 1, 1, 0, 1, 1 ]],  # 5
	[[ 0, 0, 1, 1, 1, 1, 1 ]],  # 6
	[[ 1, 1, 1, 0, 0, 0, 0 ]],  # 7
	[[ 1, 1, 1, 1, 1, 1, 1 ]],  # 8
	[[ 1, 1, 1, 0, 0, 1, 1 ]]   # 9
])
T = np.array([
	[[ 0, 0, 0, 0 ]],
	[[ 0, 0, 0, 1 ]],
	[[ 0, 0, 1, 0 ]],
	[[ 0, 0, 1, 1 ]],
	[[ 0, 1, 0, 0 ]],
	[[ 0, 1, 0, 1 ]],
	[[ 0, 1, 1, 0 ]],
	[[ 0, 1, 1, 1 ]],
	[[ 1, 0, 0, 0 ]],
	[[ 1, 0, 0, 1 ]]
])
O = np.zeros((NUM_PATTERN, 1, NUM_OUT))
WH = np.random.randn(NUM_IN, NUM_HID)/np.sqrt(NUM_IN/2) # He
BH = np.zeros((1, NUM_HID))
WO = np.random.randn(NUM_HID, NUM_OUT)/np.sqrt(NUM_HID) # Lecun
BO = np.zeros((1, NUM_OUT))

shuffled_pattern = [pc for pc in range(NUM_PATTERN)] #정수로!

random.seed(int(time.time()))

for epoch in range(1, 10001):
	
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
				
	if epoch%100==0 :	
		print("epoch : %5d, sum error : %f" %(epoch, sumError))
		for i in range(NUM_IN) :
			for j in range(NUM_HID) :
				print("%7.3f "%WH[i][j], end='')
			print(flush=True)
			
	if sumError<0.0001 : break
	
print()

for pc in range(NUM_PATTERN) :
	print("target %d : "%pc, end='')
	for node in range(NUM_OUT) :
		print("%.0f "%T[pc][0][node], end='')
	print("pattern %d : "%pc, end='');
	for node in range(NUM_OUT) :
		print("%.2f "%O[pc][0][node], end='')
	print()