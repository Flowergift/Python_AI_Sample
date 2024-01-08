import numpy as np

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

for epoch in range(1, 1000001):
	
	H = I[2] @ WH + BH
	H = (H>0)*H # ReLU

	O[2] = H @ WO + BO
	O[2] = 1/(1+np.exp(-O[2])) #sigmoid

	E = np.sum((O[2]-T[2])**2/2) #mean squared error

	if epoch==1 :
		print("epoch  = %d" %epoch)
		print("Error  = %.4f" %E)
		print("output =", O[2])
		print()

	if E<0.0001 :
		print("epoch  = %d" %epoch)
		print("Error  = %.4f" %E)
		print("output =", O[2])
		break
		
	Ob = O[2] - T[2]
	Ob = Ob*O[2]*(1-O[2]) #sigmoid

	Hb = Ob @ WO.T
	Hb = Hb*(H>0)*1 # ReLU

	WHb = I[2].T @ Hb
	BHb = 1 * Hb
	WOb = H.T @ Ob
	BOb = 1 * Ob

	lr = 0.01	
	WH = WH - lr * WHb
	BH = BH - lr * BHb
	WO = WO - lr * WOb
	BO = BO - lr * BOb