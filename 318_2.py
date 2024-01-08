import numpy as np

np.set_printoptions(formatter={'float_kind':lambda x: "{0:6.3f}".format(x)})

I = np.array([[.05, .10]])
T = np.array([[  0,   1]])
WH = np.array([[.15, .25],
		[.20, .30]])
BH = np.array([[.35, .35]])
WO = np.array([[.40, .50],
		[.45, .55]])
BO = np.array([[.60, .60]])

for epoch in range(10000000):

	print('epoch = %d' %epoch)

	H = I @ WH + BH
	H = np.tanh(H)
	
	O = H @ WO + BO
	OM = O - np.max(O)
	O = np.exp(OM)/np.sum(np.exp(OM)) 

	print(' O  =\n', O)

	E = np.sum(-T*np.log(O))
	if E < 0.0001:
		break

	Ob = O - T
	# nothing for softmax + cross entropy error	
	
	Hb = Ob @ WO.T
	Hb = Hb*(1+H)*(1-H) 

	WHb = I.T @ Hb
	BHb = 1 * Hb
	WOb = H.T @ Ob
	BOb = 1 * Ob

	lr = 0.01	
	WH = WH - lr * WHb
	BH = BH - lr * BHb
	WO = WO - lr * WOb
	BO = BO - lr * BOb