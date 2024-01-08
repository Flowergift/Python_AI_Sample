import numpy as np

np.set_printoptions(formatter={'float_kind':lambda x: "{0:6.4f}".format(x)})

I = np.array([[.05, .10]])
T = np.array([[.01, .99]])
WH = np.random.randn(2, 2)/np.sqrt(2/2) # He
BH = np.zeros((1, 2))
WO = np.random.randn(2, 2)/np.sqrt(2) # Lecun
BO = np.zeros((1, 2))

print("WH =\n", WH)
print("WO =\n", WO)
print()

for epoch in range(1, 1000001):
	
	H = I @ WH + BH
	H = (H>0)*H # ReLU

	O = H @ WO + BO
	O = 1/(1+np.exp(-O)) #sigmoid

	E = np.sum((O-T)**2/2) #mean squared error

	if epoch==1 :
		print("epoch  = %d" %epoch)
		print("Error  = %.4f" %E)
		print("output =", O)
		print()

	if E<0.0001 :
		print("epoch  = %d" %epoch)
		print("Error  = %.4f" %E)
		print("output =", O)
		break
		
	Ob = O - T
	Ob = Ob*O*(1-O) #sigmoid

	Hb = Ob @ WO.T
	Hb = Hb*(H>0)*1 # ReLU

	WHb = I.T @ Hb
	BHb = 1 * Hb
	WOb = H.T @ Ob
	BOb = 1 * Ob

	lr = 0.01	
	WH = WH - lr * WHb
	BH = BH - lr * BHb
	WO = WO - lr * WOb
	BO = BO - lr * BOb