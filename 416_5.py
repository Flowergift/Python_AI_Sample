import numpy as np

np.set_printoptions(formatter={'float_kind':lambda x: "{0:6.3f}".format(x)})
 
I = np.array([[.05, .10]])
T = np.array([[.01, .99]])
WH = np.array([[.15, .25],
		[.20, .30]])
BH = np.array([[.35, .35]])
WO = np.array([[.40, .50],
		[.45, .55]])
BO = np.array([[.60, .60]])

for epoch in range(2):

	print('epoch = %d' %epoch)
	
	H = I @ WH + BH
	H = (H>0)*H #
	
	O = H @ WO + BO
	O = (O>0)*O #
	
	print(' O  =', O)
	
	E = np.sum((O - T) ** 2) / O.shape[1]

	Ob = 2 * (O - T) / O.shape[1]
	Ob = Ob*(O>0)*1 #
	
	Hb = Ob@WO.T
	Hb = Hb*(H>0)*1 #
	
	lr = 0.01
	
	WO = WO - lr * WOb # ⓫
	BO = BO - lr * BOb # ⓬
	WH = WH - lr * WHb # ⓭
	BH = BH - lr * BHb # ⓮
	print(' WH =\n', WH)
	print(' BH =\n', BH)
	print(' WO =\n', WO)
	print(' BO =\n', BO)