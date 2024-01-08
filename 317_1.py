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

for epoch in range(1000):

	print('epoch = %d' %epoch)

	H = I @ WH + BH
	H = 1/(1+np.exp(-H)) # ➊

	O = H @ WO + BO
	O = 1/(1+np.exp(-O)) # ➊

	print(' O  =\n', O)

	E = np.sum((O - T) ** 2 / 2)
	if E < 0.0000001:
		break

	Ob = O - T
	Ob = Ob*O*(1-O) # ➋

	Hb = Ob @ WO.T
	Hb = Hb*H*(1-H) # ➋

	WHb = I.T @ Hb
	BHb = 1 * Hb
	WOb = H.T @ Ob
	BOb = 1 * Ob

	lr = 0.01
	WH = WH - lr * WHb
	BH = BH - lr * BHb
	WO = WO - lr * WOb
	BO = BO - lr * BOb