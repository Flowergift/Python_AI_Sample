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
	
	H = I @ WH + BH # ➊
	O = H @ WO + BO # ➋
	print(' O  =\n', O)
	
	E = np.sum((O - T) ** 2 / 2)
	print(' E  = %.7f' %E)
	if E < 0.0000001:
		break

	Ob = O - T
	Hb = Ob @ WO.T # ➍
	WHb = I.T @ Hb # ➐
	BHb = 1 * Hb # ➑
	WOb = H.T @ Ob # ➎
	BOb = 1 * Ob # ➏
	print(' WHb =\n', WHb)
	print(' BHb =\n', BHb)
	print(' WOb =\n', WOb)
	print(' BOb =\n', BOb)

	lr = 0.01 	
	WH = WH - lr * WHb # ⓫
	BH = BH - lr * BHb # ⓬
	WO = WO - lr * WOb # ➒
	BO = BO - lr * BOb # ➓
	print(' WH  =\n', WH)
	print(' BH  =\n', BH)
	print(' WO  =\n', WO)
	print(' BO  =\n', BO)