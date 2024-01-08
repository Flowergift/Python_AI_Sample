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
	
	H = I @ WH + BH # ➊
	O = H @ WO + BO # ➋
	print(' O  =', O)
	
	E = np.sum((O - T) ** 2) / O.shape[1] # ➓

	Ob = 2 * (O - T) / O.shape[1]
	Hb = Ob@WO.T # ➍
	
	WOb = H.T @ Ob # ➎
	BOb = 1 * Ob # ➏
	WHb = I.T @ Hb # ➐
	BHb = 1 * Hb # ➑
	
	lr = 0.01
	
	WO = WO - lr * WOb # ⓫
	BO = BO - lr * BOb # ⓬
	WH = WH - lr * WHb # ⓭
	BH = BH - lr * BHb # ⓮
	print(' WH =\n', WH)
	print(' BH =\n', BH)
	print(' WO =\n', WO)
	print(' BO =\n', BO)