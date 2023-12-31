import numpy as np

np.set_printoptions(formatter={'float_kind':lambda x: "{0:6.3f}".format(x)})

IMG_SIZE = 4
HALF_SIZE = int(IMG_SIZE/2)

I_DEPTH = 1
C_DEPTH = 2
K_SIZE = 3
P_SIZE = 2
NUM_OUT = 3

np.random.seed(1)

I = np.random.rand(1,IMG_SIZE,IMG_SIZE,I_DEPTH)
WC = np.random.rand(K_SIZE,K_SIZE,I_DEPTH,C_DEPTH)
BC = np.random.rand(C_DEPTH,)

WO = np.random.rand(HALF_SIZE*HALF_SIZE*C_DEPTH,NUM_OUT)
BO = np.random.rand(1, NUM_OUT)

T = np.zeros((1, NUM_OUT))

C = np.zeros((IMG_SIZE,IMG_SIZE,C_DEPTH))
P = np.zeros((HALF_SIZE,HALF_SIZE,C_DEPTH))

IP = np.pad(I, ((0,0), (1,1), (1,1), (0,0)))

for fn in range(C_DEPTH):
	for row in range(IMG_SIZE):
		for col in range(IMG_SIZE):
			Win3D = IP[0, row:row+K_SIZE, col:col+K_SIZE]
			C[row,col,fn] = np.sum(Win3D*WC[:,:,:,fn])+BC[fn]

for fn in range(C_DEPTH):
	for row in range(0,HALF_SIZE):
		for col in range(0,HALF_SIZE):
			Win3D = C[2*row:2*row+P_SIZE, 2*col:2*col+P_SIZE, fn]
			P[row,col,fn] = np.max(Win3D)

(p_size, _, p_depth) = P.shape
F = P.reshape((1,p_size*p_size*p_depth))

O = F@WO + BO

E = np.sum((O-T)**2)/O.shape[1]

Ob = 2*(O-T)/O.shape[1]

WOb = F.T@Ob
BOb = Ob

Fb = Ob@WO.T

Pb = Fb.reshape(P.shape)

Cb = np.zeros_like(C)

for fn in range(C_DEPTH):
	for row in range(0,HALF_SIZE):
		for col in range(0,HALF_SIZE):
			Win3D = C[2*row:2*row+P_SIZE, 2*col:2*col+P_SIZE, fn]
			Win3Db = Cb[2*row:2*row+P_SIZE, 2*col:2*col+P_SIZE, fn]
			Win3Db[Win3D==np.max(Win3D)] = Pb[row,col,fn]

IPb = np.zeros_like(IP)
WCb = np.zeros_like(WC)
BCb = np.zeros_like(BC)

for fn in range(C_DEPTH):
	for row in range(IMG_SIZE):
		for col in range(IMG_SIZE):
			Win3D = IP[0, row:row+K_SIZE, col:col+K_SIZE]
			WCb[:,:,:,fn] += Win3D*Cb[row,col,fn]

for fn in range(C_DEPTH):
	BCb[fn] = np.sum(Cb[:,:,fn])

for fn in range(C_DEPTH):
	for row in range(IMG_SIZE):
		for col in range(IMG_SIZE):
			Win3Db = IPb[0,row:row+K_SIZE, col:col+K_SIZE]
			Win3Db += Cb[row,col,fn]*WC[:,:,:,fn]			

lr = 0.01
WO -= lr*WOb
BO -= lr*BOb
WC -= lr*WCb
BC -= lr*BCb

print('WO =\n', WO[:,0])
print('BO =\n', BO)
print('WC_0 =\n', WC[:,:,0,0])
print('WC_1 =\n', WC[:,:,0,1])
print('BC =\n', BC)