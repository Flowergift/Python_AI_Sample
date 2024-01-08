import numpy as np
from my_cnn3D import *

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

C = conv_3D_f(I[0], WC, BC)

P = max_pool_3D_f(C)

(p_size, _, p_depth) = P.shape
F = P.reshape((1,p_size*p_size*p_depth))

O = F@WO + BO

E = np.sum((O-T)**2/O.shape[1])

Ob = 2*(O-T)/O.shape[1]

WOb = F.T@Ob
BOb = Ob

Fb = Ob@WO.T

Pb = Fb.reshape(P.shape)

Cb = max_pool_3D_b(Pb, C)

WCb, BCb, Ib = conv_3D_b(Cb, I[0], WC, BC, input_back=False)

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