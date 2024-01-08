import numpy as np
import time
import matplotlib.pyplot as plt

NUM_SAMPLES = 1000

np.random.seed(int(time.time()))

ps = np.random.uniform(-6, 2, NUM_SAMPLES)

es = 0.5*(ps+2)**2

plt.plot(ps, es, 'b.')
plt.show()

p = -5
E = 0.5*(p+2)**2
lr = 0.5

m = 0
v = 0
t = 0
beta_1 = 0.9
beta_2 = 0.999
eps = 10**-8
for i in range(16):
	t = t + 1
	DpE = p+2  
	m = beta_1*m + (1-beta_1)*DpE
	v = beta_2*v + (1-beta_2)*DpE*DpE
	M = m/(1-beta_1**t)
	V = v/(1-beta_2**t)
	p = p - lr*M/(np.sqrt(V)+eps)
	print('p :', p)