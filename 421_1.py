import numpy as np
import time
import matplotlib.pyplot as plt

NUM_SAMPLES = 1000

np.random.seed(int(time.time()))

ps = np.random.uniform(-6, 2, NUM_SAMPLES)

es = 0.5*(ps+2)**2

plt.plot(ps, es, 'b.')
plt.show()