import numpy as np
import matplotlib.pyplot as plt

He = np.random.randn(1000000)/np.sqrt(10000/2)
Le = np.random.randn(1000000)/np.sqrt(10000)

plt.hist(He, bins=100, density=True, alpha=0.7)
plt.hist(Le, bins=100, density=True, alpha=0.5)
plt.show()