import numpy as np
import matplotlib.pyplot as plt


samples = []
for i in range(500000):
    s = np.abs(np.random.randn()) * 0.005
    samples.append(s)


_ = plt.hist(samples, bins='auto')  # arguments are passed to np.histogram
plt.show()