import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

csv = pd.read_csv(sys.argv[1])
data = csv.values[1000:,1] / 1e6
diffs_ts = np.array([*map(lambda x: (x[0], (x[1]-x[0])), zip(data[:-1], data[1:]))])
diffs = diffs_ts[:,1]

plt.figure(figsize=(12,3))
plt.title("Time elapsed from previous gyro sample")
plt.plot(diffs_ts[:,0], diffs_ts[:,1])
plt.show()