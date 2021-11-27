import camio
import resampler
import find_sync
import sys
import matplotlib.pyplot as plt
import numpy as np

hawk_data = camio.read_hawk(sys.argv[1])
hawk_synced = find_sync.fix_sync(hawk_data)
hawk_fixed = resampler.proc_all(hawk_synced)
camio.write_hawk(sys.argv[2], hawk_fixed)

plt.figure(figsize=(20,5))
plt.xlim(130,150)
plt.ylim(-.5,.5)
plt.plot(hawk_fixed[0], hawk_fixed[1], '-')
plt.plot(hawk_synced[0], hawk_synced[1], '-', alpha=.7)
plt.plot(hawk_synced[0], np.zeros((len(hawk_synced[0]),)),'+')
plt.show()

