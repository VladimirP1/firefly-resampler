import camio
import resampler
import sys
import numpy as np


hawk_data = camio.read_hawk(sys.argv[1])
mean = np.mean(hawk_data[1:4,:], axis=1, keepdims=True)
hawk_data[1:4,:] -= mean
print(mean)
camio.write_hawk(sys.argv[2], hawk_data)
