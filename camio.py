import numpy as np
import pandas as pd

def read_hawk(path):
  data = pd.read_csv(path)
  gyroscale = np.pi/180
  ts,xs,ys,zs = data.values[:,[1,2,3,4]].T
  return np.vstack(((ts - ts[0]) / 1e6, ys*gyroscale, zs*gyroscale, xs*gyroscale)).astype(np.float64)

def read_rc(path):
  data = pd.read_csv(path)
  gyroscale = 500 / 2**15 * np.pi/180
  ts,xs,ys,zs = data.values[:,[0,1,2,3]].T
  return np.vstack((ts/1e3, zs*gyroscale, xs*gyroscale, -ys*gyroscale)).astype(np.float64)

def write_hawk(path, data):
    new_data = np.hstack((
      np.arange(len(data[0]))[np.newaxis].T, \
      (np.array(data[0])[np.newaxis].T + 10) * 1e6, \
      np.array(data[3])[np.newaxis].T * 180 / np.pi, \
      np.array(data[1])[np.newaxis].T * 180 / np.pi, \
      np.array(data[2])[np.newaxis].T * 180 / np.pi \
    ))
    new_df = pd.DataFrame(new_data, columns=['loopIteration','time','gyroADC[0]','gyroADC[1]','gyroADC[2]'])
    new_df.to_csv(path, index=False)