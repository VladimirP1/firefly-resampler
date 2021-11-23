import itertools
import numpy as np
from scipy.optimize import least_squares
from concurrent.futures import ThreadPoolExecutor

def motion_model(timestamps, tvels, init=None):
  def m(accs):
    dts = timestamps[1:] - timestamps[:-1]
    vels = np.zeros_like(tvels)
    vels[0] =  tvels[0] if init is None else init
    vels[1:] = np.cumsum(accs * dts) + vels[0]
    return vels

  def evaluate(accs):
    return m(accs) - tvels

  def predict(accs, tss):
    return np.interp(tss, timestamps, m(accs))

  return (evaluate, predict)

def proc_axis(hawk):
  # cleanup
  clusters = [[0]]
  critical_dt = 3e-3
  min_cluster_length = 16
  for i in range(1,len(hawk[0])):
    if hawk[0][i] - hawk[0][i-1] > critical_dt:
      clusters.append([])
    clusters[-1].append(i)
  total_clusters = len(clusters)
  clusters = list(filter(lambda x: len(x) > min_cluster_length, clusters))
  good_samples = list(itertools.chain.from_iterable(clusters))

  print('Clusters discarded ', int(100 - len(clusters) / total_clusters * 100), '%')
  print('Samples discarded ', int(100 - len(good_samples) / len(hawk[0]) * 100), '%')

  hawk = hawk[:, good_samples]

  window_size = 100
  window = []
  new_smpl_dt = 1e-3
  new_data = [[0],[0]]
  max_force = 3

  next_init = None

  cc = 0
  for sample in hawk.T:
    time, value = sample[0], sample[1]

    if len(window) < window_size: 
      window.append((time, value))
      continue

    # generate new samples
    state = np.zeros(len(window) - 1)
    window_snapshot = np.array(window, dtype=np.float64)
    residuals, predict = motion_model(window_snapshot[:,0], window_snapshot[:,1], next_init)

    result = least_squares(residuals, state, bounds = (-max_force, max_force),  method='trf', ftol=1e-2, loss='soft_l1')
    accs = result.x

    calc_end = (window[-1][0]*3 + window[0][0]) / 4
    new_smpl_times = np.arange(new_data[0][-1], calc_end, new_smpl_dt)
    new_smpl_vals = predict(accs, new_smpl_times)

    new_data[0].extend(new_smpl_times)
    new_data[1].extend(new_smpl_vals)
  
    # drop samples from window
    drop_max = np.searchsorted(window_snapshot[:,0], calc_end) - 1
    window = window[drop_max:]

    next_init = predict(accs, [window[0][0]])[0]

    if cc % 100 == 0:
      print((time - hawk[0][0]) / (hawk[0][-1] - hawk[0][0]) * 100)
    cc += 1
  return np.array(new_data, dtype = np.float64)


def proc_all(hawk_data):
  old_x = 0
  base_x = 0
  def progress_cb(x):
    nonlocal old_x
    nonlocal base_x
    new_x = x / 3 + base_x
    if new_x - old_x > 1:
      print(int (new_x))
      old_x = new_x

  hawk_X = hawk_data[[0,1],:]
  hawk_Y = hawk_data[[0,2],:]
  hawk_Z = hawk_data[[0,3],:]

  with ThreadPoolExecutor(max_workers=3) as executor:
      future_X = executor.submit(proc_axis, hawk_X)
      future_Y = executor.submit(proc_axis, hawk_Y)
      future_Z = executor.submit(proc_axis, hawk_Z)

      fixed_X = future_X.result()
      fixed_Y = future_Y.result()
      fixed_Z = future_Z.result()

  return np.vstack((fixed_X[0], fixed_X[1], fixed_Y[1], fixed_Z[1]))
