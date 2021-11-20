import warnings
import numpy as np
from sklearn import linear_model
from sklearn.exceptions import ConvergenceWarning

def blackman(a):
  a_0 = .35875
  a_1 = .48829
  a_2 = .14128
  a_3 = .01168
  x=np.maximum(np.minimum(a,1.),0.)
  return a_0 - a_1 * np.cos(2 * np.pi * x) + a_2 * np.cos(4 * np.pi * x) - a_3 * np.cos(6 * np.pi * x)

def gen_basis_cos(t, state_size):
  hertz = np.arange(0, state_size)
  basis = np.cos(hertz * t * 2 * np.pi)
  return basis

def proc_axis(hawk, progress_callback):
  alpha = 1e-3
  state_size = 500
  window_size = .1
  window = []
  new_smpl_dt = 1e-3
  new_data = [[0],[0]]
  samples_per_window = 7

  cc = 0
  for sample in hawk.T:
    time, value = sample[0], sample[1]

    # add new data into window
    basis = gen_basis_cos(time, state_size)
    window.append((time, basis, value))
    
    # drop old data from window
    if len(window) >= 2:
      i = 0
      while window[-1][0] - window[i + 1][0] > window_size + new_smpl_dt:
        i += 1
      window = window[i:]

    # generate new samples
    while new_data[0][-1] + new_smpl_dt < time - window_size / 2:
      new_time = new_data[0][-1] + new_smpl_dt * (samples_per_window // 2 + 1)
      window_begin, window_end = new_time - window_size / 2, new_time + window_size / 2

      # construct a least-squares problem
      sample_weights = np.expand_dims(np.array([blackman((w[0] - window_begin) / window_size) for w in window]), 1)
      equations = sample_weights * np.array([w[1] for w in window], dtype=np.float64)
      targets =  sample_weights * np.array([w[2] for w in window])[np.newaxis].T

      # solve it using RLS
      lasso = linear_model.Lasso(alpha, copy_X = False, tol = 1e-3, max_iter = 100)

      with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        fit = lasso.fit(equations, targets)
      sol = fit.coef_

      for i in range(samples_per_window):
        sample_time =  new_time + (i - samples_per_window // 2) * new_smpl_dt
        new_data[0].append(sample_time)
        new_data[1].append(gen_basis_cos(sample_time, state_size) @ sol)

    if cc % 1024 == 0:
      progress_callback((time - hawk[0][0]) / (hawk[0][-1] - hawk[0][0]) * 100)
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
  fixed_X = proc_axis(hawk_X, progress_cb)
  base_x = 1/3 * 100
  fixed_Y = proc_axis(hawk_Y, progress_cb)
  base_x = 2/3 * 100
  fixed_Z = proc_axis(hawk_Z, progress_cb)

  return np.vstack((fixed_X[0], fixed_X[1], fixed_Y[1], fixed_Z[1]))
