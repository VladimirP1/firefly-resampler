import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model


csv = pd.read_csv(sys.argv[1])
data = csv.values[:,1] / 1e3

W,H = 1000, 800

img = np.zeros((H,W)) + 0
for ti in range(int(data[0]), int(data[-1])):
    img[ti // W, ti % W] = 1

for t in data:
    ti = int(t)
    img[ti // W, ti % W] = 0


state_size = 12
hole_width = 30
hole_offset = -10
bighole_tresh = 4
hole_mid = np.argmax(np.mean(img, axis=0)) + hole_offset
hole_start = hole_mid - hole_width // 2
hole_end = hole_mid + hole_width // 2

shifts = np.zeros(img.shape[0]) - 1
for i in range(img.shape[0]):
    for j in range(hole_start, hole_end - bighole_tresh):
        if np.all(img[i,j: j + bighole_tresh]):
            shifts[i] = j
            img[i, j] = .5
            break

def gen_basis_cos(t, state_size):
  hertz = np.arange(0, state_size,.2)
  basis = np.cos(hertz * t * 2 * np.pi)
  return basis

lasso = linear_model.Lasso(1e-5, tol = 1e-4, max_iter = 9000, fit_intercept = False)

equations = []
targets = []
s_start = np.nonzero(shifts > .5)[0][0]
s_end = np.nonzero(shifts > .5)[0][-1]
for i,s in enumerate(shifts):
    t = (i - s_start) / (s_end - s_start)
    if s > 0:
        equations.append(gen_basis_cos(t, state_size))
        targets.append(s)
        print(t, s)

fit = lasso.fit(equations, targets)
sol = fit.coef_

tss = np.array(range(s_start, s_end), dtype=np.float64)
fitted = np.array([gen_basis_cos((i - s_start) / (s_end - s_start), state_size) for i in tss]) @ sol

def calc_shift(x):
    (np.array([gen_basis_cos((i - s_start) / (s_end - s_start), state_size) for i in tss]) @ sol) / 1000


# plt.figure(figsize=(10,10))
# plt.imshow(img, cmap='Greys')
# plt.show()

plt.imshow(img[:,hole_start:hole_end])#, cmap='Greys')

plt.subplot()
print(np.max(fitted), np.min(fitted))

plt.plot(range(0,H),shifts)
plt.plot(tss, fitted)
plt.show()
