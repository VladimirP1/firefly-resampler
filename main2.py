import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.interpolate as spi
import scipy.signal as ssi
from numpy.lib.stride_tricks import sliding_window_view
import scipy.io.wavfile
import camio
import sys
import find_sync

hawk_raw = camio.read_hawk(sys.argv[1])
hawk_data = find_sync.fix_sync(hawk_raw)

#hawk_data = hawk_data[:,118056:120556]

freq_min = 200
freq_max = 490
freq_step = 1
phase_step = 2 * np.pi / 15
ampl_min = 0
ampl_max = .3
ampl_step = .002
ampl_initial = .01

def dynamic_filter(ts, ds):
    freqs = np.expand_dims(np.arange(freq_min, freq_max, freq_step), (1,2))
    # print(freqs.shape)
    phases =  np.expand_dims(np.arange(0, 2 * np.pi, phase_step),(0,2))
    # print(phases.shape)
    datas = np.expand_dims(ts, (0,1))
    # print(datas.shape)
    sines = np.sin(2 * np.pi * freqs * datas + phases) * ampl_initial
    # print(sines.shape)
    values = np.expand_dims(ds, (0,1))
    # print(values.shape)
    img = np.sum((np.diff(sines) - np.diff(values))**2, axis=2)
    # print(img.shape)

    best = np.unravel_index(np.argmin(img), img.shape)
    best_freq, best_phase = best[0] * freq_step + freq_min, best[1] * phase_step

    # plt.figure(figsize=(20,20))
    # plt.imshow(img.T, origin='lower', extent=(freq_min, freq_max, 0, 2 * np.pi))
    # plt.show()

    # plt.figure(figsize=(20,5))
    # plt.plot(ts], ds)
    # plt.plot(ts, sines[best[0], best[1]])
    # plt.show()

    amplitudes = np.expand_dims(np.arange(ampl_min, ampl_max, ampl_step), 1)
    # print(amplitudes.shape)
    sines = np.sin(2 * np.pi * best_freq * ts + best_phase) * amplitudes
    # print(sines.shape)
    values = np.expand_dims(ds, 0)
    # print(values.shape)
    img = np.sum((np.diff(sines) - np.diff(values))**2, axis=1)
    img.shape

    best = np.argmin(img)
    best_ampl = best * ampl_step + ampl_min

    #print(best_freq, best_phase, best_ampl)

    # plt.figure(figsize=(20,5))
    # plt.plot(ts, ds)
    # plt.plot(ts, sines[best])
    # plt.show()

    # plt.figure(figsize=(20,5))
    # plt.plot(ts, ds - sines[best])
    # plt.show()

    return ds - sines[best]

def dynamic_filter_n(ts, ds, n = 4):
    for i in range(n):
        ds = dynamic_filter(ts, ds)
    return ds

def remove_spikes(ds):
    ds2 = ds.copy()

    percentile = 95
    window_size = 8
    means = np.mean(sliding_window_view(ds, window_size), axis=1)[1:]
    errors =  (means - ds[:-(window_size)])**2
    tresh = np.percentile(errors, percentile)
    indicies = np.nonzero(errors > tresh)
    ds2[indicies] = means[indicies]

    return ds2

chunk_size = 50

for i in range(len(hawk_data[0]) // chunk_size):
    print(100 * i / (len(hawk_data[0]) // chunk_size))
    start_i = chunk_size * i
    end_i = chunk_size * (i + 1)
    hawk_data[1,start_i:end_i] = dynamic_filter_n(hawk_data[0,start_i:end_i], hawk_data[1,start_i:end_i])
    hawk_data[2,start_i:end_i] = dynamic_filter_n(hawk_data[0,start_i:end_i], hawk_data[2,start_i:end_i])
    hawk_data[3,start_i:end_i] = dynamic_filter_n(hawk_data[0,start_i:end_i], hawk_data[3,start_i:end_i])

hawk_data[1] = remove_spikes(hawk_data[1])
hawk_data[2] = remove_spikes(hawk_data[2])
hawk_data[3] = remove_spikes(hawk_data[3])

dt = 1/1000
new_ts = np.arange(hawk_data[0][0], hawk_data[0][-1], dt)
new_x = np.interp(new_ts, hawk_data[0], hawk_data[1])
new_y = np.interp(new_ts, hawk_data[0], hawk_data[2])
new_z = np.interp(new_ts, hawk_data[0], hawk_data[3])

# sos = ssi.butter(1, 1, output='sos', fs=1/dt)
# new_y = ssi.sosfilt(sos, new_y)

new_data = np.vstack((new_ts, new_x, new_y, new_z))

camio.write_hawk(sys.argv[2], new_data)

plt.figure(figsize=(20,5))
plt.plot(hawk_data[0], hawk_data[1], '-')
plt.plot(hawk_data[0,:-9], np.mean(sliding_window_view(hawk_data[1], 10), axis=1), '-')
plt.show()
