import numpy as np

import matplotlib.pyplot as plt

from scipy.fftpack import fft

import scipy as sp

import scipy.io as sio

mat = sio.loadmat("C:/Users/jelme/Downloads/ae2224I_measurement_data_2024/ae2224I_measurement_data_subj1_C1.mat")
u = mat['u']
t = mat['t']
print(u)
fig, ax = plt.subplots()
ax.plot(t[0], u[:,0])

ax.set(xlabel='time (s)', ylabel='u',
       title='u-t')  
ax.grid()
plt.show()

# # Load the data
# mat = sio.loadmat("C:/Users/jelme/Downloads/ae2224I_measurement_data_2024/ae2224I_measurement_data_subj1_C1.mat")
# ft= mat['ft']
# t = mat['t']

# # Calculate the FFT
# fft_result = np.fft.fft(ft[:, 0])  # Assuming you want to perform FFT on the first column of u

# # Calculate the frequency axis
# dt = t[0, 1] - t[0, 0]  # Assuming uniform time sampling
# N = len(ft)  # Number of samples
# frequencies = np.fft.fftfreq(N, dt)

# # Keep only positive frequencies and corresponding FFT coefficients
# positive_frequencies = frequencies[:N//2]
# positive_fft_result = fft_result[:N//2]

# # Plot the Fourier transform for positive frequencies
# plt.plot(positive_frequencies, np.abs(positive_fft_result))
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Magnitude')
# plt.title('Fourier Transform of u (Positive Frequencies)')
# plt.grid()
# plt.show()