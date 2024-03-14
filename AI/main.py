import numpy as np

import matplotlib.pyplot as plt

from scipy.fftpack import fft

import scipy as sp

import scipy.io as sio


mat = sio.loadmat("C:/Users/jelme/Downloads/ae2224I_measurement_data_2024/ae2224I_measurement_data_subj1_C1.mat")
hpe = mat['Hpe_FC']

hper = np.real(hpe)
hpec = np.imag(hpe)

mag = np.sqrt(np.square(hper)+np.square(hpec))
phase = np.arctan(hpec/hper)

freq = mat['w_FC']

print(freq)

# Plot magnitude response
plt.figure()
plt.semilogx(freq, mag)  # Bode magnitude plot
plt.title('Bode Plot - Magnitude Response')
plt.xlabel('Frequency [rad/s]')
plt.ylabel('Magnitude [dB]')
plt.grid(True)

# Plot phase response
plt.figure()
plt.semilogx(freq, phase)  # Bode phase plot
plt.title('Bode Plot - Phase Response')
plt.xlabel('Frequency [rad/s]')
plt.ylabel('Phase [degrees]')
plt.grid(True)

plt.show()