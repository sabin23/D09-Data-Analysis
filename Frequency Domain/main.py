import scipy.io
import matplotlib.pyplot as plt
import numpy as np

# Load the data
mat = scipy.io.loadmat("D:/Downloads/ae2224I_measurement_data_2024 (1)/ae2224I_measurement_data_subj1_C1.mat")
ft= mat['u']
t = mat['t']

# Calculate the FFT
fft_result = np.fft.fft(ft[:, 0])  # Assuming you want to perform FFT on the first column of u
# Calculate the frequency axis
dt = t[0, 1] - t[0, 0]  # Assuming uniform time sampling
N = len(ft)  # Number of samples
frequencies = np.fft.fftfreq(N, dt)

# Keep only positive frequencies and corresponding FFT coefficients
positive_frequencies = frequencies[:N//2]
positive_fft_result = fft_result[:N//2]

# Plot the Fourier transform for positive frequencies
plt.plot(frequencies, np.abs(fft_result))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('Fourier Transform of u (Positive Frequencies)')
plt.grid()
plt.xlim(0, 5)
plt.show()