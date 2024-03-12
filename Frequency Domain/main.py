import scipy.io
import matplotlib.pyplot as plt
import numpy as np

# Load the data
mat = scipy.io.loadmat("D:/Downloads/ae2224I_measurement_data_2024 (1)/ae2224I_measurement_data_subj1_C1.mat")
ft = mat['ft']
u = mat['u']
t = mat['t']

# Compute the FFT of both signals
fft_result_ft = np.fft.fft(ft[:, 0])
fft_result_u = np.fft.fft(u[:, 0])

# Assuming uniform time sampling, compute the frequency axis
dt = t[0, 1] - t[0, 0]
N = len(ft)  # or len(u), assuming both have the same length
frequencies = np.fft.fftfreq(N, dt)

# Keep only positive frequencies and corresponding FFT coefficients
positive_frequencies = frequencies[:N//2]
positive_fft_result_ft = fft_result_ft[:N//2]
positive_fft_result_u = fft_result_u[:N//2]

# Compute the difference in frequencies
difference_in_frequencies = positive_frequencies

# Perform noise cancellation
noise_cancelled_result = np.abs(positive_fft_result_u) - np.abs(positive_fft_result_ft)
ee
# Plot the Fourier transform of the noise-cancelled signal
plt.plot(difference_in_frequencies, noise_cancelled_result)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('Noise-cancelled Fourier Transform')
plt.grid()
plt.xlim(0, 5)
plt.show()