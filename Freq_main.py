import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from get_data import Data
from scipy.signal import find_peaks
# Load the data
mat = []
for i in range(1, 7):
    line = []
    for j in range(1, 7):
        line.append(Data(i, j))
    mat.append(line)

ft = mat[2][2].ft
u = mat[2][2].u
t = mat[2][2].t

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

def plot(difference_in_frequencies,noise_cancelled_result,positive_fft_result_u, positive_fft_result_ft):
    # Plot 1: FFT of noise-cancelled signal
    plt.figure(figsize=(14, 6))
    
    # Plot 1: FFT of noise-cancelled signal
    plt.subplot(1, 3, 1)
    plt.plot(difference_in_frequencies, np.abs(noise_cancelled_result))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title('Noise-cancelled Fourier Transform')
    plt.xlim(0, 5)
    
    # Plot 2: FFT of u
    plt.subplot(1, 3, 2)
    plt.plot(difference_in_frequencies, np.abs(positive_fft_result_u))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title('FFT of u')
    plt.xlim(0, 5)
    # Plot 3: FFT of ft
    plt.subplot(1, 3, 3)
    plt.plot(difference_in_frequencies, np.abs(positive_fft_result_ft))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title('FFT of ft')
    plt.grid()
    plt.xlim(0, 5)
    
    # Find peaks in the FFT of ft
    peaks, _ = find_peaks(np.abs(positive_fft_result_ft))
    
    # Get corresponding frequencie
    peak_frequencies = difference_in_frequencies[peaks]
    
    # Plot the peaks on the FFT of f
    plt.plot(difference_in_frequencies[peaks], np.abs(positive_fft_result_ft)[peaks], 'ro')  # Mark peaks with red dots
    plt.xlim(0, 5)
    plt.grid()
    plt.show()
    
    print("Frequencies of the peaks in FFT of ft:", peak_frequencies)

plot(difference_in_frequencies,noise_cancelled_result,positive_fft_result_u, positive_fft_result_ft)