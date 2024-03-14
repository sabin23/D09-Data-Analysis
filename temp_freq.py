import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from get_data import Data

# Load the data

mat = []
for i in range(1, 7):
    line = []
    for j in range(1, 7):
        line.append(Data(i, j))
    mat.append(line)


def plot_Bode():
    # Create a figure with two subplots
    fig, axs = plt.subplots(2)

    # Plot the magnitude of Hpe_FC in the first subplot
    axs[0].semilogx(w_FC, magnitude_Hpe)
    axs[0].set_title('Magnitude of Hpe_FC')
    axs[0].set_xlabel('Frequency (Hz)')
    axs[0].set_ylabel('Magnitude')
    axs[0].grid()

    # Plot the phase of Hpe_FC in the second subplot
    axs[1].semilogx(w_FC, phase_Hpe)
    axs[1].set_title('Phase of Hpe_FC')
    axs[1].set_xlabel('Frequency (Hz)')
    axs[1].set_ylabel('Phase (degrees)')
    axs[1].grid()

    plt.tight_layout()
    plt.show()

def plot_FFT():
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

    # Create a figure with 3 subplots
    fig, axs = plt.subplots(3)

    # Plot the Fourier transform of the original signal in the first subplot
    axs[0].plot(positive_frequencies, np.abs(positive_fft_result_u))
    axs[0].set_title('Original Fourier Transform')
    axs[0].set_xlabel('Frequency (Hz)')
    axs[0].set_ylabel('Magnitude')
    axs[0].grid()
    axs[0].set_xlim(0, 5)

    # Plot the Fourier transform of the noise in the second subplot
    axs[1].semilogy(positive_frequencies, np.abs(positive_fft_result_ft))
    axs[1].set_title('Noise Fourier Transform')
    axs[1].set_xlabel('Frequency (Hz)')
    axs[1].set_ylabel('Magnitude (log scale)')
    axs[1].grid()
    axs[1].set_xlim(0, 5)

    # Plot the Fourier transform of the noise-cancelled signal in the third subplot
    axs[2].plot(difference_in_frequencies, np.abs(noise_cancelled_result))
    axs[2].set_xlim(0, 5)
    plt.xlim(0, 5)
    plt.tight_layout()
    plt.show()


a, b = 1, 1
# Load a condition
ft = mat[a][b].ft
u = mat[a][b].u
t = mat[a][b].t

# Get magnitude and phase of numbers in Hpe_FC
Hpe_FC = mat[a][b].Hpe_FC
magnitude_Hpe = np.abs(Hpe_FC)
magnitude_Hpe = 10 * np.log(magnitude_Hpe)
phase_Hpe = np.angle(Hpe_FC, deg=True)

for i in range(len(phase_Hpe)):
    if phase_Hpe[i] >= 0:
        phase_Hpe[i] -= 360 
    if i != len(phase_Hpe)-1:
        if phase_Hpe[i] <= -181 and phase_Hpe[i+1] >= -181:
            phase_Hpe[i+1] -= 360
        
print(phase_Hpe)
w_FC = mat[a][b].w_FC

frequencies = np.linspace(0, 100, 100)
#print(frequencies)


plot_Bode()