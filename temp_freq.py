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
    axs[0].semilogx(w_FC, magnitude_Hpe, label='Hpe')
    axs[0].semilogx(w_FC, magnitude_Hpxd, label='Hpxd')
    axs[0].semilogx(w_FC, (magnitude_Hpe + magnitude_Hpxd) / 2, label='Average')
    axs[0].set_title('Magnitude of Hpe_FC')
    axs[0].set_xlabel('Frequency (Hz)')
    axs[0].set_ylabel('Magnitude')
    axs[0].grid()
    axs[0].legend()

    # Plot the phase of Hpe_FC in the second subplot
    axs[1].semilogx(w_FC, phase_Hpe, label='Hpe')
    axs[1].semilogx(w_FC, phase_Hpxd, label='Hpxd')
    axs[1].semilogx(w_FC, (phase_Hpe + phase_Hpxd) / 2, label='Average')
    axs[1].set_title('Phase of Hpe_FC')
    axs[1].set_xlabel('Frequency (Hz)')
    axs[1].set_ylabel('Phase (degrees)')
    axs[1].grid()
    axs[1].legend()

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

def plot_boxplots():
    magnitudes = []
    phases = []
    for i in range(0, 6):
        for j in range(0, 6):
            hpe = 0

box = np.empty((6,20))

for i in range(6):

    a, b = i, 0
    # Load a condition
    ft = mat[a][b].ft
    u = mat[a][b].u
    t = mat[a][b].t

    # Get magnitude and phase of numbers in Hpe_FC
    Hpe_FC = mat[a][b].Hpe_FC
    hper = np.real(Hpe_FC)
    hpec = np.imag(Hpe_FC)
    magnitude_Hpe = np.sqrt(np.square(hper) + np.square(hpec))
    phase_Hpe = np.angle(Hpe_FC, deg=True)

    Hpxd_FC = mat[a][b].Hpxd_FC
    hpxdr = np.real(Hpxd_FC)
    hpxdc = np.imag(Hpxd_FC)
    magnitude_Hpxd = np.sqrt(np.square(hpxdr) + np.square(hpxdc))
    phase_Hpxd = np.angle(Hpxd_FC, deg=True)


    for j in range(len(phase_Hpe)):
        if j != len(phase_Hpe) - 1:
            if abs(phase_Hpe[j] - phase_Hpe[j + 1]) >= 180:
                #for j in range(i + 1, len (phase_Hpe)):
                phase_Hpe[j + 1] -= 360
            
            if phase_Hpe[j] * phase_Hpe[j + 1] < 0:
                print(phase_Hpe[j], phase_Hpe[j + 1])

    for k in range(len(phase_Hpxd)):
        if k != len(phase_Hpxd) - 1:
            if abs(phase_Hpxd[k] - phase_Hpxd[k + 1]) >= 180:
                #for j in range(i + 1, len (phase_Hpe)):
                phase_Hpxd[k + 1] -= 360
            
            if phase_Hpxd[k] * phase_Hpxd[k + 1] < 0:
                print(phase_Hpxd[k], phase_Hpxd[k + 1])
    box[i-1] = phase_Hpe.transpose()


w_FC = mat[a][b].w_FC

print(box)

fig, ax = plt.subplots()

# Create boxplots side by side
ax.boxplot(box)

# Add title and labels
plt.title('Boxplot')
plt.xlabel('Freq')
plt.ylabel('Phase')


# Show the plot
plt.show()