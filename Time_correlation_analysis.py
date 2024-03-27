import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from get_data import Data
from scipy import signal

# Load the data
mat = []
for i in range(1, 7):
    line = []
    for j in range(1, 7):
        line.append(Data(i, j))
    mat.append(line)

# Eliminate Noise from signal

e_matrix = mat[1][1].e
e = np.array([np.mean(lst) for lst in e_matrix])
u_matrix = mat[1][1].u
t_matrix = mat[1][1].t
f_matrix = mat[1][1].ft

#Design a low-pass Butterworth filter
nyquist_freq = 0.5 * 1000  # Nyquist frequency
cutoff_freq = 5  # Cutoff frequency for the filter
order = 3  # Filter order
rp = 0.1
btype = "lowpass"
b, a = signal.cheby1(order, rp, cutoff_freq / nyquist_freq, btype=btype)

# Apply the filter to the noisy signal
filtered_signal = signal.filtfilt(b, a, e)

#Plot the original signal and the filtered signal
plt.figure(figsize=(10, 6))
# plt.plot(np.transpose(t_matrix), e, label='Noisy Signal')
# plt.plot(np.transpose(t_matrix), filtered_signal, label='Filtered Signal')
plt.plot(f_matrix, e)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Noise Removal with Low-pass Butterworth Filter')
plt.legend()
plt.grid(True)
plt.show()





# plt.plot(np.transpose(t_matrix), e)
# plt.show()

# Correlation Analysis