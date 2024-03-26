import scipy.optimize as opt
import numpy as np
from get_data import Data
import matplotlib.pyplot as plt

# Data: subject, condition
subject = 0
condition = 0 
data = Data(subject, condition)

Hpe_data = data.Hpe_FC
Hpxd_data = data.Hpxd_FC

mat = []
for i in range(1, 7):
    line = []
    for j in range(1, 7):
        line.append(Data(i, j))
    mat.append(line)


# Coefficients
K_pe = 0.5
t_lead = 0.1
t_lag = 0.1
tau = 0.1
omega_nm = 10
zeta_nm = 0.7

def H_pe(jw):
    return K_pe * (t_lead * jw + 1) / (t_lag * jw + 1) * np.exp(- jw * tau) * H_nm(jw)

def H_nm(jw):
    return omega_nm ** 2 / (jw ** 2 + 2 * zeta_nm * omega_nm * jw + omega_nm ** 2)

def plot(subject, condition):
    a, b = subject, condition

    # Get magnitude and phase of numbers in Hpe_FC
    Hpe_FC = mat[a][b].Hpe_FC
    hper = np.real(Hpe_FC)
    hpec = np.imag(Hpe_FC)
    magnitude_Hpe = np.sqrt(np.square(hper) + np.square(hpec))
    phase_Hpe = np.angle(Hpe_FC, deg=True)
    # Get magnitude and phase of numbers in Hpxd_FC
    Hpxd_FC = mat[a][b].Hpxd_FC
    hpxdr = np.real(Hpxd_FC)
    hpxdc = np.imag(Hpxd_FC)
    magnitude_Hpxd = np.sqrt(np.square(hpxdr) + np.square(hpxdc))
    phase_Hpxd = np.angle(Hpxd_FC, deg=True)


    for i in range(len(phase_Hpe)):
        if i != len(phase_Hpe) - 1:
            if abs(phase_Hpe[i] - phase_Hpe[i + 1]) >= 180:
                #for j in range(i + 1, len (phase_Hpe)):
                phase_Hpe[i + 1] -= 360

    for i in range(len(phase_Hpxd)):
        if i != len(phase_Hpxd) - 1:
            if abs(phase_Hpxd[i] - phase_Hpxd[i + 1]) >= 180:
                #for j in range(i + 1, len (phase_Hpe)):
                phase_Hpxd[i + 1] -= 360
                
            if phase_Hpxd[i] * phase_Hpxd[i + 1] < 0:
                print(phase_Hpxd[i], phase_Hpxd[i + 1])


    w_FC = mat[a][b].w_FC
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

plot(0, 0)