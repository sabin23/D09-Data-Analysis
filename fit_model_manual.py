import scipy.optimize as opt
import numpy as np
from get_data import Data
import matplotlib.pyplot as plt

# Data: subject, condition
subject = 1
condition = 1 
run = 0
data = Data(subject, condition)

Hpe_data = data.Hpe_FC
Hpxd_data = data.Hpxd_FC
w_FC = data.w_FC

x_total = data.x
x_total = np.transpose(x_total)
x = x_total[run]
x = np.transpose(x)
t = data.t[0]

# Coefficients
K_pe = 0.5
t_lead = 0.1
t_lag = 0.1
tau = 0.1
omega_nm = 10
zeta_nm = 0.7

def H_pe(w):
    jw = w * 1j
    return K_pe * (t_lead * jw + 1) / (t_lag * jw + 1) * np.exp(- jw * tau) * H_nm(jw)

def H_nm(w):
    jw = w * 1j
    return omega_nm ** 2 / (jw ** 2 + 2 * zeta_nm * omega_nm * jw + omega_nm ** 2)

def dfdt (f, t):
    dfdt = np.zeros(f.shape)
    for i in range(0, len(f)-1):
        if i > 0 and i < len(f)-1:
            dfdt[i] = (f[i+1] - f[i-1]) / (t[i+1] - t[i-1])
    dfdt[len(f)-1] = dfdt[len(f)-2]
    dfdt[0] = dfdt[1]
    return dfdt

def plot(subject, condition):
    a, b = subject, condition
    w_FC = data.w_FC
    response = H_pe(w_FC)

    # Get magnitude and phase of numbers in Hpe_FC
    Hpe_FC = Hpe_data
    hper = np.real(Hpe_FC)
    hpec = np.imag(Hpe_FC)
    magnitude_Hpe = np.sqrt(np.square(hper) + np.square(hpec))
    phase_Hpe = np.angle(Hpe_FC, deg=True)
    # Get magnitude and phase of numbers in Hpxd_FC
    Hpxd_FC = Hpxd_data
    hpxdr = np.real(Hpxd_FC)
    hpxdc = np.imag(Hpxd_FC)
    magnitude_Hpxd = np.sqrt(np.square(hpxdr) + np.square(hpxdc))
    phase_Hpxd = np.angle(Hpxd_FC, deg=True)
    # Get magnitude and phase of numbers in response
    hper_response = np.real(response)
    hpec_response = np.imag(response)
    magnitude_response = np.sqrt(np.square(hper_response) + np.square(hpec_response))
    phase_response = np.angle(response, deg=True)

    for i in range(len(phase_Hpe)):
        if i != len(phase_Hpe) - 1:
            if abs(phase_Hpe[i] - phase_Hpe[i + 1]) >= 180:
                phase_Hpe[i + 1] -= 360
            if abs(phase_Hpxd[i] - phase_Hpxd[i + 1]) >= 180:
                phase_Hpxd[i + 1] -= 360
            if abs(phase_response[i] - phase_response[i + 1]) >= 180:
                phase_response[i + 1] -= 360


    # Create a figure with two subplots
    fig, axs = plt.subplots(2)

    # Plot the magnitude of Hpe_FC in the first subplot
    axs[0].semilogx(w_FC, magnitude_Hpe, label='Hpe')
    axs[0].semilogx(w_FC, magnitude_Hpxd, label='Hpxd')
    axs[0].semilogx(w_FC, magnitude_response, label='Resonse')
    axs[0].set_title('Magnitude of Hpe_FC')
    axs[0].set_xlabel('Frequency (Hz)')
    axs[0].set_ylabel('Magnitude')
    axs[0].grid()
    axs[0].legend()

    # Plot the phase of Hpe_FC in the second subplot
    axs[1].semilogx(w_FC, phase_Hpe, label='Hpe')
    axs[1].semilogx(w_FC, phase_Hpxd, label='Hpxd')
    axs[1].semilogx(w_FC, phase_response, label='Response')
    axs[1].set_title('Phase of Hpe_FC')
    axs[1].set_xlabel('Frequency (Hz)')
    axs[1].set_ylabel('Phase (degrees)')
    axs[1].grid()
    axs[1].legend()

    plt.tight_layout()
    plt.show()

#plot(1, 1)

print(x.shape)
print(t.shape)


v = dfdt(x, t)

plot(1, 1)