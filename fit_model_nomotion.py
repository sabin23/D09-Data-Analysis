import scipy.optimize as opt
import numpy as np
from get_data import Data
import matplotlib.pyplot as plt

# Data: subject, condition
subject = 5
condition = 3
run = 4
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
K_pe = 1.0       # magnitude of magnitude graph
t_lead = 1.0     # moves phase graph up
t_lag = 0.1      # moves phase graph down
tau = 0.25       # decreases gradient of phase graph
omega_nm = 10    # location of magnitude peak
zeta_nm = 0.6    # height of magnitude peak


initial_parameters = []
initial_parameters.append([1., 0.1, 1., 0.25, 8., 0.6])        # Position
initial_parameters.append([0.45, 0.1, 0.1, 0.25, 10., 0.6])    # Velocity
initial_parameters.append([0.55, 1.0, 0.1, 0.25, 10., 0.6])   # Acceleration

vehicle = condition - 4
parameters = initial_parameters[vehicle]

"""
- Boxplots for parameters in no motion case
- New fits for motion model (conditions 4-6)
- Boxplots for total model (4-6)
"""

def H_pe(parameters, w):
    jw = w * 1j
    K_pe = parameters[0]
    t_lead = parameters[1]
    t_lag = parameters[2]
    tau = parameters[3]
    return K_pe * (t_lead * jw + 1) / (t_lag * jw + 1) * np.exp(- jw * tau) * H_nm(parameters, jw)

def H_nm(parameters, w):
    jw = w
    omega_nm = parameters[4]
    zeta_nm = parameters[5]
    return omega_nm ** 2 / (jw ** 2 + 2 * zeta_nm * omega_nm * jw + omega_nm ** 2)

def H_scc(parameters, w):
    return (0.11 * w + 1) / (5.9 * w + 1)

def H_pxd(parameters, w):
    jw = w * 1j
    K_m = parameters[6]
    tau_m = parameters[7]
    return jw * H_scc(parameters, jw) * K_m * np.exp(-jw * tau_m) * H_nm(parameters, jw)

def cost_function(parameters, w, Hpe_data, Hpxd_data):
    Hpe_error = np.abs(Hpe_data - H_pe(parameters, w)) / np.abs(Hpe_data)

    cost = np.sum(Hpe_error)

    if parameters[0] < 0 or parameters[1] < 0 or parameters[2] < 0 or parameters[3] < 0 or parameters[4] < 0 or parameters[5]:
        cost += 1000000
    # print(cost)
    return cost

optimized_parameters = []

def plot():
    w_FC = data.w_FC
    response = H_pe(parameters, w_FC)

    opt_parameters = opt.fmin(cost_function, initial_parameters[vehicle], args=(w_FC, Hpe_data, Hpxd_data))
    opt_response = H_pe(opt_parameters, w_FC)

    # Get magnitude and phase of numbers in Hpe_FC
    Hpe_FC = Hpe_data
    hper = np.real(Hpe_FC)
    hpec = np.imag(Hpe_FC)
    magnitude_Hpe = np.sqrt(np.square(hper) + np.square(hpec))
    phase_Hpe = np.angle(Hpe_FC, deg=True)
    # Get magnitude and phase of numbers in response
    hper_response = np.real(response)
    hpec_response = np.imag(response)
    magnitude_response = np.sqrt(np.square(hper_response) + np.square(hpec_response))
    phase_response = np.angle(response, deg=True)
    # Get magnitude and phase of numbers in opt_response
    hper_opt_response = np.real(opt_response)
    hpec_opt_response = np.imag(opt_response)
    magnitude_opt_response = np.sqrt(np.square(hper_opt_response) + np.square(hpec_opt_response))
    phase_opt_response = np.angle(opt_response, deg=True)
    
    

    print(opt_parameters)

    for i in range(len(phase_Hpe)):
        if i != len(phase_Hpe) - 1:
            if abs(phase_Hpe[i] - phase_Hpe[i + 1]) >= 180:
                phase_Hpe[i + 1] -= 360
            if abs(phase_response[i] - phase_response[i + 1]) >= 180:
                phase_response[i + 1] -= 360
            if abs(phase_opt_response[i] - phase_opt_response[i + 1]) >= 180:
                phase_opt_response[i + 1] -= 360


    # Create a figure with two subplots
    fig, axs = plt.subplots(2)

    # Plot the magnitude of Hpe_FC in the first subplot
    
    axs[0].loglog(w_FC, magnitude_Hpe, label='Hpe', color='deepskyblue')
    # axs[0].loglog(w_FC, magnitude_response, label='Initial Guess')
    axs[0].loglog(w_FC, magnitude_opt_response, label='Optimized', color='tomato')
    axs[0].set_xlabel('Frequency [Hz]')
    axs[0].set_ylabel('Magnitude [Deg]')
    axs[0].grid()
    axs[0].legend()

    # Plot the phase of Hpe_FC in the second subplot
    
    axs[1].semilogx(w_FC, phase_Hpe, label='Hpe', color='deepskyblue')
    # axs[1].semilogx(w_FC, phase_response, label='Initial Guess')
    axs[1].semilogx(w_FC, phase_opt_response, label='Optimized', color='tomato')
    #axs[1].set_title('Phase of Hpe_FC')
    axs[1].set_xlabel('Frequency [Hz]')
    axs[1].set_ylabel('Phase [Deg]')
    axs[1].grid()
    axs[1].legend()

    plt.tight_layout()
    plt.show()

plot()

def boxplots_hpe():
    K_pe_arr = [[], [], []]
    t_lead_arr = [[], [], []]
    t_lag_arr = [[], [], []]
    tau_arr = [[], [], []]
    omega_nm_arr = [[], [], []]
    zeta_nm_arr = [[], [], []]
    
    for condition in range(4, 7):
        for subject in range(1, 7):
            data = Data(subject, condition)
            Hpe_data = data.Hpe_FC
            Hpxd_data = data.Hpxd_FC
            w_FC = data.w_FC
            
            optimized_parameters = opt.fmin(cost_function, initial_parameters[condition - 4], args=(w_FC, Hpe_data, Hpxd_data))
            
            K_pe_arr[condition - 4].append(optimized_parameters[0])
            t_lead_arr[condition - 4].append(optimized_parameters[1])
            t_lag_arr[condition - 4].append(optimized_parameters[2])
            tau_arr[condition - 4].append(optimized_parameters[3])
            omega_nm_arr[condition - 4].append(optimized_parameters[4])
            zeta_nm_arr[condition - 4].append(optimized_parameters[5])


    # make 8 plots, 1 for each parameter, with 3 boxplots, 1 for each condition
    fig, axs = plt.subplots(3, 2)
    axs[0, 0].boxplot(K_pe_arr, showfliers=False)
    axs[0, 0].set_title('K_pe [s]')
    #axs[0, 0].set_yscale('log')
    axs[0, 1].boxplot(t_lead_arr, showfliers=False)
    axs[0, 1].set_title('t_lead [s]')
    #axs[0, 1].set_yscale('log')
    axs[1, 0].boxplot(t_lag_arr, showfliers=False)
    axs[1, 0].set_title('t_lag [s]')
    #axs[0, 2].set_yscale('log')
    axs[1, 1].boxplot(tau_arr, showfliers=False)
    axs[1, 1].set_title('tau [s]')
    #axs[0, 3].set_yscale('log')
    axs[2, 0].boxplot(omega_nm_arr, showfliers=False)
    axs[2, 0].set_title('omega_nm [rad/s]')
    #axs[1, 0].set_yscale('log')
    axs[2, 1].boxplot(zeta_nm_arr, showfliers=False)
    axs[2, 1].set_title('zeta_nm [-]')
    #axs[1, 1].set_yscale('log')
    print("kpe_arr_nm=", K_pe_arr)
    print("t_lead_arr_nm=", t_lead_arr)
    print("t_lag_arr_nm=", t_lag_arr)
    print("tau_arr_nm=", tau_arr)
    print("omega_nm_arr=", omega_nm_arr)
    print("zeta_nm_arr=", zeta_nm_arr)
    


    
    plt.tight_layout()
    plt.show()

boxplots_hpe()

