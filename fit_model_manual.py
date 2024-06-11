import scipy.optimize as opt
import numpy as np
from get_data import Data
import matplotlib.pyplot as plt

# Data: subject, condition
subject = 5
condition = 6
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
K_m = 0.01
tau_m = 0.12

initial_parameters = []
initial_parameters.append([1., 0.1, 1., 0.25, 8., 0.6, 0.2, 0.15])        # Position
initial_parameters.append([0.45, 0.1, 0.1, 0.25, 10., 0.6, 0.55, 0.1])    # Velocity
initial_parameters.append([0.55, 1.0, 0.1, 0.25, 10., 0.6, 0.45, 0.12])   # Acceleration

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

def cost_function_pm(parameters, w, Hpe_data, Hpxd_data):
    Hpe_error = np.abs(Hpe_data - H_pe(parameters, w)) / np.abs(Hpe_data)
    Hpxd_error = np.abs(Hpxd_data - H_pxd(parameters, w)) / np.abs(Hpxd_data)

    cost = np.sum(Hpe_error)
    cost += np.sum(Hpxd_error)

    # cost = np.sum(np.abs(Hpxd_data - H_pxd(parameters, w)))
    # cost += np.sum(np.abs(Hpe_data - H_pe(parameters, w)))

    if parameters[0] < 0 or parameters[1] < 0 or parameters[2] < 0 or parameters[3] < 0 or parameters[4] < 0 or parameters[5] < 0 or parameters[6] < 0 or parameters[7] < 0:
        cost += 1000000
    # print(cost)
    return cost

optimized_parameters = []

def plot():
    w_FC = data.w_FC
    response_visual = H_pe(parameters, w_FC)
    resonse_motion = H_pxd(parameters, w_FC)

    opt_parameters = opt.fmin(cost_function_pm, initial_parameters[vehicle], args=(w_FC, Hpe_data, Hpxd_data))
    opt_response_hpe = H_pe(opt_parameters, w_FC)
    opt_response_hpxd = H_pxd(opt_parameters, w_FC)
    opt_response = opt_response_hpe + opt_response_hpxd
    oprimized_parameters = opt_parameters

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
    hper_response = np.real(response_visual)
    hpec_response = np.imag(response_visual)
    magnitude_response = np.sqrt(np.square(hper_response) + np.square(hpec_response))
    phase_response = np.angle(response_visual, deg=True)
    # Get magnitude and phase of numbers in opt_response
    hper_opt_response = np.real(opt_response)
    hpec_opt_response = np.imag(opt_response)
    magnitude_opt_response = np.sqrt(np.square(hper_opt_response) + np.square(hpec_opt_response))
    phase_opt_response = np.angle(opt_response, deg=True)
    # Get magnitude and phase of numbers in opt_response_hpe
    hper_opt_response_hpe = np.real(opt_response_hpe)
    hpec_opt_response_hpe = np.imag(opt_response_hpe)
    magnitude_opt_response_hpe = np.sqrt(np.square(hper_opt_response_hpe) + np.square(hpec_opt_response_hpe))
    phase_opt_response_hpe = np.angle(opt_response_hpe, deg=True)
    # Get magnitude and phase of numbers in opt_response_hpxd
    hper_opt_response_hpxd = np.real(opt_response_hpxd)
    hpec_opt_response_hpxd = np.imag(opt_response_hpxd)
    magnitude_opt_response_hpxd = np.abs(opt_response_hpxd)
    print(magnitude_opt_response_hpxd)
    phase_opt_response_hpxd = np.angle(opt_response_hpxd, deg=True)

    print(opt_parameters)

    for i in range(len(phase_Hpe)):
        if i != len(phase_Hpe) - 1:
            if abs(phase_Hpe[i] - phase_Hpe[i + 1]) >= 180:
                phase_Hpe[i + 1] -= 360
            if abs(phase_Hpxd[i] - phase_Hpxd[i + 1]) >= 180:
                phase_Hpxd[i + 1] -= 360
            if abs(phase_response[i] - phase_response[i + 1]) >= 180:
                phase_response[i + 1] -= 360
            if abs(phase_opt_response[i] - phase_opt_response[i + 1]) >= 180:
                phase_opt_response[i + 1] -= 360
            if abs(phase_opt_response_hpe[i] - phase_opt_response_hpe[i + 1]) >= 180:
                phase_opt_response_hpe[i + 1] -= 360
            if abs(phase_opt_response_hpxd[i] - phase_opt_response_hpxd[i + 1]) >= 180:
                phase_opt_response_hpxd[i + 1] -= 360


    # Create a figure with two subplots
    fig, axs = plt.subplots(2)

    # Plot the magnitude of Hpe_FC in the first subplot
    
    axs[0].loglog(w_FC, magnitude_Hpe, label='Hpe', color='deepskyblue')
    axs[0].loglog(w_FC, magnitude_Hpxd, label='Hpxd', color='orange')
    # axs[0].loglog(w_FC, magnitude_response, label='Initial Guess')
    axs[0].loglog(w_FC, magnitude_opt_response_hpe, label='Optimized Hpe', linestyle='dashed', color='deepskyblue')
    axs[0].loglog(w_FC, magnitude_opt_response_hpxd, label='Optimized Hpxd', linestyle='dashed', color='orange')
    axs[0].loglog(w_FC, magnitude_opt_response, label='Optimized', color='tomato')
    axs[0].set_xlabel('Frequency [Hz]')
    axs[0].set_ylabel('Magnitude [Deg]')
    axs[0].grid()
    axs[0].legend()

    # Plot the phase of Hpe_FC in the second subplot
    
    axs[1].semilogx(w_FC, phase_Hpe, label='Hpe', color='deepskyblue')
    axs[1].semilogx(w_FC, phase_Hpxd, label='Hpxd', color='orange')
    # axs[1].semilogx(w_FC, phase_response, label='Initial Guess')
    axs[1].semilogx(w_FC, phase_opt_response_hpe, label='Optimized Hpe', linestyle='dashed', color='deepskyblue')
    axs[1].semilogx(w_FC, phase_opt_response_hpxd, label='Optimized Hpxd', linestyle='dashed', color='orange')
    axs[1].semilogx(w_FC, phase_opt_response, label='Optimized', color='tomato')
    #axs[1].set_title('Phase of Hpe_FC')
    axs[1].set_xlabel('Frequency [Hz]')
    axs[1].set_ylabel('Phase [deg]')
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
    K_m_arr = [[], [], []]
    tau_m_arr = [[], [], []]
    
    for condition in range(4, 7):
        for subject in range(1, 7):
            data = Data(subject, condition)
            Hpe_data = data.Hpe_FC
            Hpxd_data = data.Hpxd_FC
            w_FC = data.w_FC
            
            optimized_parameters = opt.fmin(cost_function_pm, initial_parameters[condition - 4], args=(w_FC, Hpe_data, Hpxd_data))
            
            K_pe_arr[condition - 4].append(optimized_parameters[0])
            t_lead_arr[condition - 4].append(optimized_parameters[1])
            t_lag_arr[condition - 4].append(optimized_parameters[2])
            tau_arr[condition - 4].append(optimized_parameters[3])
            omega_nm_arr[condition - 4].append(optimized_parameters[4])
            zeta_nm_arr[condition - 4].append(optimized_parameters[5])
            K_m_arr[condition - 4].append(optimized_parameters[6])
            tau_m_arr[condition - 4].append(optimized_parameters[7])

    # make 8 plots, 1 for each parameter, with 3 boxplots, 1 for each condition
    fig, axs = plt.subplots(4, 2)
    axs[0, 0].boxplot(K_pe_arr)#, showfliers=False)
    axs[0, 0].set_title('K_pe [-]')
    #axs[0, 0].set_yscale('log')
    axs[0, 1].boxplot(t_lead_arr)#, showfliers=False)
    axs[0, 1].set_title('t_lead [s]')
    #axs[0, 1].set_yscale('log')
    axs[1, 0].boxplot(t_lag_arr)#, showfliers=False)
    axs[1, 0].set_title('t_lag [s]')
    #axs[0, 2].set_yscale('log')
    axs[1, 1].boxplot(tau_arr)#, showfliers=False)
    axs[1, 1].set_title('tau [s]')
    #axs[0, 3].set_yscale('log')
    axs[2, 0].boxplot(omega_nm_arr)#, showfliers=False)
    axs[2, 0].set_title('omega_nm [rad/s]')
    #axs[1, 0].set_yscale('log')
    axs[2, 1].boxplot(zeta_nm_arr)#, showfliers=False)
    axs[2, 1].set_title('zeta_nm [-]')
    #axs[1, 1].set_yscale('log')
    axs[3, 0].boxplot(K_m_arr)#, showfliers=False)
    axs[3, 0].set_title('K_m [-]')
    #axs[1, 1].set_yscale('log')
    axs[3, 1].boxplot(tau_m_arr)#, showfliers=False)
    axs[3, 1].set_title('tau_m [s]')
    #axs[1, 1].set_yscale('log')
    plt.tight_layout()
    plt.show()
    print("kpe_arr_m=", K_pe_arr)
    print("t_lead_arr_m=", t_lead_arr)
    print("t_lag_arr_m=", t_lag_arr)
    print("tau_arr_m=", tau_arr)
    print("omega_nm_arr_m=", omega_nm_arr)
    print("zeta_nm_arr_m=", zeta_nm_arr)
    print("K_m_arr_m=", K_m_arr)
    print("tau_m_arr_m=", tau_m_arr)
    

boxplots_hpe()

