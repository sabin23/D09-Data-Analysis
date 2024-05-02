import sys

# Add the parent directory to sys.path

sys.path.append("..")


from get_data import Data
import matplotlib.pyplot as plt
import numpy as np


position = Data(3, 3)
velocity = Data(3, 4)

u_pos = position.u
e_pos = position.e
t_pos = np.transpose(np.array(position.t))

u_vel = velocity.u
e_vel = velocity.e
t_vel = np.transpose(np.array(velocity.t))


def get_rms(data):
    return np.sqrt(np.mean(data**2))


def get_dot(data, t):
    data = data.reshape(-1, 1)
    dfdt = np.diff(data, axis=0) / np.diff(t, axis=0)
    return np.concatenate((dfdt, dfdt[-1].reshape(1, -1)), axis=0)


def rms_slices(data, slices):
    slices = np.split(data, slices)
    return np.array([get_rms(_slice) for _slice in slices])


rms_u_pos = get_rms(u_pos)
rms_e_pos = get_rms(e_pos)

rms_e_pos_dot = np.sqrt(np.mean(e_pos**2, axis=1))

rms_e_dot_pos = get_dot(rms_e_pos_dot, t_pos)


rms_slices_u_pos = rms_slices(u_pos, 32)
rms_slices_e_pos = rms_slices(e_pos, 32)
rms_slices_t_pos = rms_slices(t_pos, 32)

rms_slices_e_dot_pos = rms_slices(rms_e_dot_pos, 32)

rms_u_vel = get_rms(u_vel)
rms_e_vel = get_rms(e_vel)

rms_e_vel_dot = np.sqrt(np.mean(e_vel**2, axis=1))

rms_e_dot_vel = get_dot(rms_e_vel_dot, t_vel)

rms_slices_e_dot_vel = rms_slices(rms_e_dot_vel, 32)
rms_slices_u_vel = rms_slices(u_vel, 32)
rms_slices_e_vel = rms_slices(e_vel, 32)
rms_slices_t_vel = rms_slices(t_vel, 32)


plt.subplot(3, 2, 1)
plt.plot(rms_slices_t_pos, rms_slices_e_dot_pos, color="red")
plt.title("Plot of edot")
plt.subplot(3, 2, 3)
plt.plot(rms_slices_t_pos, rms_slices_e_pos, color="blue")
plt.title("Plot of e")
plt.subplot(3, 2, 5)
plt.scatter(
    rms_slices_e_pos,
    rms_slices_u_pos,
    color="green",
    s=2,  # Adjust the size of the dots here
)
plt.title("Plot of e vs u")


plt.subplot(3, 2, 2)
plt.plot(rms_slices_t_vel, rms_slices_e_dot_vel, color="red")
plt.title("Plot of edot")
plt.subplot(3, 2, 4)
plt.plot(rms_slices_t_vel, rms_slices_e_vel, color="blue")
plt.title("Plot of e")
plt.subplot(3, 2, 6)
plt.scatter(
    rms_slices_e_vel,
    rms_slices_u_vel,
    color="green",
    s=2,  # Adjust the size of the dots here
)
plt.title("Plot of e vs u vel")

plt.show()


plt.subplot(3, 1, 1)
plt.plot(rms_slices_t_pos, rms_slices_e_dot_pos - rms_slices_e_dot_vel, color="red")
plt.title("Difference in e dot nm, m")
plt.subplot(3, 1, 2)
plt.plot(rms_slices_t_pos, rms_slices_e_pos - rms_slices_e_vel, color="blue")
plt.title("Difference in e nm, m")
plt.subplot(3, 1, 3)
plt.scatter(
    rms_slices_e_pos - rms_slices_e_vel,
    rms_slices_u_pos - rms_slices_u_vel,
    color="green",
    s=2,  # Adjust the size of the dots here
)

plt.show()
