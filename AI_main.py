import scipy.io
import matplotlib.pyplot as plt
from get_data import Data



mat = scipy.io.loadmat("./Data/ae2224I_measurement_data_subj1_C1.mat")
u = mat['u']
t = mat['t']
data = Data(1,1)
print(data.u)
fig, ax = plt.subplots()
ax.plot(t[0], u[:,0])

ax.set(xlabel='time (s)', ylabel='u',
       title='u-t')
ax.grid()
plt.show()

