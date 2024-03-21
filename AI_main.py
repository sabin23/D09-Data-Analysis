import scipy.io
import matplotlib.pyplot as plt
from get_data import Data

data = Data(1,1)
<<<<<<< HEAD
print(data.u)
fig, ax = plt.subplots()
ax.plot(data.t[0], data.u[:,0])
=======
t = data.t
u = data.u
>>>>>>> AI_1

fig, ax = plt.subplots()
ax.plot(t[0], u[:,0])

ax.set(xlabel='time (s)', ylabel='u',
       title='u-t')
ax.grid()
plt.show()