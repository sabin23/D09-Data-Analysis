import scipy.io
import matplotlib.pyplot as plt
from get_data import Data

import matplotlib.pyplot as plt

# Sample data for two groups
data_group1 = [10, 20, 30, 40, 50]
data_group2 = [25, 35, 45, 55, 65]

# Create a figure and axes
fig, ax = plt.subplots()

# Create boxplots side by side
ax.boxplot([data_group1, data_group2])

# Add title and labels
plt.title('Boxplot Example')
plt.xlabel('Group')
plt.ylabel('Value')

# Set x-axis ticks and labels
ax.set_xticks([1, 2])
ax.set_xticklabels(['Group 1', 'Group 2'])

# Show the plot
plt.show()
