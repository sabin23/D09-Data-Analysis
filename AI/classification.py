import sys
import torch
import numpy as np

# Add the parent directory to sys.path
sys.path.append("..")

from get_data import Data

def dfdt (f, t):
    dfdt = np.zeros(f.shape)
    for i in range(0, len(f)-1):
        if i > 0 and i < len(f)-1:
            dfdt[i] = (f[i+1] - f[i-1]) / (t[i+1] - t[i-1])
    dfdt[len(f)-1] = dfdt[len(f)-2]
    dfdt[0] = dfdt[1]
    return dfdt

data = Data(1, 1)

u1 = Data(1, 1).u
print(u1)
u2 = Data(1, 2).u
print(u2)


t = Data(1, 1).t[0]
x = Data(1, 1).x

x_t = x.transpose(1, 0)
dxdt = np.zeros(x_t.shape)
for i in range(0,len(x_t)):
    dxdt[i] = dfdt(x_t[i], t)





# import matplotlib.pyplot as plt
# import torch.nn as nn
# import torch.optim as optim

# # Define the classification neural network
# class Classifier(nn.Module):
#     def __init__(self):
#         super(Classifier, self).__init__()
#         self.fc1 = nn.Linear(5, 10)
#         self.fc2 = nn.Linear(10, 1)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.sigmoid(x)
#         x = self.fc2(x)
#         return x
# # Create an instance of the classifier
# classifier = Classifier()

# # Define the loss function and optimizer
# criterion = nn.MSELoss()
# optimizer = optim.SGD(classifier.parameters(), lr=0.01)

# # Train the classifier
# u2t = u2.transpose(1, 0)
# # u2t = u2t[0:1]
# for u2t_subset in u2t:
#     for epoch in range(100):
#         optimizer.zero_grad()
#         u1t = torch.tensor(u1, dtype=torch.float32)
#         u2t = torch.tensor(u2t_subset, dtype=torch.float32)
#         outputs = classifier(u1t)
#         loss = criterion(outputs[:,0], u2t)
#         loss.backward()
#         optimizer.step()

#     # Plot the result
#     plt.scatter(classifier(u1t).detach().numpy(), u2t_subset, label='True', marker='.')

# plt.xlabel('t')
# plt.ylabel('u')
# plt.legend()

# plt.savefig('classification.png')