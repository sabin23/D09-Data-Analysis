import sys


# Add the parent directory to sys.path

sys.path.append("..")


from get_data import Data

import torch
import warnings
import numpy as np

import matplotlib.pyplot as plt

from matplotlib import gridspec
import time

# Ignore MatplotlibDeprecationWarning
warnings.filterwarnings("ignore", category=plt.cbook.VisibleDeprecationWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


print(f"[!] Using computing device: {device}.")

torch.set_default_device(device)


print("[+] Initializing data.")


def load_data(subject, condition):

    data = Data(subject, condition)
    data.set_tensor_mode()

    data.standardize()

    inputs, outputs = data.get_regression_data(device)

    inputs.to(device), outputs.to(device)
    return inputs, outputs


inputs, outputs = load_data(1, 3)

inputs2, outputs2 = load_data(1, 4)

print("[+] Initialized data and moved to computing device.")


print("[!] Creating model")


def epoch(model, inputs, outputs, loss_fn, optimizer):

    y_pred = model(inputs).output

    loss = loss_fn(y_pred, outputs)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()
    return loss


# A Super Amazing Regression Model (ASARM)


class Asarm(torch.nn.Module):

    class ModelResult:
        def __init__(self, model, output):
            self.model = model
            self.output = output

        def get_mse(self, outputs):
            mse = torch.nn.functional.mse_loss(self.output, outputs)
            return mse.item()

        def get_rmse(self, outputs):
            mse = torch.nn.functional.mse_loss(self.output, outputs)
            return torch.sqrt(mse).item()

        def get_error(self, outputs):
            error = self.output - outputs
            return error

        def get_plot_data(self, outputs):
            y_pred_np = self.output.cpu().detach().numpy()
            outputs_np = outputs.cpu().detach().numpy()
            return y_pred_np, outputs_np

    def __init__(self, name):

        super(Asarm, self).__init__()

        # Initialize the module with a name attribute
        self.name = name

        # Define the layers

        self.linear1 = torch.nn.Linear(2, 40)

        self.activation = torch.nn.ReLU()

        self.activation2 = torch.nn.Sigmoid()

        self.linear2 = torch.nn.Linear(40, 120)

        self.linear3 = torch.nn.Linear(120, 30)

        self.linear4 = torch.nn.Linear(30, 1)

    def forward(self, x):

        # Define the forward pass using the layers and activation function

        x = self.activation(self.linear1(x))

        x = self.activation(self.linear2(x))

        x = self.activation(self.linear3(x))

        x = self.activation2(self.linear4(x))

        return Asarm.ModelResult(self, x)

    def trainModel(self, inputs, outputs, learning_rate, epochs=1000):

        print(f"Training for {epochs} epochs on {str(self.name)}.")

        loss_fn = torch.nn.MSELoss()

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        for i in range(epochs):

            self.train()

            loss = epoch(self, inputs, outputs, loss_fn, optimizer)

            if i % 100 == 0:

                sys.stdout.flush()

                sys.stdout.write(f"\rEpoch {i} - Loss: {loss.item()}")

            if i == epochs - 1:

                sys.stdout.flush()

                sys.stdout.write(f"\rEpoch {i+1} - Loss: {loss.item()}\n")

        return self


def create_model(name, device):

    # Create an instance of the Asarm class with the given name and move it to the specified device

    model = Asarm(name).to(device)
    return model


FirstModel = create_model("Model1", device)

motionModel = create_model("Motion", device)


print("[!] Training model")


# def train(model, inputs, outputs, learning_rate, epochs=1000):

#     print(f"Training for {epochs} epochs on {str(model.name)}.")

#     loss_fn = torch.nn.MSELoss()

#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#     for i in range(epochs):
#         model.train()

#         loss = epoch(model, inputs, outputs, loss_fn, optimizer)

#         if i % 100 == 0:

#             sys.stdout.flush()

#             sys.stdout.write(f"\rEpoch {i} - Loss: {loss.item()}")

#         if i == epochs - 1:

#             sys.stdout.flush()

#             sys.stdout.write(f"\rEpoch {i+1} - Loss: {loss.item()}\n")


FirstModel.trainModel(inputs, outputs, 0.1, 4000)

motionModel.trainModel(inputs2, outputs2, 0.1, 4000)


print("[+] Training complete.")

# Plot the results

y_plot_no_motion, output_plot_no_motion = noMotionModel(inputs).get_plot_data(outputs)

# Test the model with the no motion data to see the difference in predictions
y_pred_plot_motion, output_plot_motion = motionModel(inputs2).get_graph_data(outputs2)


# mse1 = noMotionModel.calculate_mse(outputs)

# mse2 = calculate_mse(y_pred2, outputs2)

# y_pred_np1 = create_plot_data(y_pred)

# outputs_np1 = create_plot_data(outputs)

# y_pred_np2 = create_plot_data(y_pred2)

# outputs_np2 = create_plot_data(outputs2)

# error_np1 = calculate_error(y_pred_np1, outputs_np1)

# error_np2 = calculate_error(y_pred_np2, outputs_np2)

# mean1 = calculate_mse(y_pred, outputs)

# mean2 = calculate_mse(y_pred2, outputs2)


difference = y_pred_np2 - y_pred_np1


# Plotting

# Creating a figure and a set of subplots

fig, axs = plt.subplots(3, 2, figsize=(10, 12))  # 2 Rows, 2 Column

gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1])  # Adjust height ratios as needed


# First col, two rows

ax1 = plt.subplot(gs[0, 0])

ax2 = plt.subplot(gs[1, 0])


# Second row, two columns

ax3 = plt.subplot(gs[0, 1])

ax4 = plt.subplot(gs[1, 1])


# Third row, single column spanning both column spaces

ax5 = plt.subplot(gs[2, :])


# First subplot for Actual vs. Predicted

ax1.plot(outputs_np1, label="Actual", color="red")

ax1.plot(y_pred_np1, label="Predicted", color="blue")

ax1.set_title(f"Comparison of Predicted and Actual Values {noMotionModel.name}")

ax1.set_xlabel("Index")

ax1.set_ylabel("U - value")

ax1.legend()

ax1.grid(True)


# Second subplot for Error

ax2.plot(error_np1, label="Error (prediction - actual)", color="orange")

ax2.axhline(y=mean1, label="Mean error", color="black", linestyle="--")

ax2.set_title(f"Error Between Predicted and Actual Values {noMotionModel.name}")

ax2.set_xlabel("Index")

ax2.set_ylabel("Error")

ax2.legend()

ax2.grid(True)


# First subplot for Actual vs. Predicted

ax3.plot(outputs_np2, label="Trained on", color="green")
ax3.plot(outputs_np1, label="Actual", color="red")
ax3.plot(y_pred_np2, label="Predicted", color="blue")

ax3.set_title(f"Comparison of Predicted and Actual Values {motionModel.name}")

ax3.set_xlabel("Index")

ax3.set_ylabel("U - value")

ax3.legend()

ax3.grid(True)


# Second subplot for Error

ax4.plot(error_np2, label="Error (prediction - actual)", color="orange")

ax4.axhline(y=mean2, label="Mean error", color="black", linestyle="--")

ax4.set_title(f"Error Between Predicted and Actual Values {motionModel.name}")

ax4.set_xlabel("Index")

ax4.set_ylabel("Error")

ax4.legend()

ax4.grid(True)


# Difference in predictions

ax5.plot(difference, label="Difference (no motion - motion)", color="blue")

ax5.set_title("Difference between motion and no motion models on the same data.")

ax5.set_xlabel("Index")

ax5.set_ylabel("Difference in U")

ax5.legend()

ax5.grid(True)


# Adjust layout to not overlap

plt.tight_layout()


# Show the plot

plt.show()
