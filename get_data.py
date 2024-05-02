import os
import pandas as pd
import scipy.io as sio
import numpy as np
import torch


def standardize(data):
    # Compute the mean and std of the dataset
    mean = torch.mean(data, dim=0)
    std = torch.std(data, dim=0)

    # Standardize the data
    standardized_data = (data - mean) / std
    return standardized_data


class Data:
    def __init__(self, subject, condition):

        ### !!!! IMPORTANT !!!!
        # This is needed for inporting from different directories other than the parent directory.
        ### DO NOT CHANGE current_dir and data_file_path
        # Get the directory of the current script
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Construct the path to the data file
        data_file_path = os.path.join(
            current_dir, f"Data/ae2224I_measurement_data_subj{subject}_C{condition}.mat"
        )

        mat = sio.loadmat(data_file_path)

        self.tensor_mode = False
        self.u = mat["u"]
        self.t = mat["t"]
        self.x = mat["x"]
        self.ft = mat["ft"]
        self.fd = mat["fd"]
        self.e = mat["e"]
        self.Hpxd_FC = mat["Hpxd_FC"]
        self.Hpe_FC = mat["Hpe_FC"]
        self.w_FC = mat["w_FC"]
        if condition in [2, 4, 6]:
            self.type = "motion"
            self.x_dot = self.use_x_dot()
        else:
            self.type = "no_motion"
            self.x_dot = torch.zeros([8192, 1])

    def use_x_dot(self):
        if self.tensor_mode == False:
            return np.diff(np.mean(self.x, axis=1), axis=0) / np.diff(
                np.transpose(self.t), axis=0
            )

        x_dot = self.x.mean(1).unsqueeze(-1)
        t_dot = self.t.transpose(0, 1)
        x_diff = torch.diff(x_dot, axis=0)
        t_diff = torch.diff(t_dot, axis=0)
        x_dot = x_diff / t_diff
        last_value = x_dot[-1, :].unsqueeze(0)
        x_dot = torch.cat((x_dot, last_value), dim=0).float()
        return x_dot

    def standardize(self):
        # Apply the standardization of the data
        self.u = standardize(self.u)
        self.x = standardize(self.x)
        self.ft = standardize(self.ft)
        self.fd = standardize(self.fd)
        self.e = standardize(self.e)
        self.Hpxd_FC = standardize(self.Hpxd_FC)
        self.Hpe_FC = standardize(self.Hpe_FC)
        self.w_FC = standardize(self.w_FC)
        return self

    def set_tensor_mode(self):
        if self.tensor_mode == True:
            return

        self.tensor_mode = True
        self.u = torch.from_numpy(self.u).float()
        self.t = torch.from_numpy(self.t)
        self.x = torch.from_numpy(self.x).float()
        self.ft = torch.from_numpy(self.ft).float()
        self.fd = torch.from_numpy(self.fd).float()
        self.e = torch.from_numpy(self.e).float()
        self.Hpxd_FC = torch.from_numpy(self.Hpxd_FC).float()
        self.Hpe_FC = torch.from_numpy(self.Hpe_FC).float()
        self.w_FC = torch.from_numpy(self.w_FC).float()
        return self

    def get_regression_data(self, device):
        e_tensor = self.e.mean(1).unsqueeze(1)
        output = self.u.mean(1).unsqueeze(1).to(device)
        inputs = torch.cat((e_tensor.to(device), self.x_dot.to(device)), dim=1)
        return inputs, output
