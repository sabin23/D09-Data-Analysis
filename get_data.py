import os
import pandas as pd
import scipy.io as sio


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
        # print(mat)
        self.u = mat["u"]
        self.t = mat["t"]
        self.x = mat["x"]
        self.ft = mat["ft"]
        self.fd = mat["fd"]
        self.e = mat["e"]
        self.Hpxd_FC = mat["Hpxd_FC"]
        self.Hpe_FC = mat["Hpe_FC"]
        self.w_FC = mat["w_FC"]
