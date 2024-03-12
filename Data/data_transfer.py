import os
import pandas as pd
import scipy.io as sio

# Specify the directory containing the MATLAB files
mat_dir = "Data"

# Iterate over all files in the directory
for filename in os.listdir(mat_dir):
    if filename.endswith(".mat"):
        # Load the contents of the MATLAB file
        mat_contents = sio.loadmat(os.path.join(mat_dir, filename))

        # Convert the MATLAB contents to a pandas DataFrame
        df = pd.DataFrame(mat_contents)

        # Specify the path to the new CSV file
        csv_filename = os.path.splitext(filename)[0] + "data.csv"
        csv_filepath = os.path.join(mat_dir, csv_filename)

        # Save the DataFrame to the CSV file
        df.to_csv(csv_filepath, index=False)

        print(f"Saved CSV file: {csv_filepath}")