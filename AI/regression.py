import sys

# Add the parent directory to sys.path
sys.path.append("..")

from get_data import Data


u = Data(1, 1).u
print(u)
