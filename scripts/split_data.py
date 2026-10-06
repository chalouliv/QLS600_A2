from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parent.parent
data = np.loadtxt(root / "data/CookLab1UnknownChannel.txt", delimiter="\t", unpack=True)

# step protocol is used for training, the fluctuating-voltage section for testing
v = data[1]
is_step = np.isin(v, [-100, -60, -40, -20, 0, 20])
k = np.argmin(is_step)          # first row of the fluctuating section
train, test = data[:, :k], data[:, k:]

# %.15g writes values back exactly as they appear in the original file
np.savetxt(root / "data/train_data.txt", train.T, delimiter="\t", fmt="%.15g")
np.savetxt(root / "data/test_data.txt", test.T, delimiter="\t", fmt="%.15g")
