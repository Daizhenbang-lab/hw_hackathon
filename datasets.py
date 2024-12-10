import numpy as np
import torch
from torch.utils.data import Dataset


def preprocess_H(x):
    # shape of x: (n, 2, 64, 408, 2)
    xr, xi = x[:, :, :, :, 0], x[:, :, :, :, 1]
    x = np.concatenate([xr, xi], axis=1) # shape (n, 4, 64, 408)
    # x = x[:, :, :32, :]
    x = x.reshape(x.shape[0], 1, -1, x.shape[-1]) # shape (n, 1, 4*64, 408)
    return x


class ChannelTrainSet(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

        print(f'x shape: {self.x.shape}, y shape: {self.y.shape}')

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class ChannelTestSet(Dataset):
    def __init__(self, x):
        self.x = torch.tensor(x, dtype=torch.float32)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx]


def load_labelled_data(H_data_path, input_pos_path, num_samples=None):
    # load input pos data
    pos_data = np.loadtxt(input_pos_path)
    labelled_pos = pos_data[:, 0].astype(int) - 1 # 1-indexed to 0-indexed
    pos_data = pos_data[:, 1:]

    # read all_input data
    H = np.load(H_data_path)
    H = preprocess_H(H)

    # select the labelled data
    H = H[labelled_pos, :, :, :]

    # print(H.shape)
    if num_samples:
        indices = np.random.choice(H.shape[0], num_samples, replace=False)
        H = H[indices]
        pos_data = pos_data[indices]
    return ChannelTrainSet(H, pos_data)


def load_all_data(H_data_path):
    H = np.load(H_data_path)
    H = preprocess_H(H)
    return ChannelTestSet(H)


def load_eval_data(H_data_path, ground_truth_path):
    # load input pos data
    pos_data = np.loadtxt(ground_truth_path)

    # read all_input data
    H = np.load(H_data_path)
    H = preprocess_H(H)

    return ChannelTrainSet(H, pos_data)


if __name__ == '__main__':
    dataset_idx = 0
    data_idx = 1

    dataset = load_labelled_data(
        f"data/Ht{dataset_idx}_{data_idx}.npy",
        f"data/Dataset{dataset_idx}InputPos{data_idx}.txt")

