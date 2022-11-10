import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class BrainDataset(Dataset):
    def __init__(self, file_path):
        data = np.loadtxt(file_path, delimiter=',', dtype=np.float32)
        x_data = data[:, :-1]
        y_data = data[:, [-1]].flatten()
        y_data.astype(int)
        self.x_data = torch.from_numpy(x_data)
        self.y_data = torch.LongTensor(y_data)
        self.len = data.shape[0]

    def __getitem__(self, item):
        return self.x_data[item], self.y_data[item]

    def __len__(self):
        return self.len
def generate_dataloader(path_train, path_test, size=32):
    dataset_train = BrainDataset(path_train)
    dataset_test = BrainDataset(path_test)
    train_data = DataLoader(dataset=dataset_train, batch_size=size, shuffle=False)
    test_data = DataLoader(dataset=dataset_test, batch_size=size, shuffle=False)
    return train_data, test_data