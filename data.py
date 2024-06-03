from matplotlib import pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset

class Swissroll(Dataset):
    def __init__(self, theta_min, theta_max, N):
        self.N = N
        self.theta_min = theta_min
        self.theta_max = theta_max

    def __len__(self) -> int:
        return self.N
    
    def __getitem__(self, idx: int) -> torch.FloatTensor:
        theta = idx / self.N * (self.theta_max - self.theta_min) + self.theta_min
        theta = torch.tensor(theta)
        radius = idx / self.N
        x = radius * torch.cos(theta)
        y = radius * torch.sin(theta)
        return torch.stack([x, y], dim=0)

from torchvision import datasets
from torchvision.transforms import ToTensor
class MNISTDataset(Dataset):
    def __init__(self, encoder=None):
        def transform(x):
            x  = x.resize((16, 16))
            return ToTensor()(x)

        self.dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        self.encoder = encoder
        # freeze the encoder
        if self.encoder is not None:
            self.encoder.eval()
        self.encoder_dataset = {}

    # def get_or_compute_encoder(self, x):
    #     x_hash = tuple(x.flatten().tolist())
    #     if x_hash in self.cache:
    #         return self.cache[x_hash]
    #     else:
    #         y = self.encoder(x).squeeze()
    #         self.cache[x_hash] = y
    #         return y

    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> torch.FloatTensor:
        if self.encoder is None:
            return self.dataset[idx][0]
        elif idx in self.encoder_dataset:
            return self.encoder_dataset[idx]
        else:
            x, _ = self.dataset[idx]
            with torch.no_grad():
                y = self.encoder(x).squeeze()
            self.encoder_dataset[idx] = y
            return y
        
def get_dataset(dataset_name, encoder=None):
    if dataset_name == "swissroll":
        dataset = Swissroll(np.pi/2, 5*np.pi, 100)
        datapoints = [dataset[i] for i in range(len(dataset))]
        plt.scatter([x[0] for x in datapoints], [x[1] for x in datapoints])
        plt.savefig("swissroll.png")
        plt.clf()
    elif dataset_name == "mnist":
        dataset = MNISTDataset(encoder=encoder)
    return dataset