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
