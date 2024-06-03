import numpy as np
import torch
import torch.nn as nn

class Denoiser(nn.Module):
    def __init__(self, dim, hidden_dims):
        super().__init__()
        layers = []
        dims = (dim + 2,) + hidden_dims
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.extend([nn.Linear(in_dim, out_dim), nn.GELU()])
        layers.append(nn.Linear(hidden_dims[-1], dim))
        self.net = nn.Sequential(*layers)
        self.input_dims = (dim,)

    def rand_input(self, batchsize):
        return torch.randn((batchsize,) + self.input_dims)

    def forward(self, x, sigma_embeds):
        nn_input = torch.cat([x, sigma_embeds], dim=1) # shape: b x (dim + 2)
        return self.net(nn_input)


def get_dimension(data):
    if data == "swissroll":
        return 2
    elif data == "mnist":
        return 4
    
class CNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [
                nn.Conv2d(1, 16, kernel_size=3, ),
                nn.GELU(),
                nn.Conv2d(16, 16, kernel_size=3, ),
                nn.GELU(),
                nn.Conv2d(16, 16, kernel_size=3),
                nn.GELU(),
                nn.Conv2d(16, 16, kernel_size=3, ),
                nn.GELU(),
                nn.Conv2d(16, 16, kernel_size=3),
                nn.GELU(),
                nn.Conv2d(16, 16, kernel_size=3, ),
                nn.GELU(),
                nn.Conv2d(16, 16, kernel_size=3,),
                nn.GELU(),
                nn.Conv2d(16, 1, kernel_size=1, ),
                nn.GELU(),
                nn.Flatten(),
            ]
        self.net = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.net(x)
        # for i, layer in enumerate(self.layers):
        #     x = layer(x)
        # return x
    
class CNNDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [
                # nn.(4, (1, 2, 2)),
                # reshape to (1, 2, 2)
                nn.Unflatten(1, (1, 2, 2)),
                nn.Conv2d(1, 16, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(16, 16, kernel_size=1),
                nn.GELU(),
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(16, 16, kernel_size=3, padding="same"),
                nn.GELU(),
                nn.Conv2d(16, 16, kernel_size=3, padding="same"),
                nn.GELU(),
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(16, 16, kernel_size=3, padding="same"),
                nn.GELU(),
                nn.Conv2d(16, 16, kernel_size=3, padding="same"),
                nn.GELU(),
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(16, 16, kernel_size=3, padding="same"),
                nn.GELU(),
                nn.Conv2d(16, 1, kernel_size=3, padding="same"),
                nn.GELU(),
            ]
        self.net = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.net(x)
        # for i, layer in enumerate(self.layers):
        #     x = layer(x)
        # return x
    