from tqdm import tqdm
from data import Swissroll
from torch import nn
import torch
from torch.utils.data import DataLoader
import numpy as np
import math
import matplotlib.pyplot as plt

from model import TimeInputMLP

class Schedule:
    def __init__(self, sigmas: torch.FloatTensor):
        self.sigmas = sigmas
    def __getitem__(self, i) -> torch.FloatTensor:
        return self.sigmas[i]
    def __len__(self) -> int:
        return len(self.sigmas)
    def sample_batch(self, x0:torch.FloatTensor) -> torch.FloatTensor:
        return self[torch.randint(len(self), (x0.shape[0],))].to(x0)

    def sample_sigmas(self, steps: int) -> torch.FloatTensor:
        indices = list((len(self) * (1 - np.arange(0, steps)/steps))
                       .round().astype(np.int64) - 1)
        return self[indices + [0]]
    
class ScheduleLogLinear(Schedule):
    def __init__(self, N: int, sigma_min: float=0.02, sigma_max: float=10):
        super().__init__(torch.logspace(math.log10(sigma_min), math.log10(sigma_max), N))


def training_loop(loader  : DataLoader,
                  denoiser   : nn.Module,
                  schedule: Schedule,
                  epochs  : int = 10000):
    optimizer = torch.optim.Adam(denoiser.parameters())
    losses = []
    for _ in tqdm(range(epochs)):
        for x0 in loader:
            optimizer.zero_grad()
            sigma, eps = generate_train_sample(x0, schedule)
            sigma_embeds = get_sigma_embeds(sigma)
            eps_hat = denoiser(x0 + sigma.unsqueeze(1) * eps, sigma_embeds)
            loss = nn.MSELoss()(eps_hat, eps)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
    return losses

def generate_train_sample(x0: torch.FloatTensor, schedule: Schedule):
    sigma = schedule.sample_batch(x0)
    eps = torch.randn_like(x0)
    return sigma, eps

def get_sigma_embeds(sigma):
    sigma = sigma.unsqueeze(1)
    return torch.cat([torch.sin(torch.log(sigma)/2),
                      torch.cos(torch.log(sigma)/2)], dim=1)


def sample_diffusion(x, denoiser, schedule):
    sigmas = schedule.sample_sigmas(20)
    xt = x * sigmas[0]
    for sig, sig_prev in zip(sigmas, sigmas[1:]):
        # replicate sigma so that it has shape (batchsize,)
        sig_batch = sig.repeat(x.shape[0])
        with torch.no_grad():
            sigma_embeds = get_sigma_embeds(sig_batch.to(xt))
            eps = denoiser(xt, sigma_embeds)
        xt -= (sig - sig_prev) * eps
    return xt   

def get_dimension(data):
    if data == "swissroll":
        return 2
    elif data == "mnist":
        return 16

def main(
        # task = "train",
        task = "sample",
        data="swissroll",
        # data="mnist",
):
    # if data == "swissroll":
    # elif data == "mnist":
    #     encoder = ImageEncoder(dim=get_dimension(data), hidden_dims=(16,128,128,128,128,16))
    denoiser = TimeInputMLP(dim=get_dimension(data), hidden_dims=(16,128,128,128,128,16))
    schedule = ScheduleLogLinear(N=200, sigma_min=0.005, sigma_max=10)
    if task == "train":
        if data == "swissroll":
            dataset = Swissroll(np.pi/2, 5*np.pi, 100)
            # plot
            datapoints = [dataset[i] for i in range(len(dataset))]
            plt.scatter([x[0] for x in datapoints], [x[1] for x in datapoints])
            plt.savefig("swissroll.png")
            plt.clf()
        elif data == "mnist":
            from torchvision import datasets
            dataset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
        loader  = DataLoader(dataset, batch_size=2048)
        losses  = training_loop(loader, denoiser, schedule, epochs=50000)
        plt.plot(np.convolve(losses, np.ones(100)/100, mode='valid'))
        plt.savefig("loss.png")
        torch.save(denoiser.state_dict(), f"{data}.pth")
    elif task == "sample":
        denoiser.load_state_dict(torch.load(f"{data}.pth"))
        batchsize = 2000
        xT = denoiser.rand_input(batchsize) 
        x0 = sample_diffusion(xT, denoiser, schedule)
        # plot
        plt.clf()
        plt.scatter(xT[:,0], xT[:,1])
        plt.savefig("data_init.png")
        plt.clf()
        plt.scatter(x0[:,0], x0[:,1])
        plt.savefig("data_diffusion.png")
    else:
        print("Invalid task")




if __name__ == "__main__":
    main()

