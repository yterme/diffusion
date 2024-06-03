
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch import nn

from data import get_dataset
from denoiser_schedule import Schedule, ScheduleLogLinear, generate_train_sample, get_sigma_embeds
from models import CNNEncoder, Denoiser, get_dimension
from visualization import plot_losses



def train_denoiser(loader  : DataLoader,
                  denoiser   : nn.Module,
                  schedule: Schedule,
                  epochs  : int,
                  ):
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

def main(
        # dataset_name="swissroll",
        dataset_name="mnist",
):
    if dataset_name == "mnist":
        encoder = CNNEncoder()
        encoder.load_state_dict(torch.load(f"models/{dataset_name}_encoder.pth"))
    else:
        encoder = None
    dataset = get_dataset(dataset_name, encoder)
    dataloader  = DataLoader(dataset, batch_size=64)
    denoiser = Denoiser(dim=get_dimension(dataset_name), hidden_dims=(16,128,128,128,128,16))
    schedule = ScheduleLogLinear(N=200, sigma_min=0.005, sigma_max=10)
    losses  = train_denoiser(dataloader, denoiser, schedule, epochs=1000)
    plot_losses(losses, f"visualizations/loss_denoiser_{dataset_name}.png")
    torch.save(denoiser.state_dict(), f"models/{dataset_name}_denoiser.pth")

if __name__ == "__main__":
    main()