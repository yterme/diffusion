
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from data import get_dataset
from models import CNNDecoder, CNNEncoder
from visualization import plot_losses


def train_autoencoder(
        dataloader,
        encoder,
        decoder,
        epochs = 100
):
    losses = []
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()))
    for epoch in tqdm(range(epochs)):
        for x in dataloader:
            optimizer.zero_grad()
            x_hat = decoder(encoder(x))
            loss = torch.nn.functional.mse_loss(x, x_hat)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
    return losses


def main(
        # dataset_name="swissroll",
        dataset_name="mnist",
        load_autoencoder=True,
):
    assert dataset_name != "swissroll"
    encoder = CNNEncoder()
    decoder = CNNDecoder()
    encoder_path = f"models/{dataset_name}_encoder.pth"
    decoder_path = f"models/{dataset_name}_decoder.pth"
    if load_autoencoder:
        encoder.load_state_dict(torch.load(encoder_path))
        decoder.load_state_dict(torch.load(decoder_path))
    
    dataset = get_dataset(dataset_name)
    dataloader  = DataLoader(dataset, batch_size=64)
    losses = train_autoencoder(dataloader, encoder, decoder)
    plot_losses(losses, "visualizations/loss_autoencoder.png")
    # save the encoder and decoder
    torch.save(encoder.state_dict(), encoder_path)
    torch.save(decoder.state_dict(), decoder_path)

if __name__ == "__main__":
    main()