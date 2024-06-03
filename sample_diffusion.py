
from matplotlib import pyplot as plt
import torch

from denoiser_schedule import ScheduleLogLinear, get_sigma_embeds
from models import CNNDecoder, Denoiser, get_dimension


def sample_diffusion(denoiser, schedule, batch_size=100):
    xT = denoiser.rand_input(batch_size)
    sigmas = schedule.sample_sigmas(20)
    xt = xT * sigmas[0]
    for sig, sig_prev in zip(sigmas, sigmas[1:]):
        # replicate sigma so that it has shape (batchsize,)
        sig_batch = sig.repeat(xt.shape[0])
        sigma_embeds = get_sigma_embeds(sig_batch.to(xt))
        with torch.no_grad():
            eps = denoiser(xt, sigma_embeds)
        xt -= (sig - sig_prev) * eps
    return xT, xt


def main(
        # dataset_name="swissroll",
        dataset_name="mnist",
        batch_size=100
):     
    denoiser = Denoiser(dim=get_dimension(dataset_name), hidden_dims=(16,128,128,128,128,16))
    denoiser.load_state_dict(torch.load(f"models/{dataset_name}_denoiser.pth"))

    schedule = ScheduleLogLinear(N=200, sigma_min=0.005, sigma_max=10)
    xT, x0 = sample_diffusion(denoiser, schedule, batch_size)
    if dataset_name == "spirale":
        plt.scatter(xT[:,0], xT[:,1])
        plt.savefig("visualizations/data_init.png")
        plt.clf()
        plt.scatter(x0[:,0], x0[:,1])
        plt.savefig("visualizations/data_diffusion.png")
        plt.clf()
    else:
        decoder = CNNDecoder()
        decoder_path = f"models/{dataset_name}_decoder.pth"
        decoder.load_state_dict(torch.load(decoder_path))
        plt.clf()
        xT_image = decoder(xT)
        x0_image = decoder(x0)
        num_images = 10
        fig, axs = plt.subplots(2, num_images)
        for i in range(num_images):
            axs[0, i].imshow(xT_image[i].view(16, 16).detach().numpy())
            axs[1, i].imshow(x0_image[i].view(16, 16).detach().numpy())
        plt.savefig("visualizations/images.png")



if __name__ == "__main__":
    main()