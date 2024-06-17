
from matplotlib import pyplot as plt
import numpy as np


def plot_losses(losses, filename, n_window=100):
    plt.plot(np.convolve(losses, np.ones(n_window)/n_window, mode='valid'))
    plt.savefig(filename)
    plt.clf()