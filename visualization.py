
from matplotlib import pyplot as plt
import numpy as np


def plot_losses(losses, filename):
    plt.plot(np.convolve(losses, np.ones(100)/100, mode='valid'))
    plt.savefig(filename)
    plt.clf()