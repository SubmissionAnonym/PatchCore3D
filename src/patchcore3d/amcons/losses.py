import torch


def kl_loss(mu, logvar):

    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return kl_divergence
