import torch.nn.functional as F


def crossentropy_loss(output, target, weight):
    return F.binary_cross_entropy(output, target, weight)