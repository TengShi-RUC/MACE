import torch


def getLossReg(gradx, batch_x_d):
    if gradx == None:
        return 0
    grad_inn = (gradx * batch_x_d).sum(1)
    E_grad = grad_inn.mean(0)
    loss_reg = torch.abs(E_grad)
    return loss_reg


def getMix(batch_0, batch_1, gamma):
    batch_mix = batch_0 * gamma + batch_1 * (1 - gamma)
    batch_mix = batch_mix.requires_grad_(True)
    return batch_mix
