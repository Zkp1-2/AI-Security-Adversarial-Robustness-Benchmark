import torch
import torch.nn.functional as F

def fgsm_attack(model, x, y, eps: float):
    x_adv = x.clone().detach().requires_grad_(True)
    logits = model(x_adv)
    loss = F.cross_entropy(logits, y)
    loss.backward()

    grad_sign = x_adv.grad.detach().sign()
    x_adv = x_adv + eps * grad_sign
    x_adv = torch.clamp(x_adv, 0.0, 1.0).detach()
    return x_adv