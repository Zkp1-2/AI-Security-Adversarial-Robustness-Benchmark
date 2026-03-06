import torch
import torch.nn.functional as F

def pgd_attack(model, x, y, eps: float, alpha: float, steps: int):
    x_orig = x.clone().detach()
    x_adv = x_orig + torch.empty_like(x_orig).uniform_(-eps, eps)
    x_adv = torch.clamp(x_adv, 0.0, 1.0).detach()

    for _ in range(steps):
        x_adv.requires_grad_(True)
        logits = model(x_adv)
        loss = F.cross_entropy(logits, y)
        loss.backward()

        grad_sign = x_adv.grad.detach().sign()
        x_adv = x_adv + alpha * grad_sign

        delta = torch.clamp(x_adv - x_orig, min=-eps, max=eps)
        x_adv = torch.clamp(x_orig + delta, 0.0, 1.0).detach()

    return x_adv