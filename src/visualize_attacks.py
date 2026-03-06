import argparse
from pathlib import Path
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models

from src.utils.io import load_config, ensure_dirs
from src.attacks.fgsm import fgsm_attack
from src.attacks.pgd import pgd_attack

CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

def load_model(ckpt_path, num_classes, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = models.resnet18(num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model

def main(cfg_path, ckpt_path):
    cfg = load_config(cfg_path)
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    ensure_dirs("results/figures")

    tf = transforms.Compose([transforms.ToTensor()])
    ds = datasets.CIFAR10(root=cfg["data"]["data_dir"], train=False, download=True, transform=tf)

    model = load_model(ckpt_path, cfg["model"]["num_classes"], device)

    eps = cfg["attack"]["eps"]
    alpha = cfg["attack"]["alpha"]
    steps = cfg["attack"]["pgd_steps"]

    fig, axes = plt.subplots(5, 3, figsize=(9, 12))

    shown = 0
    idx = 0
    while shown < 5 and idx < len(ds):
        x, y = ds[idx]
        idx += 1

        x = x.unsqueeze(0).to(device)
        y_tensor = torch.tensor([y], device=device)

        with torch.no_grad():
            pred_clean = model(x).argmax(dim=1).item()

        x_fgsm = fgsm_attack(model, x, y_tensor, eps=eps)
        x_pgd = pgd_attack(model, x, y_tensor, eps=eps, alpha=alpha, steps=steps)

        with torch.no_grad():
            pred_fgsm = model(x_fgsm).argmax(dim=1).item()
            pred_pgd = model(x_pgd).argmax(dim=1).item()

        if pred_clean == y:
            imgs = [
                x[0].cpu().permute(1, 2, 0).numpy(),
                x_fgsm[0].cpu().permute(1, 2, 0).numpy(),
                x_pgd[0].cpu().permute(1, 2, 0).numpy(),
            ]
            titles = [
                f"Original\nT:{CLASSES[y]} P:{CLASSES[pred_clean]}",
                f"FGSM\nT:{CLASSES[y]} P:{CLASSES[pred_fgsm]}",
                f"PGD\nT:{CLASSES[y]} P:{CLASSES[pred_pgd]}",
            ]

            for j in range(3):
                axes[shown, j].imshow(imgs[j])
                axes[shown, j].set_title(titles[j], fontsize=9)
                axes[shown, j].axis("off")

            shown += 1

    plt.tight_layout()
    out_path = Path("results/figures/adversarial_examples.png")
    plt.savefig(out_path, dpi=200)
    print("Saved:", out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/cifar10_resnet18.yaml")
    parser.add_argument("--ckpt", type=str, default="results/resnet18_cifar10_clean.pth")
    args = parser.parse_args()
    main(args.config, args.ckpt)