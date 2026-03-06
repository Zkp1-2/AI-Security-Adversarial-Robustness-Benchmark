import argparse
from pathlib import Path
from tqdm import tqdm
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models

from src.utils.io import load_config, ensure_dirs
from src.utils.metrics import accuracy
from src.attacks.fgsm import fgsm_attack
from src.attacks.pgd import pgd_attack

def get_test_loader(cfg):
    tf_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    test_ds = datasets.CIFAR10(
        root=cfg["data"]["data_dir"],
        train=False,
        download=True,
        transform=tf_test
    )

    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True
    )
    return test_loader

@torch.no_grad()
def eval_clean(model, loader, device):
    model.eval()
    total_acc, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_acc += accuracy(logits, y) * x.size(0)
        n += x.size(0)
    return total_acc / n

def eval_under_attack(model, loader, device, attack_fn):
    model.eval()
    total_acc, n = 0.0, 0

    for x, y in tqdm(loader, desc="Evaluating attack"):
        x, y = x.to(device), y.to(device)
        x_adv = attack_fn(model, x, y)

        with torch.no_grad():
            logits = model(x_adv)
            total_acc += accuracy(logits, y) * x.size(0)
            n += x.size(0)

    return total_acc / n

def load_model(ckpt_path: str, num_classes: int, device):
    checkpoint = torch.load(ckpt_path, map_location=device)
    model = models.resnet18(num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model

def main(cfg_path: str, ckpt_clean: str, ckpt_adv: str):
    cfg = load_config(cfg_path)
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    ensure_dirs("results/tables", "results/figures")
    loader = get_test_loader(cfg)

    eps = cfg["attack"]["eps"]
    alpha = cfg["attack"]["alpha"]
    steps = cfg["attack"]["pgd_steps"]

    rows = []

    for tag, ckpt in [("CleanTrain", ckpt_clean), ("AdvTrain_FGSM", ckpt_adv)]:
        model = load_model(ckpt, cfg["model"]["num_classes"], device)

        clean_acc = eval_clean(model, loader, device)

        fgsm_acc = eval_under_attack(
            model,
            loader,
            device,
            attack_fn=lambda m, x, y: fgsm_attack(m, x, y, eps=eps)
        )

        pgd_acc = eval_under_attack(
            model,
            loader,
            device,
            attack_fn=lambda m, x, y: pgd_attack(m, x, y, eps=eps, alpha=alpha, steps=steps)
        )

        rows.append({
            "Model": "ResNet18",
            "Training": tag,
            "CleanAcc": clean_acc,
            "FGSMAcc": fgsm_acc,
            "PGDAcc": pgd_acc
        })

    df = pd.DataFrame(rows)

    out_csv = Path("results/tables/robustness_table.csv")
    df.to_csv(out_csv, index=False)
    print("Saved:", out_csv)

    plot_df = df.set_index("Training")[["CleanAcc", "FGSMAcc", "PGDAcc"]]
    ax = plot_df.plot(kind="bar")
    ax.set_ylabel("Accuracy")
    ax.set_title("Robustness Benchmark (CIFAR-10, ResNet18)")
    plt.tight_layout()

    out_png = Path("results/figures/robustness_plot.png")
    plt.savefig(out_png, dpi=200)
    print("Saved:", out_png)

    print(df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/cifar10_resnet18.yaml")
    parser.add_argument("--ckpt_clean", type=str, default="results/resnet18_cifar10_clean.pth")
    parser.add_argument("--ckpt_adv", type=str, default="results/resnet18_cifar10_adv_fgsm.pth")
    args = parser.parse_args()
    main(args.config, args.ckpt_clean, args.ckpt_adv)