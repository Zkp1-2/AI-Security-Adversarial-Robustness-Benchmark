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
    tf_test = transforms.Compose([transforms.ToTensor()])
    test_ds = datasets.CIFAR10(
        root=cfg["data"]["data_dir"],
        train=False,
        download=True,
        transform=tf_test
    )
    return torch.utils.data.DataLoader(
        test_ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True
    )

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
    for x, y in tqdm(loader, desc="Evaluating"):
        x, y = x.to(device), y.to(device)
        x_adv = attack_fn(model, x, y)
        with torch.no_grad():
            logits = model(x_adv)
            total_acc += accuracy(logits, y) * x.size(0)
            n += x.size(0)
    return total_acc / n

def load_model(ckpt_path, num_classes, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = models.resnet18(num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model

def main(cfg_path, ckpts):
    cfg = load_config(cfg_path)
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    ensure_dirs("results/tables", "results/figures")

    loader = get_test_loader(cfg)
    eps_list = cfg["attack"]["eps_list"]
    alpha = cfg["attack"]["alpha"]
    steps = cfg["attack"]["pgd_steps"]

    rows = []

    for tag, ckpt in ckpts:
        model = load_model(ckpt, cfg["model"]["num_classes"], device)

        clean_acc = eval_clean(model, loader, device)
        rows.append({
            "Model": "ResNet18",
            "Training": tag,
            "Attack": "Clean",
            "Epsilon": 0.0,
            "Accuracy": clean_acc
        })

        for eps in eps_list:
            fgsm_acc = eval_under_attack(
                model, loader, device,
                attack_fn=lambda m, x, y, eps=eps: fgsm_attack(m, x, y, eps=eps)
            )
            rows.append({
                "Model": "ResNet18",
                "Training": tag,
                "Attack": "FGSM",
                "Epsilon": eps,
                "Accuracy": fgsm_acc
            })

            pgd_acc = eval_under_attack(
                model, loader, device,
                attack_fn=lambda m, x, y, eps=eps: pgd_attack(m, x, y, eps=eps, alpha=alpha, steps=steps)
            )
            rows.append({
                "Model": "ResNet18",
                "Training": tag,
                "Attack": "PGD",
                "Epsilon": eps,
                "Accuracy": pgd_acc
            })

    df = pd.DataFrame(rows)
    out_csv = Path("results/tables/epsilon_sweep.csv")
    df.to_csv(out_csv, index=False)
    print("Saved:", out_csv)

    plt.figure(figsize=(8, 5))
    for training in df["Training"].unique():
        for attack in ["FGSM", "PGD"]:
            sub = df[(df["Training"] == training) & (df["Attack"] == attack)].sort_values("Epsilon")
            plt.plot(sub["Epsilon"], sub["Accuracy"], marker="o", label=f"{training}_{attack}")

    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.title("Robustness Curve under Different Attack Strengths")
    plt.legend()
    plt.tight_layout()

    out_png = Path("results/figures/epsilon_sweep_curve.png")
    plt.savefig(out_png, dpi=200)
    print("Saved:", out_png)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/cifar10_resnet18.yaml")
    parser.add_argument("--ckpt_clean", type=str, default="results/resnet18_cifar10_clean.pth")
    parser.add_argument("--ckpt_fgsm", type=str, default="results/resnet18_cifar10_adv_fgsm.pth")
    parser.add_argument("--ckpt_pgd", type=str, default="results/resnet18_cifar10_adv_pgd.pth")
    args = parser.parse_args()

    ckpts = [
        ("CleanTrain", args.ckpt_clean),
        ("AdvTrain_FGSM", args.ckpt_fgsm),
        ("AdvTrain_PGD", args.ckpt_pgd),
    ]
    main(args.config, ckpts)