import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models

from src.utils.seed import set_seed
from src.utils.io import load_config, ensure_dirs
from src.utils.metrics import accuracy
from src.attacks.pgd import pgd_attack

def get_loaders(cfg):
    tf_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    tf_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_ds = datasets.CIFAR10(root=cfg["data"]["data_dir"], train=True, download=True, transform=tf_train)
    test_ds = datasets.CIFAR10(root=cfg["data"]["data_dir"], train=False, download=True, transform=tf_test)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True
    )
    return train_loader, test_loader

@torch.no_grad()
def eval_clean(model, loader, device):
    model.eval()
    total_acc = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_acc += accuracy(logits, y) * x.size(0)
        n += x.size(0)
    return total_acc / n

def main(cfg_path):
    cfg = load_config(cfg_path)
    set_seed(cfg["seed"])

    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    ensure_dirs(cfg["train"]["save_dir"])

    train_loader, test_loader = get_loaders(cfg)

    model = models.resnet18(num_classes=cfg["model"]["num_classes"]).to(device)

    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg["train"]["lr"],
        momentum=cfg["train"]["momentum"],
        weight_decay=cfg["train"]["weight_decay"]
    )

    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[cfg["train"]["epochs"] // 2, (cfg["train"]["epochs"] * 3) // 4],
        gamma=0.1
    )

    criterion = nn.CrossEntropyLoss()
    eps = cfg["attack"]["eps"]
    alpha = cfg["attack"]["alpha"]
    steps = cfg["defense"]["pgd_train_steps"]

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"PGD Train [Epoch {epoch}/{cfg['train']['epochs']}]")

        for x, y in pbar:
            x, y = x.to(device), y.to(device)

            model.eval()
            x_adv = pgd_attack(model, x, y, eps=eps, alpha=alpha, steps=steps)
            model.train()

            optimizer.zero_grad(set_to_none=True)
            logits_adv = model(x_adv)
            loss = criterion(logits_adv, y)
            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=float(loss.item()))

        scheduler.step()
        clean_acc = eval_clean(model, test_loader, device)
        print(f"Epoch {epoch}: clean test acc = {clean_acc:.4f}")

    torch.save({"model_state": model.state_dict(), "cfg": cfg}, cfg["train"]["ckpt_pgd"])
    print("Saved:", cfg["train"]["ckpt_pgd"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/cifar10_resnet18.yaml")
    args = parser.parse_args()
    main(args.config)