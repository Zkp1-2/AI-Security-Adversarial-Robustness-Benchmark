# Adversarial Attacks & Defenses Benchmark (CIFAR-10)

This project benchmarks the robustness of deep neural networks under adversarial attacks (FGSM, PGD) and evaluates a defense strategy using FGSM adversarial training.

## Methods
- Dataset: CIFAR-10
- Model: ResNet-18
- Attacks: FGSM, PGD
- Defense: FGSM adversarial training
- Metrics: Clean accuracy, FGSM accuracy, PGD accuracy

## Attack Settings
- epsilon = 8/255
- alpha = 2/255
- PGD steps = 10

## Results
Results will be saved to:
- `results/tables/robustness_table.csv`
- `results/figures/robustness_plot.png`

## How to Run

Install dependencies:
```bash
pip install -r requirements.txt