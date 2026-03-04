# SVM & LR Parameter Exploration

> **Course:** Machine Learning (Fall 2025) — University of Birmingham Dubai  
> **Difficulty:** Easy Tasks 1 & 2  
> **Algorithms:** Support Vector Machines (SVM) | Logistic Regression (LR)

---

## What This Project Does

This repo implements two practice tasks:

**Task 1 — Exploring SVM Parameters**  
Trains an SVM classifier across a grid of hyperparameters (kernels, C values, gamma) and compares their effect on accuracy. Produces a comparison table and plots.

**Task 2 — Comparing LR and SVM**  
Repeats the same experiment using Logistic Regression (varying penalty, C, solver) then directly compares LR vs SVM performance side by side.

---

## Project Structure

```
svm-lr-parameter-exploration/
├── src/
│   ├── preprocess.py   # Data loading, missing value handling, scaling, train/test split
│   ├── train.py        # SVM + LR parameter grid search
│   ├── evaluate.py     # Accuracy tables, plots, confusion matrices
│   └── main.py         # End-to-end runner (entry point)
├── results/            # Auto-generated CSVs and plots
├── notebooks/          # Jupyter notebooks for interactive exploration
├── requirements.txt
└── README.md
```

---

## Datasets

Built-in (no download needed):
- **Iris** — 150 samples, 4 features, 3 classes (setosa / versicolor / virginica)
- **Breast Cancer** — 569 samples, 30 features, 2 classes (malignant / benign)

Both are loaded directly from `sklearn.datasets`.  
You can also pass a CSV file path to use your own dataset.

---

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/svm-lr-parameter-exploration.git
cd svm-lr-parameter-exploration

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## How to Run

### Option A — Script (recommended)

```bash
# Run on Iris with 80/20 train-test split
python src/main.py --dataset iris --cv 0

# Run on Breast Cancer with 10-fold cross-validation
python src/main.py --dataset breast_cancer --cv 10
```

### Option B — Jupyter Notebook

```bash
jupyter notebook notebooks/
```

---

## SVM Parameters Explored

| Parameter | Values Tested |
|-----------|--------------|
| Kernel    | Linear, RBF, Polynomial |
| C (soft margin) | 0.01, 0.1, 1, 10, 100 |
| Gamma     | scale, auto |

## LR Parameters Explored

| Parameter | Values Tested |
|-----------|--------------|
| Penalty   | L1, L2 |
| C (inverse regularisation) | 0.01, 0.1, 1, 10, 100 |
| Solver    | lbfgs, liblinear |

---

## Outputs

All outputs are saved to `/results/`:

| File | Description |
|------|-------------|
| `svm_results_<dataset>.csv` | Full SVM accuracy table across all combos |
| `lr_results_<dataset>.csv` | Full LR accuracy table across all combos |
| `svm_train_vs_test_<dataset>.png` | Train vs test accuracy bar chart (SVM) |
| `lr_train_vs_test_<dataset>.png` | Train vs test accuracy bar chart (LR) |
| `svm_heatmap_<dataset>.png` | Heatmap of SVM accuracy across C × kernel |
| `lr_vs_svm_comparison_<dataset>.png` | Side-by-side LR vs SVM best accuracy |
| `svm_confusion_<dataset>.png` | Confusion matrix for best SVM config |
| `lr_confusion_<dataset>.png` | Confusion matrix for best LR config |

---

## Key Concepts

- **Kernel trick** — SVM maps data to higher dimensions to find a separating hyperplane
- **C parameter** — Controls the trade-off between a smooth margin and correctly classifying training points (low C = wider margin, high C = fewer misclassifications)
- **Regularisation (LR)** — L2 penalises large weights; L1 can drive weights to zero (feature selection)
- **Overfitting** — When train accuracy >> test accuracy; visible in the train-vs-test plots
- **Cross-validation** — More robust accuracy estimate; avoids lucky/unlucky splits

---

## Reference

Course materials by Dr. Fawad Hussain, University of Birmingham Dubai (Fall 2025)