# Playing Card Recognition

An academic computer vision project (ICD/ADS) that classifies a playing card from a single
image across 53 classes (52 cards plus the joker), using transfer learning with EfficientNet-B0
in PyTorch.

The code, documentation and experiment structure are complete. Training runs in Google Colab
through the notebook [`notebooks/treino_cartas_colab.ipynb`](notebooks/treino_cartas_colab.ipynb).
The main model reaches a measured test accuracy of 94.7 percent on the Kaggle test set
(Colab, GPU T4, seed 42).

## Tech stack

- Python 3
- PyTorch and torchvision (transfer learning with pretrained backbones)
- scikit-learn and scikit-image (classical baseline and metrics)
- NumPy, pandas, Pillow (data and image handling)
- Matplotlib and seaborn (plots and confusion matrices)
- kagglehub (dataset download)
- Google Colab and Jupyter notebook (training environment)

## What it does

- Classifies a playing card image into one of 53 classes (52 cards plus joker).
- Trains a transfer learning model (EfficientNet-B0 by default) in two phases: feature
  extraction with a frozen backbone, then fine-tuning of the full network with cosine
  annealing and early stopping.
- Supports alternative backbones (MobileNet V3 small/large, ResNet18) through a single
  configuration flag.
- Provides a classical baseline (HOG features plus logistic regression or SVM) to compare
  against the deep model.
- Runs controlled experiments: feature extraction versus fine-tuning, with versus without
  data augmentation, and an out-of-distribution (OOD) check against a deck of a different
  visual design.
- Produces evaluation metrics, a per-class classification report, and confusion matrices
  (see the `reports/` folder).
- Runs single-image inference with top-k predictions from a trained checkpoint.

## Results

| Model | Test accuracy | Macro F1 | OOD accuracy |
|-------|---------------|----------|--------------|
| Baseline (HOG + logistic regression) | 70.6% | 0.698 | not measured |
| EfficientNet-B0 (feature extraction) | 38.5% | 0.363 | not measured |
| EfficientNet-B0 (fine-tuning), main model | 94.7% | 0.947 | 59.3% |

The main model is EfficientNet-B0 with fine-tuning and data augmentation. The OOD column
refers to a deck of a different visual design (clean web images), which measures the design
gap rather than a real capture gap (photos with lighting, shadow and background variation),
left as future work. Validation and test sets are small (5 images per class), so metrics
should be read with wide uncertainty, which is why macro F1 and the confusion matrix are
prioritized. See [`docs/RELATORIO_DESENVOLVIMENTO.md`](docs/RELATORIO_DESENVOLVIMENTO.md)
for the full analysis.

## Dataset

[Cards Image Dataset-Classification (gpiosenka)](https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification):
53 classes, images already cropped to 224x224, with a fixed train/valid/test split
(7,624 / 265 / 265). Download instructions are in [`data/README.md`](data/README.md).

For the OOD experiment, a deck of a different design is assembled from freely licensed clean
images through the reproducible script `src/ood_design.py`, documented in
[`docs/guia_ood_design_web.md`](docs/guia_ood_design_web.md).

## Running locally

Training on CPU is slow. Google Colab with a GPU is recommended for the full pipeline.

Clone the repository:

```bash
git clone https://github.com/nicolasfvp/playing-card-recognition.git
cd playing-card-recognition
```

### Option 1: Google Colab (recommended, free GPU)

1. Open [`notebooks/treino_cartas_colab.ipynb`](notebooks/treino_cartas_colab.ipynb) in Colab.
2. Set the runtime hardware to GPU (T4).
3. Edit the `REPO_URL` variable in the first cell to point to your repository.
4. Run the cells in order (setup, data, baseline, training, evaluation, experiments, OOD).

### Option 2: Local (CPU)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt

# Classical baseline (fast on CPU):
python -m src.baseline --data-dir data/raw/cards --max-per-class 80

# Training (slow on CPU; Colab is recommended):
python -m src.train --data-dir data/raw/cards --backbone efficientnet_b0

# Inference on a single image:
python -m src.predict path/to/card.jpg --checkpoint models/efficientnet_b0_best.pt
```

Reproducibility is handled by a fixed seed (42) across the pipeline, pinned dependency
versions in `requirements.txt`, and the dataset fixed split.

## Project structure

```
src/                    Source code (Python package)
  config.py             Central configuration (hyperparameters, paths)
  seed.py               Reproducibility (set_seed)
  data.py               Dataloaders and transforms (PyTorch)
  model.py              Transfer learning backbones (EfficientNet-B0, MobileNet, ResNet)
  baseline.py           Classical baseline (HOG + logistic regression or SVM)
  train.py              Two-phase training with early stopping
  evaluate.py           Metrics, confusion matrix, OOD evaluation
  ood_design.py         Builds the different-design OOD set
  predict.py            Single-image inference
notebooks/              Google Colab training notebook (full pipeline)
models/                 Trained checkpoints (not versioned)
data/                   Download instructions only (see data/README.md)
docs/                   Problem definition, data, ethics, model card, development report
reports/                Results, confusion matrices, classification report, experiments csv
requirements.txt
LICENSE                 MIT
```

## Documentation

- [Problem definition and requirements](docs/01_definicao_problema.md)
- [Data and preparation](docs/02_dados.md)
- [Ethics and impact assessment](docs/03_etica_impacto.md)
- [Model card](docs/MODEL_CARD.md)
- [Development report](docs/RELATORIO_DESENVOLVIMENTO.md)
- [Different-design OOD set guide](docs/guia_ood_design_web.md)
- [Real deck collection guide (capture gap, future work)](docs/guia_coleta_baralho_real.md)
- [Final report outline](reports/relatorio_final_outline.md)

Note: the documents above are written in Portuguese.

## Status

Academic project developed for an AI/ML/DL course (ICD/ADS). Code, documentation and
experiments are complete; training is executed in Google Colab.

## License

Released under the MIT License. See [LICENSE](LICENSE). The Kaggle dataset has its own
license (marked as "Other"), so confirm the terms before redistributing the images.
