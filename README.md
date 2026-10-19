# Quantum, Hybrid, and Classical ML for Banking Classification

Benchmark of quantum kernel, variational, and hybrid quantum-classical models against classical machine learning on three banking classification tasks: credit scoring, customer churn, and fraud detection. All quantum models run on noiseless statevector simulation, so results represent the ideal-case performance of each circuit at a given qubit count. The aim is a fair comparison of existing models under matched preprocessing and evaluation, not a new architecture.

## Datasets

| Task | Dataset | Samples | Features | Minority class | Source |
|---|---|---|---|---|---|
| Credit scoring | Statlog German Credit | 1,000 | 20 | 30.0% | [UCI](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data) |
| Customer churn | Bank Customer Churn (Churn_Modelling) | 10,000 | 10 | 20.4% | [Kaggle](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling) |
| Fraud detection | Credit Card Fraud (ULB) | 284,807 | 30 | 0.17% | [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |

The datasets are not redistributed here. Download them from the sources above and place `german.data`, `Churn_Modelling.csv`, and `creditcard.csv` in `data/raw/`.

German Credit and Bank Churn are used in full by every model. Bank Churn is also used for a learning-curve analysis with stratified training subsets of 500 to 8,000 samples. For fraud detection, legitimate transactions are undersampled in the training folds only; test folds keep the original fraud rate.

## Models

- **Quantum:** QSVC with a fidelity kernel, variational quantum classifier (VQC)
- **Hybrid:** dressed QNN (classical encoder, variational quantum layer, classical head)
- **Classical:** logistic regression, SVM (linear and RBF), random forest, XGBoost, LightGBM, MLP

Quantum models use 4, 6, and 8 qubits. Classical baselines are run both on the same reduced features given to the quantum models and on the full feature set. All models receive the same hyperparameter tuning budget.

## Evaluation

Stratified 5-fold cross-validation, with scaling, encoding, dimensionality reduction, and resampling fitted on the training folds only. Stochastic models are run with three seeds per fold. Reported metrics are ROC-AUC, PR-AUC, F1, recall, specificity, G-mean, and recall at fixed false positive rates for fraud. Resource figures include qubit count, circuit depth, two-qubit gate count, trainable parameters, and simulation time.

## Usage

```bash
pip install -r requirements.txt
pip install -e .
python -m qmlfin.check_data    # dataset and fold summary
pytest
```

The code runs on CPU by default. On a GPU machine, set `quantum.device` to `lightning.gpu` in `configs/default.yaml` and install `pennylane-lightning-gpu`.
