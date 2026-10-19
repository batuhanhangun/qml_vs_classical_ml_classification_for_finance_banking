# Quantum, Hybrid, and Classical ML for Banking Classification

Benchmark of pure quantum, hybrid quantum-classical, and classical machine learning models on three banking classification tasks: credit scoring, customer churn, and fraud detection. The aim is a fair comparison of existing models under matched preprocessing and evaluation, not a new architecture.

## Datasets

| Task | Dataset | Samples | Features | Minority class | Source |
|---|---|---|---|---|---|
| Credit scoring | Statlog German Credit | 1,000 | 20 | 30.0% | [UCI](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data) |
| Customer churn | Bank Customer Churn (Churn_Modelling) | 10,000 | 10 | 20.4% | [Kaggle](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling) |
| Fraud detection | Credit Card Fraud (ULB) | 284,807 | 30 | 0.17% | [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |

The datasets are not redistributed here; download them from the sources above.

German Credit is used in full. On the churn data, exact QSVC runs on a stratified 2,000-sample subset, and the other models also run on the full set. The fraud experiments keep all 492 fraud cases together with a stratified sample of legitimate transactions.

## Models

- **Quantum:** QSVC (fidelity kernel), Pegasos-QSVC, variational quantum classifier (VQC)
- **Hybrid:** dressed QNN (classical encoder, variational quantum layer, classical head)
- **Classical:** logistic regression, SVM (linear and RBF), random forest, XGBoost, LightGBM, MLP

Quantum models use 4, 6, and 8 qubits. Classical baselines are run both on the same reduced features given to the quantum models and on the full feature set.

## Evaluation

Stratified 5-fold cross-validation, with scaling, encoding, dimensionality reduction, and resampling fitted on the training folds only. Reported metrics are ROC-AUC, PR-AUC, F1, recall, specificity, G-mean, and training and inference time.
