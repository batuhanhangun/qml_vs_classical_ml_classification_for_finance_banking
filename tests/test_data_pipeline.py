import numpy as np
import pandas as pd
import pytest

from qmlfin import data, splits
from qmlfin.preprocess import make_preprocessor

rng = np.random.default_rng(0)


@pytest.fixture
def raw_dir(tmp_path):
    """Small synthetic files in the same formats as the real datasets."""
    n = 200
    # German Credit: 20 space-separated attributes, target 1 = good, 2 = bad
    num = {2, 5, 8, 11, 13, 16, 18}
    rows = []
    for _ in range(n):
        attrs = [str(rng.integers(1, 50)) if i in num else f"A{i}{rng.integers(1, 4)}" for i in range(1, 21)]
        rows.append(" ".join(attrs + [str(rng.choice([1, 2], p=[0.7, 0.3]))]))
    (tmp_path / "german.data").write_text("\n".join(rows))

    pd.DataFrame({
        "RowNumber": range(n), "CustomerId": range(n), "Surname": ["x"] * n,
        "CreditScore": rng.integers(350, 850, n), "Geography": rng.choice(["France", "Spain", "Germany"], n),
        "Gender": rng.choice(["Male", "Female"], n), "Age": rng.integers(18, 90, n), "Tenure": rng.integers(0, 11, n),
        "Balance": rng.random(n) * 1e5, "NumOfProducts": rng.integers(1, 5, n), "HasCrCard": rng.integers(0, 2, n),
        "IsActiveMember": rng.integers(0, 2, n), "EstimatedSalary": rng.random(n) * 1e5,
        "Exited": rng.choice([0, 1], n, p=[0.8, 0.2]),
    }).to_csv(tmp_path / "Churn_Modelling.csv", index=False)

    m = 5000
    ulb = pd.DataFrame(rng.normal(size=(m, 28)), columns=[f"V{i}" for i in range(1, 29)])
    ulb.insert(0, "Time", np.arange(m))
    ulb["Amount"] = rng.random(m) * 100
    ulb["Class"] = 0
    ulb.loc[rng.choice(m, 50, replace=False), "Class"] = 1
    ulb.to_csv(tmp_path / "creditcard.csv", index=False)
    return tmp_path


def test_loaders(raw_dir):
    g = data.load("german_credit", raw_dir)
    assert g.X.shape == (200, 20) and len(g.cat_cols) == 13 and len(g.num_cols) == 7
    c = data.load("bank_churn", raw_dir)
    assert c.X.shape == (200, 10) and "Surname" not in c.X
    u = data.load("ulb_fraud", raw_dir)
    assert u.X.shape == (5000, 30) and u.y.sum() == 50
    for ds in (g, c, u):
        assert set(np.unique(ds.y)) <= {0, 1}


def test_missing_file_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="kaggle"):
        data.load("ulb_fraud", tmp_path)


def test_folds_are_stratified_and_disjoint():
    y = np.array([1] * 30 + [0] * 70)
    folds = splits.outer_folds(y, 5, 42)
    test_all = np.concatenate([te for _, te in folds])
    assert sorted(test_all) == list(range(100))
    for tr, te in folds:
        assert not set(tr) & set(te)
        assert y[te].sum() == 6


def test_undersampling_touches_train_only():
    y = np.array([1] * 50 + [0] * 4950)
    tr, te = splits.outer_folds(y, 5, 42)[0]
    kept = splits.undersample_train(tr, y, 4, 0)
    assert set(kept) <= set(tr)
    assert y[kept].sum() == y[tr].sum()                    # every training fraud kept
    assert (y[kept] == 0).sum() == 4 * y[tr].sum()
    assert y[te].mean() == pytest.approx(0.01)             # test fold keeps the natural rate


def test_learning_curve_subsets_nested_and_stratified():
    y = np.array([1] * 200 + [0] * 800)
    tr, _ = splits.outer_folds(y, 5, 42)[0]
    subs = splits.learning_curve_subsets(tr, y, [100, 200, 400, 5000], 0)
    assert list(subs) == [100, 200, 400]                   # sizes above the fold size are skipped
    assert set(subs[100]) <= set(subs[200]) <= set(subs[400])
    for n, idx in subs.items():
        assert len(idx) == n and y[idx].mean() == pytest.approx(0.2, abs=0.01)


def test_preprocessor_quantum_and_full(raw_dir):
    ds = data.load("bank_churn", raw_dir)
    tr, te = splits.outer_folds(ds.y, 5, 42)[0]
    q = make_preprocessor(ds.cat_cols, ds.num_cols, n_components=4, angle_range=(0, np.pi))
    Xtr = q.fit_transform(ds.X.iloc[tr])
    Xte = q.transform(ds.X.iloc[te])
    assert Xtr.shape[1] == Xte.shape[1] == 4
    assert Xte.min() >= 0 and Xte.max() <= np.pi
    full = make_preprocessor(ds.cat_cols, ds.num_cols).fit(ds.X.iloc[tr])
    assert full.transform(ds.X.iloc[te]).shape[1] == 13      # 3 + 2 one-hot columns + 8 numeric


def test_preprocessor_fitted_on_train_only(raw_dir):
    ds = data.load("bank_churn", raw_dir)
    tr, te = splits.outer_folds(ds.y, 5, 42)[0]
    pre = make_preprocessor(ds.cat_cols, ds.num_cols).fit(ds.X.iloc[tr])
    scaler = pre.named_steps["standardise"]
    encoded_train = pre.named_steps["encode"].transform(ds.X.iloc[tr])
    np.testing.assert_allclose(scaler.mean_, encoded_train.mean(axis=0))


def test_ulb_duplicates_dropped(raw_dir):
    df = pd.read_csv(raw_dir / "creditcard.csv")
    pd.concat([df, df.iloc[:10]]).to_csv(raw_dir / "creditcard.csv", index=False)
    assert len(data.load("ulb_fraud", raw_dir).y) == 5000
    assert len(data.load("ulb_fraud", raw_dir, drop_duplicates=False).y) == 5010
