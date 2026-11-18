"""Bước 08b (Sơn): Random Forest (sklearn) trên 60 thành phần PCA, cho cả y_multi (macro-F1) và y_bin (AUC).
Đọc:  output/X_{train,test}_pca.npy, output/y_{train,test}.csv (cột y_multi, y_bin)
Ghi:  output/rf_grid.csv, output/rf_results.csv, output/rf_depth_curve.png,
      output/pred_multi_rf.npy, output/proba_val_rf.npy, output/proba_test_rf.npy
Cùng cách chia validation với 08a (phân tầng theo y_bin, random_state=42). max_features="sqrt".
Với mỗi max_depth chỉ fit một rừng 400 cây rồi lấy n cây đầu: kết quả giống hệt fit riêng RF(n_estimators=n)
vì cây thứ i chỉ phụ thuộc random_state và i (kiểm tra bằng assert)."""
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

TREES, DEPTHS = [50, 100, 200, 400], [4, 6, 8, 10, 12, 16, 20, None]
classes = np.array(["<30", ">30", "NO"])

y_tr_df, y_te_df = pd.read_csv("output/y_train.csv"), pd.read_csv("output/y_test.csv")
y_multi_all = y_tr_df["y_multi"].map({c: i for i, c in enumerate(classes)}).to_numpy()
y_multi_te = y_te_df["y_multi"].map({c: i for i, c in enumerate(classes)}).to_numpy()
y_bin_all, y_bin_te = y_tr_df["y_bin"].to_numpy(), y_te_df["y_bin"].to_numpy()
X_all, X_te = np.load("output/X_train_pca.npy"), np.load("output/X_test_pca.npy")
assert len(X_all) == len(y_bin_all) and len(X_te) == len(y_bin_te)

tr_idx, val_idx = train_test_split(np.arange(len(y_bin_all)), test_size=0.2, stratify=y_bin_all, random_state=42)
assert len(set(tr_idx) & set(val_idx)) == 0 and len(tr_idx) + len(val_idx) == len(y_bin_all)

make = lambda n, d: RandomForestClassifier(n_estimators=n, max_depth=d, random_state=42, n_jobs=-1)


def first_n(rf, n):
    """Rừng gồm n cây đầu của rf (sao chép nông, không đụng rf gốc)."""
    sub = RandomForestClassifier(random_state=42, n_jobs=-1)
    sub.__dict__.update(rf.__dict__)
    sub.estimators_ = rf.estimators_[:n]
    return sub


# self-check: n cây đầu của rừng 5 cây == rừng 3 cây
_a, _b = make(5, 4).fit(X_all[:2000], y_multi_all[:2000]), make(3, 4).fit(X_all[:2000], y_multi_all[:2000])
assert np.allclose(first_n(_a, 3).predict_proba(X_all[:500]), _b.predict_proba(X_all[:500]))

LABELS = {
    "y_multi": (y_multi_all, y_multi_te, lambda m, X, y: f1_score(y, m.predict(X), average="macro")),
    "y_bin": (y_bin_all, y_bin_te, lambda m, X, y: roc_auc_score(y, m.predict_proba(X)[:, 1])),
}

grid_rows, result_rows = [], []
for label, (y_all, y_te, metric) in LABELS.items():
    rows = []
    for d in DEPTHS:
        big = make(max(TREES), d).fit(X_all[tr_idx], y_all[tr_idx])
        for n in TREES:
            m = first_n(big, n)
            rows.append({"label": label, "n_estimators": n, "max_depth": str(d),
                         "train_metric": metric(m, X_all[tr_idx], y_all[tr_idx]),
                         "val_metric": metric(m, X_all[val_idx], y_all[val_idx])})
        print(label, "max_depth", d, "xong")
    grid_rows += rows
    best = max(rows, key=lambda r: r["val_metric"])
    n_best, d_best = best["n_estimators"], (None if best["max_depth"] == "None" else int(best["max_depth"]))
    print(label, best)

    m_tr = make(n_best, d_best).fit(X_all[tr_idx], y_all[tr_idx])
    final = make(n_best, d_best).fit(X_all, y_all)  # refit trên toàn bộ train
    res = {"label": label, "n_estimators": n_best, "max_depth": best["max_depth"],
           "train_metric": best["train_metric"], "val_metric": best["val_metric"]}
    if label == "y_multi":
        pred = final.predict(X_te)
        np.save("output/pred_multi_rf.npy", pred)
        res.update(test_accuracy=accuracy_score(y_te, pred), test_macro_f1=f1_score(y_te, pred, average="macro"))
    else:
        p_val, p_te = m_tr.predict_proba(X_all[val_idx])[:, 1], final.predict_proba(X_te)[:, 1]
        assert p_val.min() >= 0 and p_te.max() <= 1 and len(p_val) == len(val_idx) and len(p_te) == len(X_te)
        np.save("output/proba_val_rf.npy", p_val)
        np.save("output/proba_test_rf.npy", p_te)
        res.update(test_accuracy=accuracy_score(y_te, p_te >= 0.5), test_auc=roc_auc_score(y_te, p_te))
    result_rows.append(res)
    print(res)

grid = pd.DataFrame(grid_rows)
grid.round(4).to_csv("output/rf_grid.csv", index=False)
pd.DataFrame(result_rows).round(4).to_csv("output/rf_results.csv", index=False)

fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for j, (label, name) in enumerate([("y_multi", "macro-F1"), ("y_bin", "AUC")]):
    res = next(r for r in result_rows if r["label"] == label)
    by_depth = grid[(grid["label"] == label) & (grid["n_estimators"] == res["n_estimators"])]
    ax = axes[0, j]
    ax.plot(by_depth["max_depth"], by_depth["train_metric"], marker="o", label="train")
    ax.plot(by_depth["max_depth"], by_depth["val_metric"], marker="o", label="validation")
    ax.axvline(res["max_depth"], color="gray", linestyle="--", label=f"max_depth chọn = {res['max_depth']}")
    ax.set_title(f"Random Forest, {label} ({res['n_estimators']} cây)")
    ax.set_xlabel("max_depth"); ax.set_ylabel(name); ax.legend()
    by_n = grid[(grid["label"] == label) & (grid["max_depth"] == res["max_depth"])]
    ax = axes[1, j]
    ax.plot(by_n["n_estimators"], by_n["train_metric"], marker="o", label="train")
    ax.plot(by_n["n_estimators"], by_n["val_metric"], marker="o", label="validation")
    ax.set_title(f"Theo số cây (max_depth = {res['max_depth']})")
    ax.set_xlabel("n_estimators"); ax.set_ylabel(name); ax.legend()
fig.tight_layout()
fig.savefig("output/rf_depth_curve.png", dpi=150, bbox_inches="tight")
