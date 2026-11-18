"""Bước 08a (Sơn): Decision Tree (sklearn) trên 60 thành phần PCA, cho cả y_multi (macro-F1) và y_bin (AUC).
Đọc:  output/X_{train,test}_pca.npy, output/y_{train,test}.csv (cột y_multi, y_bin)
Ghi:  output/dt_grid.csv, output/dt_results.csv, output/dt_depth_curve.png,
      output/pred_multi_dt.npy, output/proba_val_dt.npy, output/proba_test_dt.npy
Validation 20% tách từ train theo chỉ số, phân tầng theo y_bin, dùng chung cho mọi mô hình của nhóm.
Cây chỉ phụ thuộc thứ tự giá trị của từng cột nên không đổi khi chuẩn hóa -> dùng bản PCA chưa chuẩn hóa."""
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

DEPTHS, LEAFS = list(range(2, 21)) + [None], [1, 20, 100]
classes = np.array(["<30", ">30", "NO"])

y_tr_df, y_te_df = pd.read_csv("output/y_train.csv"), pd.read_csv("output/y_test.csv")
y_multi_all = y_tr_df["y_multi"].map({c: i for i, c in enumerate(classes)}).to_numpy()
y_multi_te = y_te_df["y_multi"].map({c: i for i, c in enumerate(classes)}).to_numpy()
y_bin_all, y_bin_te = y_tr_df["y_bin"].to_numpy(), y_te_df["y_bin"].to_numpy()
X_all, X_te = np.load("output/X_train_pca.npy"), np.load("output/X_test_pca.npy")
assert len(X_all) == len(y_bin_all) and len(X_te) == len(y_bin_te)

tr_idx, val_idx = train_test_split(np.arange(len(y_bin_all)), test_size=0.2, stratify=y_bin_all, random_state=42)
assert len(set(tr_idx) & set(val_idx)) == 0 and len(tr_idx) + len(val_idx) == len(y_bin_all)

# self-check: chuẩn hóa từng cột không làm đổi dự đoán của cây
mu, sd = X_all[tr_idx[:5000]].mean(0), X_all[tr_idx[:5000]].std(0)
t_raw = DecisionTreeClassifier(max_depth=5, random_state=42).fit(X_all[tr_idx[:5000]], y_multi_all[tr_idx[:5000]])
t_std = DecisionTreeClassifier(max_depth=5, random_state=42).fit((X_all[tr_idx[:5000]] - mu) / sd, y_multi_all[tr_idx[:5000]])
assert (t_raw.predict(X_all[val_idx[:2000]]) == t_std.predict((X_all[val_idx[:2000]] - mu) / sd)).all()

# label -> (nhãn train, nhãn test, độ đo chọn tham số)
LABELS = {
    "y_multi": (y_multi_all, y_multi_te, lambda m, X, y: f1_score(y, m.predict(X), average="macro")),
    "y_bin": (y_bin_all, y_bin_te, lambda m, X, y: roc_auc_score(y, m.predict_proba(X)[:, 1])),
}
make = lambda d, s: DecisionTreeClassifier(max_depth=d, min_samples_leaf=s, random_state=42)

grid_rows, result_rows = [], []
for label, (y_all, y_te, metric) in LABELS.items():
    rows = []
    for d in DEPTHS:
        for s in LEAFS:
            m = make(d, s).fit(X_all[tr_idx], y_all[tr_idx])
            rows.append({"label": label, "max_depth": str(d), "min_samples_leaf": s,
                         "train_metric": metric(m, X_all[tr_idx], y_all[tr_idx]),
                         "val_metric": metric(m, X_all[val_idx], y_all[val_idx])})
    grid_rows += rows
    best = max(rows, key=lambda r: r["val_metric"])
    d_best = None if best["max_depth"] == "None" else int(best["max_depth"])
    s_best = best["min_samples_leaf"]
    print(label, best)

    m_tr = make(d_best, s_best).fit(X_all[tr_idx], y_all[tr_idx])
    final = make(d_best, s_best).fit(X_all, y_all)  # refit trên toàn bộ train
    res = {"label": label, "max_depth": best["max_depth"], "min_samples_leaf": s_best,
           "train_metric": best["train_metric"], "val_metric": best["val_metric"]}
    if label == "y_multi":
        pred = final.predict(X_te)
        np.save("output/pred_multi_dt.npy", pred)
        res.update(test_accuracy=accuracy_score(y_te, pred), test_macro_f1=f1_score(y_te, pred, average="macro"))
    else:
        p_val, p_te = m_tr.predict_proba(X_all[val_idx])[:, 1], final.predict_proba(X_te)[:, 1]
        assert p_val.min() >= 0 and p_te.max() <= 1 and len(p_val) == len(val_idx) and len(p_te) == len(X_te)
        np.save("output/proba_val_dt.npy", p_val)
        np.save("output/proba_test_dt.npy", p_te)
        res.update(test_accuracy=accuracy_score(y_te, p_te >= 0.5), test_auc=roc_auc_score(y_te, p_te))
    result_rows.append(res)
    print(res)

grid = pd.DataFrame(grid_rows)
grid.round(4).to_csv("output/dt_grid.csv", index=False)
pd.DataFrame(result_rows).round(4).to_csv("output/dt_results.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, (label, name) in zip(axes, [("y_multi", "macro-F1"), ("y_bin", "AUC")]):
    sub = grid[(grid["label"] == label) & (grid["min_samples_leaf"] == 1)]
    best_depth = next(r for r in result_rows if r["label"] == label)["max_depth"]
    ax.plot(sub["max_depth"], sub["train_metric"], marker="o", label="train")
    ax.plot(sub["max_depth"], sub["val_metric"], marker="o", label="validation")
    ax.axvline(best_depth, color="gray", linestyle="--", label=f"max_depth chọn = {best_depth}")
    ax.set_title(f"Decision Tree, {label} (min_samples_leaf = 1)")
    ax.set_xlabel("max_depth"); ax.set_ylabel(name); ax.legend()
    ax.tick_params(axis="x", labelsize=7)
fig.tight_layout()
fig.savefig("output/dt_depth_curve.png", dpi=150, bbox_inches="tight")
