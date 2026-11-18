"""Bước 08 (Sơn): Decision Tree và Random Forest (sklearn) trên 60 thành phần PCA, bài toán multi-class y_multi.
Đọc:  output/X_{train,test}_pca.npy, output/y_{train,test}.csv (cột y_multi)
Ghi:  output/dt_rf_results.csv, output/dt_grid.csv, output/rf_grid.csv, output/dt_rf_curves.png,
      output/pred_dt.npy, output/pred_rf.npy
Cùng giao thức với 07a/07b: validation 80/20 tách từ train (macro-F1), refit trên toàn bộ train, test chỉ để báo cáo.
Cây chỉ phụ thuộc thứ tự giá trị của từng cột nên không đổi khi chuẩn hóa -> chỉ dùng bản PCA chưa chuẩn hóa."""
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

DT_DEPTHS, DT_LEAFS = [3, 5, 7, 10, 15, 20, None], [1, 10, 50]
RF_TREES, RF_DEPTHS = [50, 100, 200, 300], [5, 10, 15, 20, None]
classes = np.array(["<30", ">30", "NO"])


def macro_f1(y, p):
    return f1_score(y, p, average="macro")


y_all = pd.read_csv("output/y_train.csv")["y_multi"].map({c: i for i, c in enumerate(classes)}).to_numpy()
y_te = pd.read_csv("output/y_test.csv")["y_multi"].map({c: i for i, c in enumerate(classes)}).to_numpy()
X_all, X_te = np.load("output/X_train_pca.npy"), np.load("output/X_test_pca.npy")
assert len(X_all) == len(y_all) and len(X_te) == len(y_te)
X_tr, X_val, y_tr, y_val = train_test_split(X_all, y_all, test_size=0.2, stratify=y_all, random_state=42)

# self-check: chuẩn hóa từng cột không làm đổi dự đoán của cây
mu, sd = X_tr[:5000].mean(0), X_tr[:5000].std(0)
t_raw = DecisionTreeClassifier(max_depth=5, random_state=42).fit(X_tr[:5000], y_tr[:5000])
t_std = DecisionTreeClassifier(max_depth=5, random_state=42).fit((X_tr[:5000] - mu) / sd, y_tr[:5000])
assert (t_raw.predict(X_val[:2000]) == t_std.predict((X_val[:2000] - mu) / sd)).all()


def run_grid(make_model, grid):
    rows = []
    for params in grid:
        m = make_model(**params).fit(X_tr, y_tr)
        rows.append({**{k: str(v) for k, v in params.items()},
                     "train_macro_f1": macro_f1(y_tr, m.predict(X_tr)),
                     "val_macro_f1": macro_f1(y_val, m.predict(X_val))})
        print(rows[-1])
    return pd.DataFrame(rows)


dt_params = [{"max_depth": d, "min_samples_leaf": s} for d in DT_DEPTHS for s in DT_LEAFS]
rf_params = [{"n_estimators": n, "max_depth": d} for n in RF_TREES for d in RF_DEPTHS]
dt_grid = run_grid(lambda **p: DecisionTreeClassifier(random_state=42, **p), dt_params)
rf_grid = run_grid(lambda **p: RandomForestClassifier(random_state=42, n_jobs=-1, **p), rf_params)
dt_grid.round(4).to_csv("output/dt_grid.csv", index=False)
rf_grid.round(4).to_csv("output/rf_grid.csv", index=False)

rows = []
for name, make_model, params, grid in [
    ("dt", lambda **p: DecisionTreeClassifier(random_state=42, **p), dt_params, dt_grid),
    ("rf", lambda **p: RandomForestClassifier(random_state=42, n_jobs=-1, **p), rf_params, rf_grid),
]:
    best = int(grid["val_macro_f1"].idxmax())
    pred = make_model(**params[best]).fit(X_all, y_all).predict(X_te)  # refit trên toàn bộ train
    np.save(f"output/pred_{name}.npy", pred)  # nhãn dự đoán test, bước 10 đọc để đánh giá
    rows.append({"model": name, "best_params": params[best], "val_macro_f1": grid["val_macro_f1"].max(),
                 "test_accuracy": accuracy_score(y_te, pred), "test_macro_f1": macro_f1(y_te, pred),
                 **{f"test_f1_{c}": f for c, f in zip(classes, f1_score(y_te, pred, average=None, labels=range(3)))}})
    print(rows[-1])
pd.DataFrame(rows).round(4).to_csv("output/dt_rf_results.csv", index=False)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
best_leaf = dt_params[int(dt_grid["val_macro_f1"].idxmax())]["min_samples_leaf"]
sub = dt_grid[dt_grid["min_samples_leaf"] == str(best_leaf)]
a1.plot(sub["max_depth"], sub["train_macro_f1"], marker="o", label="train")
a1.plot(sub["max_depth"], sub["val_macro_f1"], marker="o", label="validation")
a1.set_title(f"Decision Tree (min_samples_leaf = {best_leaf})")
for n in RF_TREES:
    s = rf_grid[rf_grid["n_estimators"] == str(n)]
    a2.plot(s["max_depth"], s["val_macro_f1"], marker="o", label=f"{n} cây")
a2.set_title("Random Forest (validation)")
for ax in (a1, a2):
    ax.set_xlabel("max_depth"); ax.set_ylabel("macro-F1"); ax.legend()
fig.tight_layout()
fig.savefig("output/dt_rf_curves.png", dpi=150)
