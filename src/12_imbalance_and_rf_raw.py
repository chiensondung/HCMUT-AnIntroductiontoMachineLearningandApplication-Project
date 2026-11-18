"""Bước 12 (Chiến): ảnh hưởng của Class Imbalance (4a) và Random Forest trên One-Hot chưa PCA (4b).
Đọc:  output/y_{train,test}.csv, output/train.csv (tên cột gốc),
      output/X_{train,test}_pca.npy, output/rf_results.csv, output/proba_{val,test}_rf.npy (Task 1, Sơn),
      output/X_{train,test}_onehot.npz, output/onehot_columns.txt (bước 04)
Ghi:  output/imbalance_distribution.csv, output/imbalance_rf_multi.csv, output/imbalance_threshold.csv,
      output/rf_raw_results.csv, output/rf_raw_importance_onehot.csv, output/rf_raw_importance_orig.csv,
      output/rf_raw_importance.png

4a: (1) phân bố lớp train/test; (2) RF cấu hình tốt nhất của Task 1 trên y_multi, có và không có
class_weight="balanced"; (3) ngưỡng nhị phân chọn trên validation (tối đa F1) thay vì 0,5, dùng lại
proba_val_rf/proba_test_rf của Task 1 (mô hình không đổi, chỉ đổi ngưỡng).
4b: RF trên One-Hot thưa (y_bin), chọn max_depth trên validation, so với RF trên PCA của Task 1;
độ quan trọng MDI gộp về thuộc tính gốc, đối chiếu permutation importance (hoán vị cả nhóm cột của
một thuộc tính) trên mẫu con của test.
Validation dùng chung với các bước khác: 20% train, phân tầng theo y_bin, random_state=42."""
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import train_test_split

SEED = 42
CLASSES = ["<30", ">30", "NO"]
DEPTHS_RAW = [8, 12, 16, 20, None]  # 4b: lưới max_depth cho RF trên One-Hot
N_TREES_RAW = 200
N_PERM_ROWS, N_PERM_REPEATS = 3000, 3

y_tr_df, y_te_df = pd.read_csv("output/y_train.csv"), pd.read_csv("output/y_test.csv")
y_multi_tr = y_tr_df["y_multi"].map(CLASSES.index).to_numpy()
y_multi_te = y_te_df["y_multi"].map(CLASSES.index).to_numpy()
y_bin_tr, y_bin_te = y_tr_df["y_bin"].to_numpy(), y_te_df["y_bin"].to_numpy()
tr_idx, val_idx = train_test_split(np.arange(len(y_bin_tr)), test_size=0.2,
                                   stratify=y_bin_tr, random_state=SEED)

# ---------------- 4a.1 Phân bố lớp ----------------
rows = []
for ten, d in (("train", y_tr_df), ("test", y_te_df)):
    for c in CLASSES:
        rows.append({"tap": ten, "nhan": "y_multi", "lop": c, "so_mau": int((d["y_multi"] == c).sum()),
                     "ty_le": (d["y_multi"] == c).mean()})
    for v in (0, 1):
        rows.append({"tap": ten, "nhan": "y_bin", "lop": str(v), "so_mau": int((d["y_bin"] == v).sum()),
                     "ty_le": (d["y_bin"] == v).mean()})
pd.DataFrame(rows).round(4).to_csv("output/imbalance_distribution.csv", index=False)

# ---------------- 4a.2 RF y_multi có / không class_weight ----------------
cfg = pd.read_csv("output/rf_results.csv", dtype={"max_depth": str}).set_index("label")
assert {"y_multi", "y_bin"} <= set(cfg.index), "rf_results.csv (Task 1) thiếu dòng y_multi hoặc y_bin"
depth = lambda s: None if s == "None" else int(s)
X_pca_tr, X_pca_te = np.load("output/X_train_pca.npy"), np.load("output/X_test_pca.npy")
assert len(X_pca_tr) == len(y_multi_tr) and len(X_pca_te) == len(y_multi_te)

res = []
for ten, cw in (("RF không class_weight", None), ("RF class_weight=balanced", "balanced")):
    rf = RandomForestClassifier(n_estimators=int(cfg.loc["y_multi", "n_estimators"]),
                                max_depth=depth(cfg.loc["y_multi", "max_depth"]), class_weight=cw,
                                random_state=SEED, n_jobs=-1).fit(X_pca_tr, y_multi_tr)
    pred = rf.predict(X_pca_te)
    rec = recall_score(y_multi_te, pred, labels=[0, 1, 2], average=None)
    res.append({"mo_hinh": ten, "test_accuracy": accuracy_score(y_multi_te, pred),
                "macro_f1": f1_score(y_multi_te, pred, average="macro"),
                **{f"recall_{c}": r for c, r in zip(CLASSES, rec)}, "so_lan_doan_<30": int((pred == 0).sum())})
base = np.full_like(y_multi_te, 2)  # mốc: luôn đoán NO
res.append({"mo_hinh": "Mốc: luôn đoán NO", "test_accuracy": accuracy_score(y_multi_te, base),
            "macro_f1": f1_score(y_multi_te, base, average="macro"),
            **{f"recall_{c}": float(c == "NO") for c in CLASSES}, "so_lan_doan_<30": 0})
pd.DataFrame(res).round(4).to_csv("output/imbalance_rf_multi.csv", index=False)

# ---------------- 4a.3 Ngưỡng nhị phân chọn trên validation (tối đa F1) ----------------
p_val, p_te = np.load("output/proba_val_rf.npy"), np.load("output/proba_test_rf.npy")
assert len(p_val) == len(val_idx) and len(p_te) == len(y_bin_te)
ngưỡng = np.linspace(0.05, 0.95, 91)
f1_val = [f1_score(y_bin_tr[val_idx], p_val >= t) for t in ngưỡng]
t_best = float(ngưỡng[int(np.argmax(f1_val))])  # chọn trên validation, không nhìn test
res = []
for ten, t in (("Ngưỡng 0,5", 0.5), (f"Ngưỡng chọn trên validation ({t_best:.2f})", t_best)):
    pred = (p_te >= t).astype(int)
    res.append({"ngưỡng": ten, "t": t, "test_accuracy": accuracy_score(y_bin_te, pred),
                "precision": precision_score(y_bin_te, pred), "recall": recall_score(y_bin_te, pred),
                "f1": f1_score(y_bin_te, pred), "test_auc": roc_auc_score(y_bin_te, p_te)})
pd.DataFrame(res).round(4).to_csv("output/imbalance_threshold.csv", index=False)

# ---------------- 4b RF trên One-Hot (chưa PCA) ----------------
X_tr = sparse.load_npz("output/X_train_onehot.npz").tocsr()
X_te = sparse.load_npz("output/X_test_onehot.npz").tocsr()
cols = open("output/onehot_columns.txt", encoding="utf-8").read().split("\n")
assert X_tr.shape == (len(y_bin_tr), len(cols)) and X_te.shape == (len(y_bin_te), len(cols))

grid = []
for d in DEPTHS_RAW:
    m = RandomForestClassifier(n_estimators=N_TREES_RAW, max_depth=d, random_state=SEED, n_jobs=-1)
    m.fit(X_tr[tr_idx], y_bin_tr[tr_idx])
    grid.append((roc_auc_score(y_bin_tr[val_idx], m.predict_proba(X_tr[val_idx])[:, 1]), d))
    print("RF One-Hot max_depth", d, "val AUC", round(grid[-1][0], 4))
auc_val, d_best = max(grid, key=lambda g: g[0])
final = RandomForestClassifier(n_estimators=N_TREES_RAW, max_depth=d_best, random_state=SEED,
                               n_jobs=-1).fit(X_tr, y_bin_tr)  # refit trên toàn bộ train
p_raw = final.predict_proba(X_te)[:, 1]
pd.DataFrame([
    {"du_lieu": "One-Hot (chưa PCA)", "n_estimators": N_TREES_RAW, "max_depth": str(d_best),
     "val_auc": auc_val, "test_accuracy": accuracy_score(y_bin_te, p_raw >= 0.5),
     "test_auc": roc_auc_score(y_bin_te, p_raw)},
    {"du_lieu": "PCA 60 chiều (Task 1)", "n_estimators": int(cfg.loc["y_bin", "n_estimators"]),
     "max_depth": cfg.loc["y_bin", "max_depth"], "val_auc": float(cfg.loc["y_bin", "val_metric"]),
     "test_accuracy": float(cfg.loc["y_bin", "test_accuracy"]), "test_auc": float(cfg.loc["y_bin", "test_auc"])},
]).round(4).to_csv("output/rf_raw_results.csv", index=False)

# Độ quan trọng MDI: top 20 cột One-Hot, rồi gộp về thuộc tính gốc (tiền tố dài nhất khớp)
orig = pd.read_csv("output/train.csv", nrows=1).drop(columns=["y_multi", "y_bin"]).columns


def to_orig(col):
    return max((o for o in orig if col == o or col.startswith(o + "_")), key=len)


imp = pd.DataFrame({"cot": cols, "thuoc_tinh": [to_orig(c) for c in cols],
                    "mdi": final.feature_importances_})
assert abs(imp["mdi"].sum() - 1) < 1e-6
mdi_orig = imp.groupby("thuoc_tinh")["mdi"].sum().sort_values(ascending=False)
assert abs(mdi_orig.sum() - 1) < 1e-6  # gộp không làm mất khối lượng
top20 = imp.sort_values("mdi", ascending=False).head(20)
top20.round(6).to_csv("output/rf_raw_importance_onehot.csv", index=False)

# Permutation importance theo nhóm cột của từng thuộc tính gốc, trên mẫu con của test
rng = np.random.default_rng(SEED)
sub = rng.choice(len(y_bin_te), N_PERM_ROWS, replace=False)
Xs, ys = X_te[sub].toarray(), y_bin_te[sub]
auc0 = roc_auc_score(ys, final.predict_proba(Xs)[:, 1])
perm = {}
for o, g in imp.groupby("thuoc_tinh"):
    idx, drops = g.index.to_numpy(), []
    for _ in range(N_PERM_REPEATS):
        Xp = Xs.copy()
        Xp[:, idx] = Xs[rng.permutation(len(Xs))][:, idx]  # hoán vị cả nhóm cột cùng lúc
        drops.append(auc0 - roc_auc_score(ys, final.predict_proba(Xp)[:, 1]))
    perm[o] = float(np.mean(drops))
pd.DataFrame({"thuoc_tinh": mdi_orig.index, "mdi": mdi_orig.values,
              "permutation_giam_auc": [perm[o] for o in mdi_orig.index]}).round(6).to_csv(
    "output/rf_raw_importance_orig.csv", index=False)

# Hình: (a) top 20 cột One-Hot, (b) MDI theo thuộc tính gốc, (c) permutation theo thuộc tính gốc
fig, ax = plt.subplots(1, 3, figsize=(20, 8))
ax[0].barh(top20["cot"][::-1], top20["mdi"][::-1])
ax[0].set_title("(a) Top 20 cột One-Hot (MDI)")
ax[1].barh(mdi_orig.index[::-1], mdi_orig.values[::-1])
ax[1].set_title("(b) MDI gộp theo thuộc tính gốc")
po = pd.Series(perm).sort_values()
ax[2].barh(po.index, po.values)
ax[2].set_title(f"(c) Permutation: AUC giảm (mẫu con {N_PERM_ROWS} dòng test)")
for a in ax:
    a.tick_params(axis="y", labelsize=7)
fig.tight_layout()
fig.savefig("output/rf_raw_importance.png", dpi=150, bbox_inches="tight")
plt.close(fig)
