"""Bước 09a (Chiến): MLP 120-60-20 (ReLU), quét alpha (L2) và learning_rate_init theo validation.
Đọc:  output/X_train_pca_scaled.npy, output/X_test_pca_scaled.npy (bước 06),
      output/y_train.csv, output/y_test.csv (bước 04; cột y_multi, y_bin)
Ghi:  output/pred_multi_mlp.npy, output/proba_val_mlp.npy, output/proba_test_mlp.npy,
      output/mlp_results.csv, output/mlp_sweep.csv, output/mlp_alpha_curve.png, output/mlp_loss.png

Chạy cho cả hai nhãn: y_multi (chọn theo macro-F1) và y_bin (chọn theo AUC-ROC).
Validation dùng chung một cách chia (tr_idx, val_idx) cho mọi mô hình. Sau khi chọn
siêu tham số trên validation, mô hình được huấn luyện lại trên toàn bộ train và báo
cáo trên test một lần duy nhất. proba_val_mlp.npy lấy từ mô hình fit trên tr_idx
(đúng hợp đồng để bước 09b tính trọng số Ensemble)."""
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

SEED = 42
CLASSES = ["<30", ">30", "NO"]
ALPHAS = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
LRS = [1e-3, 1e-2]

X_train = np.load("output/X_train_pca_scaled.npy")
X_test = np.load("output/X_test_pca_scaled.npy")
y_train_df = pd.read_csv("output/y_train.csv")
y_test_df = pd.read_csv("output/y_test.csv")
assert len(X_train) == len(y_train_df) and len(X_test) == len(y_test_df)
assert X_train.shape[1] == X_test.shape[1] == 60

y_bin_all = y_train_df["y_bin"].to_numpy()
tr_idx, val_idx = train_test_split(np.arange(len(y_bin_all)), test_size=0.2,
                                   stratify=y_bin_all, random_state=SEED)
assert not set(tr_idx) & set(val_idx) and len(tr_idx) + len(val_idx) == len(y_bin_all)

nhan = {
    "y_multi": (y_train_df["y_multi"].map(CLASSES.index).to_numpy(),
                y_test_df["y_multi"].map(CLASSES.index).to_numpy(), "macro_f1"),
    "y_bin": (y_bin_all, y_test_df["y_bin"].to_numpy(), "auc"),
}


def tao_mlp(alpha, lr):
    return MLPClassifier(hidden_layer_sizes=(120, 60, 20), activation="relu", solver="adam",
                         alpha=alpha, learning_rate_init=lr, max_iter=300,
                         early_stopping=True, validation_fraction=0.1, n_iter_no_change=10,
                         random_state=SEED)


def do(y, proba, ten):
    if ten == "auc":
        return roc_auc_score(y, proba[:, 1])
    return f1_score(y, proba.argmax(1), average="macro")


def chay_mot(y_all, ten_do, alpha, lr):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        m = tao_mlp(alpha, lr).fit(X_train[tr_idx], y_all[tr_idx])
    p_tr, p_val = m.predict_proba(X_train[tr_idx]), m.predict_proba(X_train[val_idx])
    return {"alpha": alpha, "lr": lr, "train": do(y_all[tr_idx], p_tr, ten_do),
            "val": do(y_all[val_idx], p_val, ten_do), "n_iter": m.n_iter_}, p_val


sweep_rows, ket_qua, mo_hinh_cuoi = [], [], {}
for ten_nhan, (y_all, y_te, ten_do) in nhan.items():
    print(f"\n=== {ten_nhan}: quét {len(ALPHAS) * len(LRS)} cấu hình, độ đo {ten_do} ===")
    out = Parallel(n_jobs=-1)(delayed(chay_mot)(y_all, ten_do, a, lr) for lr in LRS for a in ALPHAS)
    bang = pd.DataFrame([o[0] for o in out])
    bang.insert(0, "label", ten_nhan)
    sweep_rows.append(bang)
    k = int(bang["val"].idxmax())
    tot, proba_val = bang.loc[k], out[k][1]
    print(bang.round(4).to_string(index=False))
    print(f"Chọn alpha={tot['alpha']}, lr={tot['lr']} (val {ten_do} = {tot['val']:.4f})")

    if ten_nhan == "y_bin":
        np.save("output/proba_val_mlp.npy", proba_val[:, 1])
        assert np.load("output/proba_val_mlp.npy").shape == (len(val_idx),)

    # Refit trên toàn bộ train, đánh giá test một lần
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        m = tao_mlp(tot["alpha"], tot["lr"]).fit(X_train, y_all)
    p_te = m.predict_proba(X_test)
    pred = p_te.argmax(1)
    mo_hinh_cuoi[ten_nhan] = m
    dong = {"label": ten_nhan, "alpha": tot["alpha"], "learning_rate_init": tot["lr"],
            f"train_{ten_do}": tot["train"], f"val_{ten_do}": tot["val"],
            "test_accuracy": accuracy_score(y_te, pred)}
    if ten_nhan == "y_multi":
        np.save("output/pred_multi_mlp.npy", pred)
        assert set(np.unique(pred)) <= {0, 1, 2}
        dong["test_macro_f1"] = f1_score(y_te, pred, average="macro")
    else:
        assert p_te[:, 1].min() >= 0 and p_te[:, 1].max() <= 1
        np.save("output/proba_test_mlp.npy", p_te[:, 1])
        dong["test_auc"] = roc_auc_score(y_te, p_te[:, 1])
    ket_qua.append(dong)

sweep = pd.concat(sweep_rows)
# round(4) trực tiếp sẽ làm tròn alpha=1e-5 thành 0, nên chỉ làm tròn cột độ đo
sweep.round({"train": 4, "val": 4}).to_csv("output/mlp_sweep.csv", index=False)
ket_qua_df = pd.DataFrame(ket_qua)
ket_qua_df.loc[:, "train_macro_f1":] = ket_qua_df.loc[:, "train_macro_f1":].round(4)
ket_qua_df.to_csv("output/mlp_results.csv", index=False)
print("\n", ket_qua_df.to_string(index=False))

# --- Hình 1: validation theo alpha (trục x log), mỗi learning rate một đường ---
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, (ten_nhan, (_, _, ten_do)) in zip(axes, nhan.items()):
    b = sweep[sweep["label"] == ten_nhan]
    for lr in LRS:
        s = b[b["lr"] == lr]
        ax.plot(s["alpha"], s["val"], "o-", label=f"val, lr={lr:g}")
        ax.plot(s["alpha"], s["train"], "--", alpha=0.5, label=f"train, lr={lr:g}")
    ax.set_xscale("log")
    ax.set_xlabel("alpha (hệ số L2)")
    ax.set_ylabel(ten_do)
    ax.set_title(ten_nhan)
    ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("output/mlp_alpha_curve.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# --- Hình 2: loss huấn luyện và validation score của mô hình cuối ---
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, (ten_nhan, m) in zip(axes, mo_hinh_cuoi.items()):
    ax.plot(m.loss_curve_, label="loss (train)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax2 = ax.twinx()
    ax2.plot(m.validation_scores_, color="tab:orange", label="validation score (accuracy)")
    ax2.set_ylabel("validation accuracy")
    ax.set_title(f"{ten_nhan} (alpha={m.alpha:g}, lr={m.learning_rate_init:g})")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8)
fig.tight_layout()
fig.savefig("output/mlp_loss.png", dpi=150, bbox_inches="tight")
plt.close(fig)
