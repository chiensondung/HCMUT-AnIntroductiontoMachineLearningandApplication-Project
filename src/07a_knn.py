"""Bước 07a (Dũng): k-NN tự cài (numpy), so sánh PCA chưa chuẩn hóa vs đã chuẩn hóa.
Đọc:  output/X_{train,test}_pca.npy, output/X_{train,test}_pca_scaled.npy, output/y_{train,test}.csv (cột y_multi)
Ghi:  output/knn_results.csv, output/knn_k_curve.png
Chọn k bằng validation 80/20 tách từ train (macro-F1); test chỉ dùng để báo cáo."""
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

KS = list(range(1, 52, 2))
KMAX, BATCH = max(KS), 1024
classes = np.array(["<30", ">30", "NO"])

y_tr_all = pd.read_csv("output/y_train.csv")["y_multi"].map({c: i for i, c in enumerate(classes)}).to_numpy()
y_te = pd.read_csv("output/y_test.csv")["y_multi"].map({c: i for i, c in enumerate(classes)}).to_numpy()


def neighbor_labels(Xq, Xr, yr, kmax=KMAX):
    """Nhãn của kmax láng giềng gần nhất (Euclid), sắp theo khoảng cách tăng dần: (n_query, kmax)."""
    r2 = (Xr ** 2).sum(1)
    out = np.empty((len(Xq), kmax), dtype=np.int8)
    for s in range(0, len(Xq), BATCH):
        q = Xq[s:s + BATCH]
        d = r2[None, :] - 2 * q @ Xr.T  # bỏ ||q||^2: không đổi thứ tự trong 1 hàng
        idx = np.argpartition(d, kmax, axis=1)[:, :kmax]
        idx = np.take_along_axis(idx, np.argsort(np.take_along_axis(d, idx, 1), axis=1), 1)
        out[s:s + BATCH] = yr[idx]
    return out


def predict(nl, k):
    votes = np.stack([(nl[:, :k] == c).sum(1) for c in range(len(classes))], 1)
    return votes.argmax(1)  # hòa phiếu -> lớp có chỉ số nhỏ hơn


# self-check: 1-NN trên chính tập tham chiếu phải trả lại đúng nhãn của nó
_x = np.random.default_rng(0).normal(size=(50, 4)); _y = np.arange(50) % 3
assert (predict(neighbor_labels(_x, _x, _y, 3), 1) == _y).all()

rows, curves = [], {}
for mode, suffix in [("chua_chuan_hoa", ""), ("da_chuan_hoa", "_scaled")]:
    X_all = np.load(f"output/X_train_pca{suffix}.npy")
    X_te = np.load(f"output/X_test_pca{suffix}.npy")
    X_tr, X_val, y_tr, y_val = train_test_split(X_all, y_tr_all, test_size=0.2, stratify=y_tr_all, random_state=42)

    nl = neighbor_labels(X_val, X_tr, y_tr)
    f1s = [f1_score(y_val, predict(nl, k), average="macro") for k in KS]
    curves[mode] = f1s
    best_k = KS[int(np.argmax(f1s))]

    pred = predict(neighbor_labels(X_te, X_all, y_tr_all), best_k)  # refit trên toàn bộ train
    rows.append({"mode": mode, "best_k": best_k, "val_macro_f1": max(f1s),
                 "test_accuracy": accuracy_score(y_te, pred),
                 "test_macro_f1": f1_score(y_te, pred, average="macro"),
                 **{f"test_f1_{c}": f for c, f in zip(classes, f1_score(y_te, pred, average=None, labels=range(3)))}})
    print(rows[-1])

pd.DataFrame(rows).round(4).to_csv("output/knn_results.csv", index=False)
for mode, f1s in curves.items():
    plt.plot(KS, f1s, marker="o", label=mode)
plt.xlabel("k"); plt.ylabel("macro-F1 (validation)"); plt.legend(); plt.title("k-NN: chọn k")
plt.savefig("output/knn_k_curve.png", dpi=150, bbox_inches="tight")
