"""Bước 07b (Dũng): Softmax Regression (Logistic Regression multi-class) tự cài bằng numpy,
so sánh PCA chưa chuẩn hóa vs đã chuẩn hóa.
Đọc:  output/X_{train,test}_pca.npy, output/X_{train,test}_pca_scaled.npy, output/y_{train,test}.csv (cột y_multi)
Ghi:  output/softmax_results.csv, output/softmax_loss.png
Gradient descent toàn batch + L2; chọn (lr, lambda) bằng validation 80/20 tách từ train (macro-F1)."""
import itertools
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

LRS, LAMBDAS, EPOCHS = [0.001, 0.01, 0.1, 0.5], [0.0, 1e-3, 1e-2], 500
classes = np.array(["<30", ">30", "NO"])
y_all = pd.read_csv("output/y_train.csv")["y_multi"].map({c: i for i, c in enumerate(classes)}).to_numpy()
y_te = pd.read_csv("output/y_test.csv")["y_multi"].map({c: i for i, c in enumerate(classes)}).to_numpy()


def softmax(Z):
    Z = Z - Z.max(1, keepdims=True)  # ổn định số học
    E = np.exp(Z)
    return E / E.sum(1, keepdims=True)


def fit(X, y, lr, lam, epochs=EPOCHS):
    n, d = X.shape
    Y = np.eye(3)[y]
    W, b, hist = np.zeros((d, 3)), np.zeros(3), []
    for _ in range(epochs):
        P = softmax(X @ W + b)
        hist.append(-np.log(np.clip(P[np.arange(n), y], 1e-12, None)).mean() + lam / 2 * (W ** 2).sum())
        G = (P - Y) / n
        W -= lr * (X.T @ G + lam * W)
        b -= lr * G.sum(0)
        if not np.isfinite(W).all():  # lr quá lớn -> phân kỳ
            hist.append(np.nan); break
    return W, b, hist


def predict(X, W, b):
    return (X @ W + b).argmax(1)


# self-check: dữ liệu tách được tuyến tính thì phải học gần đúng hoàn hảo
_r = np.random.default_rng(0); _y = _r.integers(0, 3, 300); _x = np.eye(3)[_y] * 3 + _r.normal(size=(300, 3)) * 0.3
_W, _b, _ = fit(_x, _y, 0.5, 0.0, 200)
assert (predict(_x, _W, _b) == _y).mean() > 0.95

rows, best_hist = [], {}
for mode, suffix in [("chua_chuan_hoa", ""), ("da_chuan_hoa", "_scaled")]:
    X_all = np.load(f"output/X_train_pca{suffix}.npy")
    X_te = np.load(f"output/X_test_pca{suffix}.npy")
    X_tr, X_val, y_tr, y_val = train_test_split(X_all, y_all, test_size=0.2, stratify=y_all, random_state=42)

    best = (-1, None)
    for lr, lam in itertools.product(LRS, LAMBDAS):
        W, b, hist = fit(X_tr, y_tr, lr, lam)
        s = f1_score(y_val, predict(X_val, W, b), average="macro") if np.isfinite(W).all() else -1
        print(mode, lr, lam, round(s, 4), "loss cuối:", round(hist[-1], 4))
        if s > best[0]:
            best = (s, (lr, lam))
    lr, lam = best[1]
    W, b, hist = fit(X_all, y_all, lr, lam)  # refit trên toàn bộ train
    best_hist[mode] = hist
    pred = predict(X_te, W, b)
    rows.append({"mode": mode, "lr": lr, "lambda": lam, "val_macro_f1": best[0],
                 "test_accuracy": accuracy_score(y_te, pred),
                 "test_macro_f1": f1_score(y_te, pred, average="macro"),
                 **{f"test_f1_{c}": f for c, f in zip(classes, f1_score(y_te, pred, average=None, labels=range(3)))}})
    print(rows[-1])

pd.DataFrame(rows).round(4).to_csv("output/softmax_results.csv", index=False)
for mode, h in best_hist.items():
    plt.plot(h, label=mode)
plt.xlabel("epoch"); plt.ylabel("cross-entropy loss (train)"); plt.legend(); plt.title("Softmax Regression: đường loss")
plt.savefig("output/softmax_loss.png", dpi=150, bbox_inches="tight")
