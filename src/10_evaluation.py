"""Bước 10 (Dũng): đánh giá tổng hợp bài toán multi-class (3 lớp): Test Accuracy và ma trận nhầm lẫn.
Đọc:  output/pred_{knn,softmax}_{chua,da}_chuan_hoa.npy (từ 07a, 07b), output/y_test.csv (cột y_multi)
Ghi:  output/accuracy_summary.csv, output/confusion_*.csv, output/confusion_matrices.png"""
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

classes = np.array(["<30", ">30", "NO"])
y = pd.read_csv("output/y_test.csv")["y_multi"].map({c: i for i, c in enumerate(classes)}).to_numpy()
baseline = np.bincount(y).max() / len(y)  # luôn đoán lớp đa số

MODELS = [("knn", "k-NN"), ("softmax", "Softmax Regression")]
MODES = [("chua_chuan_hoa", "Chưa chuẩn hóa"), ("da_chuan_hoa", "Đã chuẩn hóa")]

rows = []
fig, axes = plt.subplots(len(MODELS), len(MODES), figsize=(11, 9))
for (m, m_name), axr in zip(MODELS, axes):
    for (mode, mode_name), ax in zip(MODES, axr):
        pred = np.load(f"output/pred_{m}_{mode}.npy")
        assert len(pred) == len(y), "nhãn dự đoán không cùng số dòng với y_test"
        cm = confusion_matrix(y, pred, labels=range(3))
        assert cm.sum() == len(y)
        pd.DataFrame(cm, index=classes, columns=classes).to_csv(f"output/confusion_{m}_{mode}.csv")
        rows.append({"model": m_name, "data": mode_name, "test_accuracy": accuracy_score(y, pred),
                     "macro_f1": f1_score(y, pred, average="macro"), "baseline_majority": baseline})
        pct = cm / cm.sum(1, keepdims=True)
        ax.imshow(pct, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{cm[i, j]}\n{pct[i, j]:.1%}", ha="center", va="center",
                        color="white" if pct[i, j] > 0.5 else "black")
        ax.set_xticks(range(3), classes); ax.set_yticks(range(3), classes)
        ax.set_xlabel("Dự đoán"); ax.set_ylabel("Thực tế")
        ax.set_title(f"{m_name} - {mode_name}\nAccuracy = {rows[-1]['test_accuracy']:.4f}")
fig.suptitle("Ma trận nhầm lẫn trên tập test (% theo hàng)")
fig.tight_layout()
fig.savefig("output/confusion_matrices.png", dpi=150)

summary = pd.DataFrame(rows).round(4)
summary.to_csv("output/accuracy_summary.csv", index=False)
print(summary.to_string(index=False))
