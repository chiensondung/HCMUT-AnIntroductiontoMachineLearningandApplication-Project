"""Bước 11 (Sơn): đánh giá bài toán nhị phân (y_bin): Test Accuracy, AUC, Precision/Recall/F1 lớp 1, ROC, ma trận nhầm lẫn.
Đọc:  output/proba_test_*.npy (dt, rf, mlp, ensemble... mô hình nào có file thì được đánh giá), output/y_test.csv (cột y_bin)
Ghi:  output/binary_summary.csv, output/roc_curves.png, output/confusion_binary.png
Ngưỡng quyết định 0,5. Tập test chỉ dùng để báo cáo."""
import glob
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve

NAMES = {"dt": "Decision Tree", "rf": "Random Forest", "lr": "Logistic Regression", "mlp": "MLP", "ensemble": "Ensemble (trọng số AUC)"}

y = pd.read_csv("output/y_test.csv")["y_bin"].to_numpy()
files = sorted(glob.glob("output/proba_test_*.npy"))
assert files, "chưa có file output/proba_test_*.npy nào"
probas = {os.path.basename(f)[len("proba_test_"):-len(".npy")]: np.load(f) for f in files}
for key, p in probas.items():
    assert p.shape == y.shape and p.min() >= 0 and p.max() <= 1, f"proba_test_{key}.npy sai kích thước hoặc ngoài [0, 1]"

baseline = max(y.mean(), 1 - y.mean())  # luôn đoán lớp đa số
rows = []
fig_roc, ax_roc = plt.subplots(figsize=(6, 6))
fig_cm, axes_cm = plt.subplots(1, len(probas), figsize=(4.2 * len(probas), 4), squeeze=False)
for (key, p), ax in zip(probas.items(), axes_cm[0]):
    name, pred = NAMES.get(key, key), (p >= 0.5).astype(int)
    auc = roc_auc_score(y, p)
    rows.append({"model": name, "test_accuracy": accuracy_score(y, pred), "auc": auc,
                 "precision": precision_score(y, pred, zero_division=0), "recall": recall_score(y, pred),
                 "f1": f1_score(y, pred), "baseline_majority": baseline})
    fpr, tpr, _ = roc_curve(y, p)
    ax_roc.plot(fpr, tpr, label=f"{name} (AUC = {auc:.4f})")

    cm = confusion_matrix(y, pred, labels=[0, 1])
    assert cm.sum() == len(y)
    pct = cm / cm.sum(1, keepdims=True)
    ax.imshow(pct, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}\n{pct[i, j]:.1%}", ha="center", va="center", color="white" if pct[i, j] > 0.5 else "black")
    ax.set_xticks(range(2), ["Not readmitted", "Readmitted"]); ax.set_yticks(range(2), ["Not readmitted", "Readmitted"])
    ax.set_xlabel("Dự đoán"); ax.set_ylabel("Thực tế")
    ax.set_title(f"{name}\nAccuracy = {rows[-1]['test_accuracy']:.4f}")

ax_roc.plot([0, 1], [0, 1], "k--", label="Đoán ngẫu nhiên (AUC = 0,5)")
ax_roc.set_xlabel("False positive rate"); ax_roc.set_ylabel("True positive rate")
ax_roc.set_title("Đường cong ROC trên tập test (y_bin)"); ax_roc.legend(loc="lower right")
fig_roc.savefig("output/roc_curves.png", dpi=150, bbox_inches="tight")
fig_cm.suptitle("Ma trận nhầm lẫn nhị phân trên tập test (% theo hàng)")
fig_cm.tight_layout()
fig_cm.savefig("output/confusion_binary.png", dpi=150, bbox_inches="tight")

summary = pd.DataFrame(rows).round(4)
summary.to_csv("output/binary_summary.csv", index=False)
print(summary.to_string(index=False))
