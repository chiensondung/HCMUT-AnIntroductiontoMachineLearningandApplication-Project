"""Bước 09b (Chiến): Aggregate Ensemble theo trọng số AUC-ROC (chỉ bài toán nhị phân y_bin).
Đọc:  output/proba_val_{dt,rf,mlp}.npy, output/proba_test_{dt,rf,mlp}.npy (Task 1 và 09a),
      output/X_train_pca_scaled.npy, output/X_test_pca_scaled.npy, output/y_train.csv, output/y_test.csv
Ghi:  output/proba_test_ensemble.npy, output/ensemble_weights.csv, output/ensemble_results.csv,
      output/proba_val_lr.npy, output/proba_test_lr.npy (thành viên Logistic Regression)

w_i = AUC_i / sum_j AUC_j, tính trên VALIDATION (không dùng test). p_ens = sum_i w_i p_i.
Thành viên: DT, RF, MLP (đọc từ file; file nào chưa có thì bỏ qua và báo) và Logistic
Regression (fit tại đây trên tr_idx, refit trên toàn bộ train). Ngoài trọng số AUC còn so
sánh với trung bình không trọng số và trọng số theo (AUC - 0,5)."""
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

SEED = 42
X_train = np.load("output/X_train_pca_scaled.npy")
X_test = np.load("output/X_test_pca_scaled.npy")
y_train = pd.read_csv("output/y_train.csv")["y_bin"].to_numpy()
y_test = pd.read_csv("output/y_test.csv")["y_bin"].to_numpy()
assert len(X_train) == len(y_train) and len(X_test) == len(y_test)

# Cùng cách chia validation với mọi mô hình
tr_idx, val_idx = train_test_split(np.arange(len(y_train)), test_size=0.2,
                                   stratify=y_train, random_state=SEED)
y_val = y_train[val_idx]

# --- Thành viên Logistic Regression nhị phân ---
lr = LogisticRegression(max_iter=1000, random_state=SEED).fit(X_train[tr_idx], y_train[tr_idx])
np.save("output/proba_val_lr.npy", lr.predict_proba(X_train[val_idx])[:, 1])
lr = LogisticRegression(max_iter=1000, random_state=SEED).fit(X_train, y_train)
np.save("output/proba_test_lr.npy", lr.predict_proba(X_test)[:, 1])

# --- Đọc xác suất của các thành viên ---
THANH_VIEN = ["dt", "rf", "mlp", "lr"]
p_val, p_test, thieu = {}, {}, []
for ten in THANH_VIEN:
    fv, ft = f"output/proba_val_{ten}.npy", f"output/proba_test_{ten}.npy"
    if not (os.path.exists(fv) and os.path.exists(ft)):
        thieu.append(ten)
        continue
    p_val[ten], p_test[ten] = np.load(fv), np.load(ft)
    assert p_val[ten].shape == (len(val_idx),), f"{ten}: proba_val sai số dòng"
    assert p_test[ten].shape == (len(y_test),), f"{ten}: proba_test sai số dòng"
    assert 0 <= p_val[ten].min() and p_val[ten].max() <= 1 and 0 <= p_test[ten].max() <= 1
if thieu:
    print("CHƯA CÓ file proba của:", thieu, "-> Ensemble chỉ gồm", list(p_val))
assert len(p_val) >= 2, "Cần ít nhất 2 thành viên"
ten_list = list(p_val)

# --- Trọng số (tính trên validation) ---
auc_val = pd.Series({t: roc_auc_score(y_val, p_val[t]) for t in ten_list})
trong_so = pd.DataFrame({
    "auc_val": auc_val,
    "w_auc": auc_val / auc_val.sum(),
    "w_equal": 1.0 / len(ten_list),
    "w_auc_tru_0.5": (auc_val - 0.5) / (auc_val - 0.5).sum(),
})
for c in ["w_auc", "w_equal", "w_auc_tru_0.5"]:
    assert abs(trong_so[c].sum() - 1) < 1e-9, c
trong_so.round(4).to_csv("output/ensemble_weights.csv", index_label="model")
print(trong_so.round(4))

# --- Ensemble trên test ---
def tron(w):
    return sum(w[t] * p_test[t] for t in ten_list)

p_ens = tron(trong_so["w_auc"])
assert p_ens.min() >= 0 and p_ens.max() <= 1
np.save("output/proba_test_ensemble.npy", p_ens)


def danh_gia(proba):
    pred = (proba >= 0.5).astype(int)
    return {"test_auc": roc_auc_score(y_test, proba), "test_accuracy": accuracy_score(y_test, pred),
            "test_recall": recall_score(y_test, pred), "test_f1": f1_score(y_test, pred)}


bang = {t: danh_gia(p_test[t]) for t in ten_list}
bang["ensemble_w_auc"] = danh_gia(p_ens)
bang["ensemble_trung_binh"] = danh_gia(tron(trong_so["w_equal"]))
bang["ensemble_w_auc_tru_0.5"] = danh_gia(tron(trong_so["w_auc_tru_0.5"]))
kq = pd.DataFrame(bang).T.round(4)
kq.to_csv("output/ensemble_results.csv", index_label="model")
print("\n", kq)
