"""Task 3.3: Thiết kế & huấn luyện Deep Neural Network / MLP
(Hidden layers: 120, 60, 20, ReLU) và lập trình Aggregate Ensemble Model
dựa trên trọng số AUC-ROC.

Đọc (pipeline chung, kết quả bước One-Hot + PCA):
    output/X_train_pca.npy, output/X_test_pca.npy  (60 thành phần chính)
    output/y_train.csv, output/y_test.csv          (cột y_multi, y_bin; cùng thứ tự dòng)
Ghi:
    output/3.3_ket_qua_test.csv, output/3.3_roc_curve_test.png, output/3.3_tomtat.md

Mô hình nền (base models): Logistic Regression, Random Forest, Gradient
Boosting, MLP — mỗi mô hình huấn luyện trên train, đánh giá AUC-ROC trên
validation (20% tách từ train). Ensemble = trung bình có trọng số các xác suất
dự đoán, trọng số tỉ lệ thuận với AUC-ROC của từng mô hình trên validation.
Đánh giá cuối cùng trên test (tách riêng, không dùng để chọn trọng số).
Nhãn: y_bin (1 = tái nhập viện <30 hoặc >30 ngày, 0 = không).
"""
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    RocCurveDisplay, accuracy_score, f1_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

KET_QUA_DIR = "output"
RANDOM_STATE = 42

# --- 1. Đọc dữ liệu PCA và nhãn từ các bước trước ---
X_train = np.load("output/X_train_pca.npy")
X_test = np.load("output/X_test_pca.npy")
y_train = pd.read_csv("output/y_train.csv")["y_bin"].to_numpy()
y_test = pd.read_csv("output/y_test.csv")["y_bin"].to_numpy()
assert len(X_train) == len(y_train) and len(X_test) == len(y_test)

# --- 2. Validation = 20% của train (phân tầng); test giữ riêng ---
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.2, stratify=y_train, random_state=RANDOM_STATE,
)
scaler = StandardScaler().fit(X_tr)  # chuẩn hóa sau PCA, fit trên train
X_tr_s = scaler.transform(X_tr)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)
print("Số đặc trưng:", X_tr_s.shape[1],
      "| train/val/test:", len(y_tr), len(y_val), len(y_test))

# --- 3. Các mô hình nền ---
mo_hinh = {
    "LogisticRegression": LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE,
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=300, max_depth=None, class_weight="balanced",
        n_jobs=-1, random_state=RANDOM_STATE,
    ),
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.1, random_state=RANDOM_STATE,
    ),
    # Hidden layers 120-60-20, ReLU; alpha = hệ số L2 (mục 2.4.2);
    # early_stopping + n_iter_no_change = Early Stopping (mục 2.4.3)
    "MLP_120_60_20": MLPClassifier(
        hidden_layer_sizes=(120, 60, 20),
        activation="relu",
        alpha=1e-4,
        solver="adam",
        early_stopping=True,
        n_iter_no_change=10,
        validation_fraction=0.1,
        max_iter=300,
        random_state=RANDOM_STATE,
    ),
}

# GradientBoosting không có class_weight -> dùng sample_weight nghịch tần suất lớp
trong_so_mau = np.where(y_tr == 1, (y_tr == 0).sum() / len(y_tr), (y_tr == 1).sum() / len(y_tr))

ket_qua_val, xac_suat_test = {}, {}
for ten, model in mo_hinh.items():
    print(f"Huấn luyện {ten} ...")
    if ten == "GradientBoosting":
        model.fit(X_tr_s, y_tr, sample_weight=trong_so_mau)
    else:
        model.fit(X_tr_s, y_tr)

    proba_val = model.predict_proba(X_val_s)[:, 1]
    auc_val = roc_auc_score(y_val, proba_val)
    ket_qua_val[ten] = auc_val
    xac_suat_test[ten] = model.predict_proba(X_test_s)[:, 1]
    print(f"  AUC-ROC (validation) = {auc_val:.4f}")

# --- 4. Aggregate Ensemble: trung bình có trọng số theo AUC-ROC (validation) ---
auc_series = pd.Series(ket_qua_val)
trong_so = auc_series / auc_series.sum()
print("\nTrọng số ensemble (chuẩn hóa theo AUC-ROC validation):")
print(trong_so.round(4).to_dict())

proba_ensemble_test = sum(trong_so[ten] * xac_suat_test[ten] for ten in mo_hinh)

# --- 5. Đánh giá trên test (giữ riêng, không dùng để chọn trọng số) ---
def danh_gia(y_true, proba, nguong=0.5):
    pred = (proba >= nguong).astype(int)
    return {
        "AUC-ROC": roc_auc_score(y_true, proba),
        "Accuracy": accuracy_score(y_true, pred),
        "Recall": recall_score(y_true, pred),
        "F1": f1_score(y_true, pred),
    }


bang_ket_qua = {ten: danh_gia(y_test, xac_suat_test[ten]) for ten in mo_hinh}
bang_ket_qua["Aggregate_Ensemble"] = danh_gia(y_test, proba_ensemble_test)
df_ket_qua = pd.DataFrame(bang_ket_qua).T.round(4)
df_ket_qua.insert(0, "trong_so_ensemble", [round(trong_so.get(ten, np.nan), 4) for ten in df_ket_qua.index])
print("\nKết quả trên tập test:\n", df_ket_qua)
df_ket_qua.to_csv(f"{KET_QUA_DIR}/3.3_ket_qua_test.csv")

# --- 6. Biểu đồ ROC các mô hình + ensemble trên test ---
fig, ax = plt.subplots(figsize=(7, 6))
for ten in mo_hinh:
    RocCurveDisplay.from_predictions(y_test, xac_suat_test[ten], name=ten, ax=ax)
RocCurveDisplay.from_predictions(y_test, proba_ensemble_test, name="Aggregate Ensemble",
                                  ax=ax, curve_kwargs={"linewidth": 2.5, "color": "black"})
ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
ax.set_title("ROC curve trên tập test: mô hình nền vs Aggregate Ensemble")
fig.tight_layout()
fig.savefig(f"{KET_QUA_DIR}/3.3_roc_curve_test.png", dpi=150)
plt.close(fig)

# --- 7. Tóm tắt markdown ---
dong = [
    "# Kết quả Task 3.3: MLP (120-60-20, ReLU) + Aggregate Ensemble theo AUC-ROC\n",
    "## Trọng số ensemble (tỉ lệ thuận AUC-ROC trên validation)\n",
    trong_so.round(4).to_string(),
    "\n## Kết quả trên tập test\n",
    df_ket_qua.to_markdown(),
]
with open(f"{KET_QUA_DIR}/3.3_tomtat.md", "w", encoding="utf-8") as f:
    f.write("\n".join(dong))

print("\nĐã lưu kết quả vào", KET_QUA_DIR)
