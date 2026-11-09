"""Task 3.3: Thiết kế & huấn luyện Deep Neural Network / MLP
(Hidden layers: 120, 60, 20, ReLU) và lập trình Aggregate Ensemble Model
dựa trên trọng số AUC-ROC.

Đầu vào: 1 file CSV duy nhất (input_dir), là kết quả sau các bước 1.1-1.3
(lọc/làm sạch, tạo nhãn) và 2.1-2.3 (mã hóa biến phân loại, chuẩn hóa/PCA) —
tức các cột đặc trưng trong file này phải là số (đã mã hóa), kèm cột nhãn
y_bin. Chỉnh input_dir và feature_cols/target_col bên dưới cho khớp với
file đầu ra thực tế của các bước trước.

Mô hình nền (base models): Logistic Regression, Random Forest, Gradient
Boosting, MLP — mỗi mô hình huấn luyện trên train, đánh giá AUC-ROC trên
validation. Ensemble = trung bình có trọng số các xác suất dự đoán, trọng
số tỉ lệ thuận với AUC-ROC của từng mô hình trên validation. Đánh giá cuối
cùng trên test (tách riêng, không dùng để chọn trọng số).

Kết quả lưu vào task_3-3-result/.
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

# --- Cấu hình đầu vào: sửa lại cho khớp với file đầu ra của bước 2.3 ---
input_dir = "data/train.csv"
target_col = "y_bin"
cot_loai_bo = ["y_multi", "y_bin"]  # các cột không phải đặc trưng

KET_QUA_DIR = "task_3-3-result"
RANDOM_STATE = 42

# --- 1. Đọc dữ liệu đã tiền xử lý/mã hóa/chuẩn hóa từ các bước trước ---
df = pd.read_csv(input_dir, low_memory=False)
feature_cols = [c for c in df.columns if c not in cot_loai_bo]
X = df[feature_cols].astype(float)
y = df[target_col].to_numpy()

# --- 2. Chia train / validation / test (60/20/20), phân tầng theo nhãn ---
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE,
)
X_tr, X_val, y_tr, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.25, stratify=y_trainval, random_state=RANDOM_STATE,
)  # 0.25 * 0.8 = 0.2 -> train 60% / val 20% / test 20%

scaler = StandardScaler().fit(X_tr)
X_tr_s = scaler.transform(X_tr)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

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
df_ket_qua.to_csv(f"{KET_QUA_DIR}/ket_qua_test.csv")

# --- 6. Biểu đồ ROC các mô hình + ensemble trên test ---
fig, ax = plt.subplots(figsize=(7, 6))
for ten in mo_hinh:
    RocCurveDisplay.from_predictions(y_test, xac_suat_test[ten], name=ten, ax=ax)
RocCurveDisplay.from_predictions(y_test, proba_ensemble_test, name="Aggregate Ensemble",
                                  ax=ax, curve_kwargs={"linewidth": 2.5, "color": "black"})
ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
ax.set_title("ROC curve trên tập test: mô hình nền vs Aggregate Ensemble")
fig.tight_layout()
fig.savefig(f"{KET_QUA_DIR}/roc_curve_test.png", dpi=150)
plt.close(fig)

# --- 7. Tóm tắt markdown ---
dong = [
    "# Kết quả Task 3.3: MLP (120-60-20, ReLU) + Aggregate Ensemble theo AUC-ROC\n",
    "## Trọng số ensemble (tỉ lệ thuận AUC-ROC trên validation)\n",
    trong_so.round(4).to_string(),
    "\n## Kết quả trên tập test\n",
    df_ket_qua.to_markdown(),
]
with open(f"{KET_QUA_DIR}/task_3-3_tomtat.md", "w", encoding="utf-8") as f:
    f.write("\n".join(dong))

print("\nĐã lưu kết quả vào", KET_QUA_DIR)
