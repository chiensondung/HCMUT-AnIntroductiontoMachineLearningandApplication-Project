"""Task 2.3: Chuẩn hóa dữ liệu (Scaling/Normalization), sau đó PCA;
vẽ biểu đồ Eigenvalues (scree plot) và phân tích top 5 vector riêng (eigenvector)
quan trọng nhất. Kết quả lưu vào task_2-3-result/, chạy từ thư mục gốc dự án.

Đầu vào: data/train.csv (đã qua bước 1.1-1.3: bỏ cột định danh/khuyết nhiều,
xóa dòng khuyết <2%, tạo nhãn, chia train/test).
"""
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

KET_QUA_DIR = "task_2-3-result"

df = pd.read_csv("data/train.csv", low_memory=False)

# --- 1. Chọn và mã hóa đặc trưng ---
# medical_specialty, diag_1/2/3: phân loại bậc cao (70-759 giá trị), khuyết nhiều
# (medical_specialty ~49%) -> chưa đưa vào PCA ở bước này (cần gom nhóm riêng).
cot_so = [
    "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
    "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses",
]
cot_thuoc = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
    "examide", "citoglipton", "insulin", "glyburide-metformin", "glipizide-metformin",
    "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone",
]
cot_onehot = ["race", "gender", "admission_type_id", "discharge_disposition_id",
              "admission_source_id", "change", "diabetesMed"] + cot_thuoc

# age là biến thứ bậc -> mã hóa thành số nguyên 0..9 theo thứ tự khoảng tuổi
nhom_tuoi = sorted(df["age"].unique(), key=lambda s: int(s.split("-")[0][1:]))
age_so = df["age"].map({nhom: i for i, nhom in enumerate(nhom_tuoi)}).rename("age_ordinal")

df_onehot = df[cot_onehot].fillna("Missing")
X_onehot = pd.get_dummies(df_onehot, columns=cot_onehot, drop_first=False)

X = pd.concat([df[cot_so].reset_index(drop=True),
               age_so.reset_index(drop=True),
               X_onehot.reset_index(drop=True)], axis=1).astype(float)
print("Số đặc trưng đưa vào PCA:", X.shape[1])

# --- 2. Chuẩn hóa (StandardScaler: mean=0, std=1) ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 3. PCA trên dữ liệu đã chuẩn hóa ---
pca = PCA(n_components=min(X_scaled.shape))
pca.fit(X_scaled)

eigenvalues = pca.explained_variance_
ty_le_giai_thich = pca.explained_variance_ratio_
ty_le_tich_luy = np.cumsum(ty_le_giai_thich)

pd.DataFrame({
    "component": np.arange(1, len(eigenvalues) + 1),
    "eigenvalue": eigenvalues,
    "ty_le_giai_thich": ty_le_giai_thich,
    "ty_le_tich_luy": ty_le_tich_luy,
}).to_csv(f"{KET_QUA_DIR}/eigenvalues.csv", index=False)

so_pc_80 = int(np.searchsorted(ty_le_tich_luy, 0.80) + 1)
print(f"Số PC cần để giải thích >=80% phương sai: {so_pc_80}/{len(eigenvalues)}")

# --- 4. Biểu đồ Eigenvalues (scree plot) + phương sai tích lũy ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
so_pc_ve = 30
ax1.plot(range(1, so_pc_ve + 1), eigenvalues[:so_pc_ve], "o-", color="tab:blue")
ax1.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="Eigenvalue = 1")
ax1.set_xlabel("Thành phần chính (PC)")
ax1.set_ylabel("Eigenvalue")
ax1.set_title(f"Scree plot ({so_pc_ve} PC đầu)")
ax1.legend()

ax2.plot(range(1, len(ty_le_tich_luy) + 1), ty_le_tich_luy, color="tab:orange")
ax2.axhline(0.8, color="gray", linestyle="--", linewidth=1, label="80%")
ax2.set_xlabel("Số thành phần chính")
ax2.set_ylabel("Tỉ lệ phương sai giải thích tích lũy")
ax2.set_title("Phương sai tích lũy theo số PC")
ax2.legend()

fig.tight_layout()
fig.savefig(f"{KET_QUA_DIR}/eigenvalues_scree_plot.png", dpi=150)
plt.close(fig)

# --- 5. Phân tích top 5 vector riêng (PC1..PC5): các đặc trưng đóng góp lớn nhất ---
ten_dac_trung = X.columns.to_numpy()
dong_gop = []
for i in range(5):
    vector = pca.components_[i]
    idx_top = np.argsort(-np.abs(vector))[:10]
    for idx in idx_top:
        dong_gop.append({
            "PC": f"PC{i + 1}",
            "ty_le_giai_thich_PC": round(ty_le_giai_thich[i], 4),
            "dac_trung": ten_dac_trung[idx],
            "trong_so": round(vector[idx], 4),
        })
df_dong_gop = pd.DataFrame(dong_gop)
df_dong_gop.to_csv(f"{KET_QUA_DIR}/top5_pc_dac_trung_dong_gop.csv", index=False)

# --- 6. Tóm tắt dạng markdown ---
dong_tomtat = [
    "# Kết quả Task 2.3: Chuẩn hóa dữ liệu + PCA\n",
    f"- Số đặc trưng đưa vào PCA (sau mã hóa one-hot/ordinal): **{X.shape[1]}**",
    f"- Số mẫu (train): **{X.shape[0]}**",
    f"- Số PC cần để giải thích >=80% phương sai: **{so_pc_80}/{len(eigenvalues)}**",
    f"- Phương sai giải thích bởi PC1..PC5: "
    + ", ".join(f"PC{i+1}={ty_le_giai_thich[i]:.4f}" for i in range(5)),
    "\n## Top 5 đặc trưng đóng góp lớn nhất cho mỗi PC (trong top 5 PC)\n",
]
for i in range(5):
    top3 = df_dong_gop[df_dong_gop["PC"] == f"PC{i+1}"].head(5)
    dong_tomtat.append(f"**PC{i+1}** (giải thích {ty_le_giai_thich[i]:.2%} phương sai):")
    for _, row in top3.iterrows():
        dong_tomtat.append(f"  - `{row['dac_trung']}`: trọng số {row['trong_so']}")
    dong_tomtat.append("")

with open(f"{KET_QUA_DIR}/task_2-3_tomtat.md", "w", encoding="utf-8") as f:
    f.write("\n".join(dong_tomtat))

print("Đã lưu kết quả vào", KET_QUA_DIR)
