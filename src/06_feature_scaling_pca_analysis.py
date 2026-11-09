"""Bước 06 (Chiến): chuẩn hóa dữ liệu sau PCA, vẽ scree plot eigenvalues, phân tích top 5 vector riêng.
Đọc:  output/X_train_pca.npy, output/X_test_pca.npy,
      output/pca_eigenvalues.npy (toàn bộ trị riêng, giảm dần),
      output/pca_components.npy (d x 60, mỗi CỘT là một vector riêng),
      output/pca_feature_names.txt (d tên đặc trưng, cùng thứ tự hàng của pca_components)
Ghi:  output/X_train_pca_scaled.npy, output/X_test_pca_scaled.npy, output/eigenvalues.png"""
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

TOP_PC = 5
TOP_DAC_TRUNG = 10

X_train = np.load("output/X_train_pca.npy")
X_test = np.load("output/X_test_pca.npy")
eigvals = np.load("output/pca_eigenvalues.npy")
V = np.load("output/pca_components.npy")
ten_dac_trung = open("output/pca_feature_names.txt", encoding="utf-8").read().splitlines()
assert V.shape == (len(ten_dac_trung), X_train.shape[1]) and X_test.shape[1] == X_train.shape[1]
print("X_train_pca:", X_train.shape, "| X_test_pca:", X_test.shape, "| components:", V.shape)

# --- 1. Chuẩn hóa sau PCA: fit trên train, áp dụng sang test ---
scaler = StandardScaler().fit(X_train)
np.save("output/X_train_pca_scaled.npy", scaler.transform(X_train))
np.save("output/X_test_pca_scaled.npy", scaler.transform(X_test))

# --- 2. Scree plot ---
k = V.shape[1]
ty_le = eigvals / eigvals.sum()
luy_tich = np.cumsum(ty_le)
print(f"{k} thành phần giải thích {luy_tich[k - 1]:.2%} phương sai")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.plot(range(1, k + 1), eigvals[:k], "o-", markersize=3)
ax1.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="Eigenvalue = 1 (Kaiser)")
ax1.set_xlabel("Thành phần chính")
ax1.set_ylabel("Eigenvalue")
ax1.set_title(f"Scree plot ({k} thành phần đã giữ)")
ax1.legend()

ax2.plot(range(1, len(luy_tich) + 1), luy_tich, color="tab:orange")
ax2.axvline(k, color="gray", linestyle="--", linewidth=1, label=f"{k} thành phần")
ax2.set_xlabel("Số thành phần chính")
ax2.set_ylabel("Tỉ lệ phương sai tích lũy")
ax2.set_title("Phương sai tích lũy (toàn bộ trị riêng)")
ax2.legend()
fig.tight_layout()
fig.savefig("output/eigenvalues.png", dpi=150)
plt.close(fig)

# --- 3. Top 5 vector riêng: các đặc trưng có trọng số |loading| lớn nhất ---
for i in range(TOP_PC):
    v = V[:, i]
    idx = np.argsort(-np.abs(v))[:TOP_DAC_TRUNG]
    print(f"\nPC{i + 1}: eigenvalue = {eigvals[i]:.4f}, giải thích {ty_le[i]:.2%} phương sai")
    print(pd.DataFrame({"dac_trung": [ten_dac_trung[j] for j in idx],
                        "trong_so": v[idx].round(4)}).to_string(index=False))
