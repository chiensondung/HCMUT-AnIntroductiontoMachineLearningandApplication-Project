"""One-Hot (fit trên train, transform test) rồi PCA tự cài đặt trên ma trận tương quan, lấy 60
thành phần. Chạy sau bước 03.

Ghi chú: One-Hot ở src/04 (nhánh Dung) chạy trên toàn bộ dữ liệu (chưa lọc >50% khuyết, chưa
chia train/test) nên không dùng được cho PCA ở đây (sẽ rò rỉ nhãn qua việc fit encoder trên cả
test) -> tự làm lại One-Hot trong file này, fit trên train, transform test với
handle_unknown="ignore". Cần thống nhất với Dũng.
"""
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder

N_COMPONENTS = 60
NGUONG_PHUONG_SAI = 1e-12
ID_AS_CAT = ["admission_type_id", "discharge_disposition_id", "admission_source_id"]

train = pd.read_csv("data/train.csv", low_memory=False)
test = pd.read_csv("data/test.csv", low_memory=False)

y_train = train[["y_multi", "y_bin"]]
y_test = test[["y_multi", "y_bin"]]
X_train_raw = train.drop(columns=["y_multi", "y_bin"])
X_test_raw = test.drop(columns=["y_multi", "y_bin"])

cat_cols = X_train_raw.select_dtypes(exclude="number").columns.tolist() + ID_AS_CAT
num_cols = [c for c in X_train_raw.columns if c not in cat_cols]
print("Số biến phân loại:", len(cat_cols), "| số biến số:", len(num_cols))

# Giá trị khuyết (NaN) thành một loại riêng "Missing", như src/04
X_train_raw[cat_cols] = X_train_raw[cat_cols].astype("string").fillna("Missing")
X_test_raw[cat_cols] = X_test_raw[cat_cols].astype("string").fillna("Missing")

enc = OneHotEncoder(sparse_output=True, dtype="float64", handle_unknown="ignore")
X_train_cat = enc.fit_transform(X_train_raw[cat_cols])
X_test_cat = enc.transform(X_test_raw[cat_cols])
cat_names = enc.get_feature_names_out(cat_cols).tolist()

X_train_num = sparse.csr_matrix(X_train_raw[num_cols].to_numpy(dtype="float64"))
X_test_num = sparse.csr_matrix(X_test_raw[num_cols].to_numpy(dtype="float64"))

X_train = sparse.hstack([X_train_num, X_train_cat]).tocsr()
X_test = sparse.hstack([X_test_num, X_test_cat]).tocsr()
cols = num_cols + cat_names
assert X_train.shape[1] == len(cols) and X_test.shape[1] == len(cols)
print("One-Hot + số (train):", X_train.shape, "| (test):", X_test.shape)

n = X_train.shape[0]

# Trung bình và phương sai từng cột trên TRAIN, tính trực tiếp từ ma trận thưa (không densify)
mu = np.asarray(X_train.mean(axis=0)).ravel()
mean_sq = np.asarray(X_train.multiply(X_train).mean(axis=0)).ravel()
var = mean_sq - mu**2

giu = var > NGUONG_PHUONG_SAI
print("Bỏ", (~giu).sum(), "cột phương sai 0 trên train:", [c for c, g in zip(cols, giu) if not g])

X_train, X_test = X_train[:, giu], X_test[:, giu]
mu, sigma = mu[giu], np.sqrt(var[giu])
cols = [c for c, g in zip(cols, giu) if g]
d = X_train.shape[1]
print("Số chiều sau khi bỏ cột phương sai 0:", d)

# Ma trận hiệp phương sai trực tiếp từ ma trận thưa: Cov = X^T X / n - mu mu^T (d x d, không densify X)
cov = (X_train.T @ X_train).toarray() / n - np.outer(mu, mu)
corr = cov / np.outer(sigma, sigma)
corr = (corr + corr.T) / 2  # đối xứng hóa, khử sai số số học

eigvals, eigvecs = np.linalg.eigh(corr)
order = np.argsort(eigvals)[::-1]
eigvals, eigvecs = eigvals[order], eigvecs[:, order]
eigvals = np.clip(eigvals, 0, None)  # sai số số học có thể cho trị riêng âm rất nhỏ

V = eigvecs[:, :N_COMPONENTS]
ty_le_giai_thich = eigvals[:N_COMPONENTS] / eigvals.sum()
ty_le_luy_tich = np.cumsum(eigvals) / eigvals.sum()
so_tp_95 = int(np.searchsorted(ty_le_luy_tich, 0.95) + 1)
print(f"\nTỉ lệ phương sai giải thích bởi {N_COMPONENTS} thành phần: {ty_le_giai_thich.sum():.4f}")
print("Số thành phần cần để đạt 95%:", so_tp_95)

# Kiểm tra bằng sklearn.decomposition.PCA trên một mẫu con (toàn bộ train densify quá nặng RAM)
from sklearn.decomposition import PCA

rng = np.random.default_rng(42)
mau = rng.choice(n, size=min(8000, n), replace=False)
X_mau_std = (np.asarray(X_train[mau].todense()) - mu) / sigma
pca_sklearn = PCA(n_components=N_COMPONENTS, svd_solver="full").fit(X_mau_std)

# sklearn.PCA tự trừ trung bình của MẪU CON (không phải mu của train) và dùng mẫu số n-1
# (phương sai mẫu không chệch) -> làm giống vậy để trị riêng khớp với pca_sklearn.explained_variance_
X_mau_c = X_mau_std - X_mau_std.mean(axis=0)
cov_mau = X_mau_c.T @ X_mau_c / (len(mau) - 1)
cov_mau = (cov_mau + cov_mau.T) / 2
eigvals_mau, _ = np.linalg.eigh(cov_mau)
eigvals_mau = np.clip(eigvals_mau[::-1], 0, None)[:N_COMPONENTS]

le_sai_so = np.abs(eigvals_mau - pca_sklearn.explained_variance_).max()
print(f"Kiểm tra trên mẫu {len(mau)} dòng: lệch tối đa trị riêng so với sklearn.PCA = {le_sai_so:.2e}")
assert le_sai_so < 1e-6, "Trị riêng tự tính không khớp sklearn.decomposition.PCA"

# Chiếu lên 60 thành phần không densify: Z = ((X - mu)/sigma) @ V = X @ W - mu @ W, W = V / sigma
W = V / sigma[:, None]
Z_train = np.asarray(X_train @ W) - mu @ W
Z_test = np.asarray(X_test @ W) - mu @ W
assert Z_train.shape == (len(train), N_COMPONENTS) and Z_test.shape == (len(test), N_COMPONENTS)

np.save("data/X_train_pca.npy", Z_train)
np.save("data/X_test_pca.npy", Z_test)
np.save("data/pca_eigenvalues.npy", eigvals)
np.save("data/pca_components.npy", V)
open("data/pca_feature_names.txt", "w", encoding="utf-8").write("\n".join(cols))
y_train.to_csv("data/y_train.csv", index=False)
y_test.to_csv("data/y_test.csv", index=False)
pd.DataFrame({
    "thanh_phan": np.arange(1, N_COMPONENTS + 1),
    "tri_rieng": eigvals[:N_COMPONENTS],
    "ty_le_giai_thich": ty_le_giai_thich,
    "ty_le_luy_tich": ty_le_luy_tich[:N_COMPONENTS],
}).to_csv("data/pca_explained_variance.csv", index=False)

print("\nĐã lưu Z_train:", Z_train.shape, "| Z_test:", Z_test.shape)
