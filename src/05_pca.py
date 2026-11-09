"""Bước 05 (Sơn): PCA tự cài đặt trên ma trận tương quan, lấy 60 thành phần. One-Hot đã làm ở bước 04.
Đọc:  output/X_train_onehot.npz, output/X_test_onehot.npz, output/onehot_columns.txt (bước 04)
Ghi:  output/X_train_pca.npy, output/X_test_pca.npy, output/pca_eigenvalues.npy,
      output/pca_components.npy, output/pca_feature_names.txt"""
import numpy as np
from scipy import sparse

N_COMPONENTS = 60
NGUONG_PHUONG_SAI = 1e-12

X_train = sparse.load_npz("output/X_train_onehot.npz").tocsr()
X_test = sparse.load_npz("output/X_test_onehot.npz").tocsr()
cols = open("output/onehot_columns.txt", encoding="utf-8").read().split("\n")
assert X_train.shape[1] == X_test.shape[1] == len(cols)
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
assert Z_train.shape == (n, N_COMPONENTS) and Z_test.shape == (X_test.shape[0], N_COMPONENTS)

np.save("output/X_train_pca.npy", Z_train)
np.save("output/X_test_pca.npy", Z_test)
np.save("output/pca_eigenvalues.npy", eigvals)
np.save("output/pca_components.npy", V)
open("output/pca_feature_names.txt", "w", encoding="utf-8").write("\n".join(cols))

print("\nĐã lưu Z_train:", Z_train.shape, "| Z_test:", Z_test.shape)
