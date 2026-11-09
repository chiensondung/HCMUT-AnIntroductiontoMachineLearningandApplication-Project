"""Bước 04 (Dũng): One-Hot Encoding cho toàn bộ biến phân loại, fit trên train.
Đọc:  output/train.csv, output/test.csv (bước 03)
Ghi:  output/X_train_onehot.npz, output/X_test_onehot.npz, output/onehot_columns.txt,
      output/y_train.csv, output/y_test.csv (cột y_multi, y_bin)"""
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder

train = pd.read_csv("output/train.csv", low_memory=False)
test = pd.read_csv("output/test.csv", low_memory=False)
print("Train:", train.shape, "| Test:", test.shape)

LABELS = ["y_multi", "y_bin"]  # nhãn, không mã hóa
X_train_raw, X_test_raw = train.drop(columns=LABELS), test.drop(columns=LABELS)

# 3 cột này là số nhưng thực chất là mã loại, không có thứ tự
ID_AS_CAT = ["admission_type_id", "discharge_disposition_id", "admission_source_id"]
cat_cols = X_train_raw.select_dtypes(exclude="number").columns.tolist() + ID_AS_CAT
num_cols = [c for c in X_train_raw.columns if c not in cat_cols]

print("Số biến phân loại:", len(cat_cols))
print("Biến số giữ nguyên:", num_cols)
print(X_train_raw[cat_cols].nunique().sort_values(ascending=False).head(8))

# Giá trị khuyết (NaN) thành một loại riêng "Missing"
for part in (X_train_raw, X_test_raw):
    part[cat_cols] = part[cat_cols].astype("string").fillna("Missing")

# Fit trên train; loại chỉ có ở test (ví dụ mã ICD-9 hiếm) thành toàn 0
enc = OneHotEncoder(sparse_output=True, dtype="float64", handle_unknown="ignore")
Xc_train = enc.fit_transform(X_train_raw[cat_cols])
Xc_test = enc.transform(X_test_raw[cat_cols])
cat_names = enc.get_feature_names_out(cat_cols).tolist()

print("\nSố cột One-Hot:", Xc_train.shape[1])
for d in ["diag_1", "diag_2", "diag_3"]:
    print(d, "->", sum(n.startswith(d + "_") for n in cat_names), "cột")
print(f"Mỗi dòng train có đúng {len(cat_cols)} số 1:", Xc_train.sum() == len(train) * len(cat_cols))

X_train = sparse.hstack([sparse.csr_matrix(X_train_raw[num_cols].to_numpy("float64")), Xc_train]).tocsr()
X_test = sparse.hstack([sparse.csr_matrix(X_test_raw[num_cols].to_numpy("float64")), Xc_test]).tocsr()
all_cols = num_cols + cat_names
assert X_train.shape[1] == X_test.shape[1] == len(all_cols)

sparse.save_npz("output/X_train_onehot.npz", X_train)
sparse.save_npz("output/X_test_onehot.npz", X_test)
open("output/onehot_columns.txt", "w").write("\n".join(all_cols))
train[LABELS].to_csv("output/y_train.csv", index=False)
test[LABELS].to_csv("output/y_test.csv", index=False)
print("\nĐã lưu X_train:", X_train.shape, "| X_test:", X_test.shape)
