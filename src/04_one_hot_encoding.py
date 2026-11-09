import pandas as pd

IN = "output/diabetic_data_step1.csv"

df = pd.read_csv(IN, low_memory=False)
print("Kích thước:", df.shape)

# 3 cột này là số nhưng thực chất là mã loại, không có thứ tự
ID_AS_CAT = ["admission_type_id", "discharge_disposition_id", "admission_source_id"]

y = df.pop("readmitted")  # nhãn, không mã hóa
cat_cols = df.select_dtypes(exclude="number").columns.tolist() + ID_AS_CAT
num_cols = [c for c in df.columns if c not in cat_cols]

print("Số biến phân loại:", len(cat_cols))
print("Biến số giữ nguyên:", num_cols)
print(df[cat_cols].nunique().sort_values(ascending=False).head(8))

from scipy import sparse
from sklearn.preprocessing import OneHotEncoder

# Giá trị khuyết (NaN) thành một loại riêng "Missing"
df[cat_cols] = df[cat_cols].astype("string").fillna("Missing")

enc = OneHotEncoder(sparse_output=True, dtype="int8")
X_cat = enc.fit_transform(df[cat_cols])
cat_names = enc.get_feature_names_out(cat_cols).tolist()

print("\nSố cột One-Hot:", X_cat.shape[1])
for d in ["diag_1", "diag_2", "diag_3"]:
    print(d, "->", sum(n.startswith(d + "_") for n in cat_names), "cột")
print("Mỗi dòng có đúng 38 số 1:", X_cat.sum() == len(df) * len(cat_cols))

import os

X = sparse.hstack([sparse.csr_matrix(df[num_cols].to_numpy(dtype="float32")), X_cat]).tocsr()
all_cols = num_cols + cat_names
assert X.shape == (len(df), len(all_cols))

os.makedirs("output", exist_ok=True)
sparse.save_npz("output/X_onehot.npz", X)
open("output/X_onehot_cols.txt", "w").write("\n".join(all_cols))
y.to_csv("output/y.csv", index=False)
print("\nĐã lưu X:", X.shape)
