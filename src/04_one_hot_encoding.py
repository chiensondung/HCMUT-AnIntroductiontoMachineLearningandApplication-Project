"""One-Hot Encoding toàn bộ biến phân loại (gồm ICD-9 diag_1..3). Chạy: python src/04_one_hot_encoding.py
Đầu vào: data/diabetic_data_step1.csv (đã bỏ định danh). Đầu ra: ma trận thưa data/X_onehot.npz + data/X_onehot_cols.txt + data/y.csv"""
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder

ID_AS_CAT = ["admission_type_id", "discharge_disposition_id", "admission_source_id"]  # số nhưng thực chất là mã loại
df = pd.read_csv("data/diabetic_data_step1.csv", low_memory=False)
y = df.pop("readmitted")
cat = df.select_dtypes(exclude="number").columns.tolist() + ID_AS_CAT
df[cat] = df[cat].astype("string").fillna("Missing")  # khuyết thành 1 loại riêng
num = [c for c in df.columns if c not in cat]

enc = OneHotEncoder(sparse_output=True, dtype="int8")
Xc = enc.fit_transform(df[cat])
X = sparse.hstack([sparse.csr_matrix(df[num].to_numpy(dtype="float32")), Xc]).tocsr()
cols = num + enc.get_feature_names_out(cat).tolist()

assert X.shape == (len(df), len(cols))
assert Xc.sum() == len(df) * len(cat)  # mỗi dòng đúng 1 số 1 cho mỗi biến phân loại
sparse.save_npz("data/X_onehot.npz", X)
open("data/X_onehot_cols.txt", "w").write("\n".join(cols))
y.to_csv("data/y.csv", index=False)
print(f"{len(cat)} biến phân loại + {len(num)} số -> {X.shape[1]} cột; diag_1/2/3: "
      f"{[sum(c.startswith(d + '_') for c in cols) for d in ('diag_1', 'diag_2', 'diag_3')]}")
