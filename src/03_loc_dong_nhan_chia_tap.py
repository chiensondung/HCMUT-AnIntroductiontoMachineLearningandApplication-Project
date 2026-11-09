"""Xóa dòng khuyết ở cột khuyết <2%, tạo nhãn đa lớp + nhị phân, chia train/test 80/20. Chạy sau bước 02."""
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

df = pd.read_csv("data/diabetic_data_step2.csv", low_memory=False)
# patient_nbr đã bị bỏ ở bước 01; lấy lại từ file gốc (step2 chưa xóa dòng nào nên cùng thứ tự) chỉ để chia nhóm
nhom = pd.read_csv("data/diabetic_data.csv", usecols=["patient_nbr"])["patient_nbr"]
assert len(nhom) == len(df)

ty_le = df.isna().mean()
cot_xoa_dong = ty_le[(ty_le > 0) & (ty_le < 0.02)].index.tolist()
giu = df[cot_xoa_dong].notna().all(axis=1)
print("Xóa dòng theo cột:", cot_xoa_dong, "| bỏ", (~giu).sum(), "dòng")
df, nhom = df[giu].reset_index(drop=True), nhom[giu].reset_index(drop=True)

df["y_multi"] = df.pop("readmitted")                 # '<30' / '>30' / 'NO'
df["y_bin"] = (df["y_multi"] != "NO").astype(int)    # 1 = Readmitted (<30 hoặc >30), 0 = Not readmitted

# 80/20 phân tầng theo nhãn đa lớp, không để một bệnh nhân nằm ở cả hai tập
tr, te = next(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(df, df["y_multi"], nhom))
assert not set(nhom[tr]) & set(nhom[te])
df.iloc[tr].to_csv("data/train.csv", index=False)
df.iloc[te].to_csv("data/test.csv", index=False)
print("Kích thước:", df.shape, "| train", len(tr), "| test", len(te))
for n, i in (("train", tr), ("test", te)):
    print(n, df.iloc[i]["y_multi"].value_counts(normalize=True).round(4).to_dict(), "| bin=1:", df.iloc[i]["y_bin"].mean().round(4))
print("Còn khuyết:", df.isna().sum()[lambda s: s > 0].to_dict())
