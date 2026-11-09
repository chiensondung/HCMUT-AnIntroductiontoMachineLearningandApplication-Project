"""Bước 03 (Chiến): xóa dòng khuyết ở cột khuyết <2%, tạo nhãn đa lớp + nhị phân, chia train/test 80/20.
Đọc:  output/step2.csv, output/step1_patient_nbr.csv (cùng thứ tự dòng, chỉ dùng để chia theo bệnh nhân)
Ghi:  output/train.csv (80.195 dòng), output/test.csv (20.049 dòng), giữ cột y_multi, y_bin"""
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

df = pd.read_csv("output/step2.csv", low_memory=False)
nhom = pd.read_csv("output/step1_patient_nbr.csv")["patient_nbr"]
assert len(nhom) == len(df), "step1_patient_nbr.csv phải cùng số dòng/thứ tự với step2.csv"
print("Kích thước step2:", df.shape)

ty_le_khuyet = df.isna().mean()
cot_khuyet_duoi_2pc = ty_le_khuyet[(ty_le_khuyet > 0) & (ty_le_khuyet < 0.02)].index.tolist()
giu = df[cot_khuyet_duoi_2pc].notna().all(axis=1)
print("Cột khuyết <2%:", cot_khuyet_duoi_2pc, "| xóa", int((~giu).sum()), "dòng")
df, nhom = df[giu].reset_index(drop=True), nhom[giu].reset_index(drop=True)

df["y_multi"] = df.pop("readmitted")                 # '<30' / '>30' / 'NO'
df["y_bin"] = (df["y_multi"] != "NO").astype(int)    # 1 = Readmitted (<30 hoặc >30), 0 = Not readmitted

# 80/20 phân tầng theo y_multi, không để một bệnh nhân nằm ở cả hai tập
splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, test_idx = next(splitter.split(df, df["y_multi"], nhom))
assert not set(nhom[train_idx]) & set(nhom[test_idx])

train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]
train_df.to_csv("output/train.csv", index=False)
test_df.to_csv("output/test.csv", index=False)

print("Kích thước sau xử lý:", df.shape, "| train:", len(train_df), "| test:", len(test_df))
for ten, phan in (("train", train_df), ("test", test_df)):
    print(ten, phan["y_multi"].value_counts(normalize=True).round(4).to_dict(),
          "| y_bin=1:", round(phan["y_bin"].mean(), 4))
print("Còn khuyết:", df.isna().sum()[lambda s: s > 0].to_dict())
