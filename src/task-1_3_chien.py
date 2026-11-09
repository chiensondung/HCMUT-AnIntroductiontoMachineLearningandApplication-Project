"""Task 1.3 (độc lập, không phụ thuộc src/01-03):
đọc trực tiếp diabetic_data.csv, xóa dòng khuyết <2%, tạo nhãn đa lớp (y_multi)
+ nhị phân (y_bin), chia train/test 80/20."""
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

df = pd.read_csv("data/diabetic_data.csv", na_values="?", low_memory=False)

ty_le_khuyet = df.isna().mean()
cot_khuyet_duoi_2pc = ty_le_khuyet[(ty_le_khuyet > 0) & (ty_le_khuyet < 0.02)].index.tolist()
print("Cột khuyết <2%:", cot_khuyet_duoi_2pc)

giu = df[cot_khuyet_duoi_2pc].notna().all(axis=1)
print(f"Xóa {int((~giu).sum())} dòng khuyết, còn {int(giu.sum())} dòng")
df = df[giu].reset_index(drop=True)

df["y_multi"] = df.pop("readmitted")                 # '<30' / '>30' / 'NO'
df["y_bin"] = (df["y_multi"] != "NO").astype(int)    # 1 = Readmitted, 0 = Not readmitted

# Chia 80/20, phân tầng theo y_multi, không để một bệnh nhân ở cả hai tập
splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, test_idx = next(splitter.split(df, df["y_multi"], df["patient_nbr"]))
assert not set(df.loc[train_idx, "patient_nbr"]) & set(df.loc[test_idx, "patient_nbr"])

train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]
train_df.to_csv("data/train_task1_3.csv", index=False)
test_df.to_csv("data/test_task1_3.csv", index=False)

print("Kích thước sau xử lý:", df.shape, "| train:", len(train_df), "| test:", len(test_df))
for name, part in (("train", train_df), ("test", test_df)):
    print(
        name,
        part["y_multi"].value_counts(normalize=True).round(4).to_dict(),
        "| y_bin=1:", round(part["y_bin"].mean(), 4),
    )
