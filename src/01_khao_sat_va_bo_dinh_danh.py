"""Khảo sát diabetic_data.csv và loại bỏ thuộc tính định danh. Chạy: python src/01_khao_sat_va_bo_dinh_danh.py"""
import pandas as pd

ID_COLS = ["encounter_id", "patient_nbr", "payer_code"]  # định danh / không mang ý nghĩa dự đoán

df = pd.read_csv("data/diabetic_data.csv", na_values="?", low_memory=False)  # '?' là giá trị khuyết
print("Kích thước gốc:", df.shape)
print("\nTỉ lệ khuyết (%):\n", (df.isna().mean() * 100).round(2)[lambda s: s > 0].sort_values(ascending=False))
print("\nPhân bố nhãn readmitted:\n", df["readmitted"].value_counts())
print("\nThuộc tính hằng số:", df.columns[df.nunique() == 1].tolist())

df = df.drop(columns=ID_COLS)
assert not set(ID_COLS) & set(df.columns)
print("\nKích thước sau khi bỏ định danh:", df.shape)
df.to_csv("data/diabetic_data_step1.csv", index=False)
