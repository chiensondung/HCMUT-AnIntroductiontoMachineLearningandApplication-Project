"""Bước 01 (Dũng): khảo sát và bỏ thuộc tính định danh.
Đọc:  data/diabetic_data.csv
Ghi:  output/step1.csv (101766 x 47), output/step1_patient_nbr.csv (để bước 03 chia tập theo bệnh nhân)"""
import os
import pandas as pd

df = pd.read_csv("data/diabetic_data.csv", na_values="?", low_memory=False)
print("Kích thước gốc:", df.shape)

ID_COLS = ["encounter_id", "patient_nbr", "payer_code"]

print("\nKhảo sát cột định danh:")
print(df[ID_COLS].head())
print("Số giá trị khác nhau:\n", df[ID_COLS].nunique())
print("Tỉ lệ khuyết (%):\n", (df[ID_COLS].isna().mean() * 100).round(2))

os.makedirs("output", exist_ok=True)
df[["patient_nbr"]].to_csv("output/step1_patient_nbr.csv", index=False)  # cùng thứ tự dòng với step1.csv

df = df.drop(columns=ID_COLS)
assert not set(ID_COLS) & set(df.columns)
print("\nKích thước sau khi bỏ định danh:", df.shape)
df.to_csv("output/step1.csv", index=False)
