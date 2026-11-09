import os
import pandas as pd

df = pd.read_csv("data/diabetic_data.csv", na_values="?", low_memory=False)
print("Kích thước gốc:", df.shape)

ID_COLS = ["encounter_id", "patient_nbr", "payer_code"]

print("\nKhảo sát cột định danh:")
print(df[ID_COLS].head())
print("Số giá trị khác nhau:\n", df[ID_COLS].nunique())
print("Tỉ lệ khuyết (%):\n", (df[ID_COLS].isna().mean() * 100).round(2))

df = df.drop(columns=ID_COLS)
assert not set(ID_COLS) & set(df.columns)
print("\nKích thước sau khi bỏ định danh:", df.shape)
os.makedirs("output", exist_ok=True)
df.to_csv("output/diabetic_data_step1.csv", index=False)
