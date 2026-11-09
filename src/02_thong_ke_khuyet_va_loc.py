"""Thống kê tỉ lệ khuyết và loại thuộc tính khuyết > 50%. Chạy sau bước 01."""
import pandas as pd

NGUONG = 0.5
df = pd.read_csv("data/diabetic_data_step1.csv", low_memory=False)  # '?' đã thành NaN khi lưu ở bước 01

ty_le = df.isna().mean().sort_values(ascending=False)
ty_le[ty_le > 0].mul(100).round(2).rename("khuyet_%").to_csv("data/ty_le_khuyet.csv")
print((ty_le[ty_le > 0] * 100).round(2))

bo = ty_le[ty_le > NGUONG].index.tolist()
print("\nBỏ (khuyết >50%):", bo)
df = df.drop(columns=bo)
print("Kích thước:", df.shape)
df.to_csv("data/diabetic_data_step2.csv", index=False)
