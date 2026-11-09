"""Bước 02 (Sơn): thống kê tỉ lệ khuyết và loại thuộc tính khuyết > 50%.
Đọc:  output/step1.csv (bước 01, '?' đã thành NaN)
Ghi:  output/step2.csv (101766 x 44), output/missing_rate.csv"""
import pandas as pd

NGUONG = 0.5
df = pd.read_csv("output/step1.csv", low_memory=False)

ty_le = df.isna().mean().sort_values(ascending=False)
ty_le[ty_le > 0].mul(100).round(2).rename("khuyet_%").to_csv("output/missing_rate.csv")
print((ty_le[ty_le > 0] * 100).round(2))

bo = ty_le[ty_le > NGUONG].index.tolist()
print("\nBỏ (khuyết >50%):", bo)
df = df.drop(columns=bo)
print("Kích thước:", df.shape)
df.to_csv("output/step2.csv", index=False)
