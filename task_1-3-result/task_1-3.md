# Mục 1.3: Mô tả chi tiết tập dữ liệu lâm sàng gốc

## 1.3.1. Nguồn dữ liệu

Bộ dữ liệu *Diabetes 130-US hospitals for years 1999–2008* (Strack và cộng sự, 2014; kho UCI), lấy từ hồ sơ của 130 bệnh viện và mạng lưới y tế tại Hoa Kỳ, chỉ gồm các lượt nhập viện của bệnh nhân đái tháo đường (nằm viện 1–14 ngày, có chỉ định xét nghiệm và/hoặc dùng thuốc trong lượt nhập viện).

File gốc: `data/diabetic_data.csv`.

- Kích thước: **101.766 dòng × 50 cột**, mỗi dòng ứng với một lượt nhập viện.
- Số bệnh nhân khác nhau: **71.518** (`patient_nbr`) — một bệnh nhân có thể xuất hiện ở nhiều dòng (nhiều lượt nhập viện khác nhau).
- Giá trị khuyết được mã hóa bằng ký tự `?` (không phải `NaN` chuẩn), cần khai báo khi đọc dữ liệu (`na_values="?"`).

Danh sách đầy đủ 50 cột (lấy trực tiếp khi đọc `diabetic_data.csv`):

```
encounter_id, patient_nbr, race, gender, age, weight, admission_type_id,
discharge_disposition_id, admission_source_id, time_in_hospital, payer_code,
medical_specialty, num_lab_procedures, num_procedures, num_medications,
number_outpatient, number_emergency, number_inpatient, diag_1, diag_2, diag_3,
number_diagnoses, max_glu_serum, A1Cresult, metformin, repaglinide, nateglinide,
chlorpropamide, glimepiride, acetohexamide, glipizide, glyburide, tolbutamide,
pioglitazone, rosiglitazone, acarbose, miglitol, troglitazone, tolazamide,
examide, citoglipton, insulin, glyburide-metformin, glipizide-metformin,
glimepiride-pioglitazone, metformin-rosiglitazone, metformin-pioglitazone,
change, diabetesMed, readmitted
```

Ví dụ 2 dòng dữ liệu đầu (rút gọn để minh họa định dạng, giá trị khuyết hiển thị là `?`):

| encounter_id | patient_nbr | race | gender | age | weight | ... | diag_1 | diag_2 | diag_3 | ... | change | diabetesMed | readmitted |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2278392 | 8222157 | Caucasian | Female | [0-10) | ? | ... | 250.83 | ? | ? | ... | No | No | NO |
| 149190 | 55629189 | Caucasian | Female | [10-20) | ? | ... | 276 | 250.01 | 255 | ... | Ch | Yes | >30 |

## 1.3.2. Các nhóm thuộc tính (50 cột)

| Nhóm | Thuộc tính | Ghi chú |
| --- | --- | --- |
| Định danh | `encounter_id`, `patient_nbr`, `payer_code` | Không mang ý nghĩa dự đoán / có nguy cơ rò rỉ dữ liệu |
| Nhân khẩu học | `race` (5 nhóm), `gender` (3 giá trị, có 3 dòng `Unknown/Invalid`), `age` (10 khoảng 10 tuổi, đông nhất `[70-80)`), `weight` | `weight` khuyết ≈ 96,86% |
| Thông tin lượt nhập viện | `admission_type_id` (8 mã), `discharge_disposition_id` (26 mã), `admission_source_id` (17 mã), `time_in_hospital` (1–14 ngày), `medical_specialty` | Ba cột `*_id` là mã phân loại, không phải số thứ tự có quan hệ thứ bậc |
| Xét nghiệm và thủ thuật | `num_lab_procedures`, `num_procedures`, `num_medications`, `number_diagnoses` (1–16), `max_glu_serum`, `A1Cresult` | `max_glu_serum` khuyết ≈ 94,75%, `A1Cresult` khuyết ≈ 83,28% |
| Tiền sử sử dụng dịch vụ | `number_outpatient`, `number_emergency`, `number_inpatient` | Số lần trong năm trước lượt nhập viện hiện tại |
| Chẩn đoán | `diag_1`, `diag_2`, `diag_3` | Mã ICD-9, số giá trị khác nhau rất lớn (biến phân loại bậc cao) |
| Thuốc (23 cột) | `metformin`, `insulin`, `glipizide`, `glyburide`, `repaglinide`, `nateglinide`, `chlorpropamide`, `glimepiride`, `acetohexamide`, `tolbutamide`, `pioglitazone`, `rosiglitazone`, `acarbose`, `miglitol`, `troglitazone`, `tolazamide`, `examide`, `citoglipton`, `glyburide-metformin`, `glipizide-metformin`, `glimepiride-pioglitazone`, `metformin-rosiglitazone`, `metformin-pioglitazone` | Mỗi cột có 4 mức: `No`, `Steady`, `Up`, `Down` |
| Tổng hợp điều trị | `change` (`Ch`/`No`), `diabetesMed` (`Yes`/`No`) | Có thay đổi thuốc / có dùng thuốc đái tháo đường trong lượt nhập viện |
| Nhãn | `readmitted` | `NO` 54.864 (53,9%), `>30` 35.545 (34,9%), `<30` 11.357 (11,2%) |

## 1.3.3. Đặc điểm cần xử lý

1. **Giá trị khuyết**: `weight`, `max_glu_serum`, `A1Cresult` khuyết trên 80%; `medical_specialty` khuyết ≈ 49,08%; `payer_code` khuyết ≈ 39,6%; `race` khuyết ≈ 2,23%; `diag_1`, `diag_2`, `diag_3` khuyết dưới 1,5%.
2. **Thuộc tính hằng số**: `examide` và `citoglipton` chỉ nhận một giá trị duy nhất (`No`) trong toàn bộ dữ liệu — không mang thông tin phân biệt, có thể loại bỏ.
3. **Biến phân loại bậc cao**: mã chẩn đoán ICD-9 (`diag_1/2/3`) và ba cột `*_id` (`admission_type_id`, `discharge_disposition_id`, `admission_source_id`) cần gom nhóm hoặc mã hóa phù hợp trước khi đưa vào mô hình.
4. **Mất cân bằng lớp**: lớp `<30` (tái nhập viện trong 30 ngày) chỉ chiếm khoảng 11,2% — cần kỹ thuật xử lý mất cân bằng và độ đo đánh giá phù hợp (Recall, F1, ROC-AUC/PR-AUC).
5. **Nhiều lượt nhập viện trên cùng một bệnh nhân**: 101.766 dòng nhưng chỉ có 71.518 bệnh nhân khác nhau → khi chia train/test cần nhóm theo `patient_nbr` để một bệnh nhân không xuất hiện ở cả hai tập, tránh rò rỉ dữ liệu.

## 1.3.4. Xử lý theo yêu cầu Task 1.3

Thực hiện độc lập trên `data/diabetic_data.csv` (script `2_task-1_3_chien.py`), gồm 3 bước:

1. **Lọc xóa dòng khuyết <2%**: tính tỉ lệ khuyết từng cột, xác định các cột có tỉ lệ khuyết trong (0%, 2%) là `diag_1`, `diag_2`, `diag_3`; xóa toàn bộ dòng khuyết ở các cột này.
2. **Tạo 2 bộ nhãn mục tiêu**:
   - Đa lớp (`y_multi`): giữ nguyên 3 giá trị gốc của `readmitted` — `<30`, `>30`, `NO`.
   - Nhị phân (`y_bin`): `1` nếu `y_multi` khác `NO` (Readmitted), `0` nếu `y_multi = NO` (Not readmitted).
3. **Chia Train/Test tỉ lệ 80/20**: phân tầng theo `y_multi`, nhóm theo `patient_nbr` để không bệnh nhân nào xuất hiện ở cả hai tập (`random_state=42`).

**Kết quả chạy thực tế:**

| Bước | Kích thước |
| --- | --- |
| Dữ liệu gốc | 101.766 × 50 |
| Sau khi xóa dòng khuyết ở `diag_1/2/3` (bỏ 1.522 dòng) | 100.244 × 51 (thêm `y_multi`, `y_bin`) |
| Train (80%) | 80.195 dòng |
| Test (20%) | 20.049 dòng |

Phân phối nhãn giống nhau giữa hai tập (chứng tỏ chia phân tầng đúng):

| Tập | NO | >30 | <30 | y_bin = 1 |
| --- | --- | --- | --- | --- |
| Train | 53,69% | 35,09% | 11,22% | 46,31% |
| Test | 53,69% | 35,09% | 11,22% | 46,31% |

Dữ liệu được lưu tại `data/train_task1_3.csv` và `data/test_task1_3.csv`.
