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
| Nhân khẩu học | `race` (5 nhóm), `gender` (3 giá trị: `Female` 54.708, `Male` 47.055, `Unknown/Invalid` 3 dòng), `age` (10 khoảng 10 tuổi, đông nhất `[70-80)` với 26.068 dòng), `weight` (khoảng cân nặng, ví dụ `[75-100)`) | `weight` khuyết ≈ 96,86% |
| Thông tin lượt nhập viện | `admission_type_id` (8 mã), `discharge_disposition_id` (26 mã), `admission_source_id` (17 mã), `time_in_hospital` (1–14 ngày), `medical_specialty` (72 giá trị) | Ba cột `*_id` là mã phân loại, không phải số thứ tự có quan hệ thứ bậc |
| Xét nghiệm và thủ thuật | `num_lab_procedures`, `num_procedures`, `num_medications`, `number_diagnoses` (1–16), `max_glu_serum` (`Norm`, `>200`, `>300`), `A1Cresult` (`Norm`, `>7`, `>8`) | `max_glu_serum` khuyết ≈ 94,75%, `A1Cresult` khuyết ≈ 83,28% (khuyết nghĩa là không làm xét nghiệm) |
| Tiền sử sử dụng dịch vụ | `number_outpatient`, `number_emergency`, `number_inpatient` | Số lần trong năm trước lượt nhập viện hiện tại (tối đa lần lượt 42, 76, 21) |
| Chẩn đoán | `diag_1`, `diag_2`, `diag_3` | Mã ICD-9, số giá trị khác nhau rất lớn: 716 (`diag_1`), 748 (`diag_2`), 789 (`diag_3`) |
| Thuốc (23 cột) | `metformin`, `insulin`, `glipizide`, `glyburide`, `repaglinide`, `nateglinide`, `chlorpropamide`, `glimepiride`, `acetohexamide`, `tolbutamide`, `pioglitazone`, `rosiglitazone`, `acarbose`, `miglitol`, `troglitazone`, `tolazamide`, `examide`, `citoglipton`, `glyburide-metformin`, `glipizide-metformin`, `glimepiride-pioglitazone`, `metformin-rosiglitazone`, `metformin-pioglitazone` | Tối đa 4 mức: `No`, `Steady`, `Up`, `Down`. Một số cột ít mức hơn: `examide`, `citoglipton` chỉ có `No`; `tolazamide` có 3 mức; `acetohexamide`, `tolbutamide`, `troglitazone`, `glipizide-metformin`, `glimepiride-pioglitazone`, `metformin-rosiglitazone`, `metformin-pioglitazone` có 2 mức |
| Tổng hợp điều trị | `change` (`Ch`/`No`), `diabetesMed` (`Yes`/`No`) | Có thay đổi thuốc / có dùng thuốc đái tháo đường trong lượt nhập viện |
| Nhãn | `readmitted` | `NO` 54.864 (53,9%), `>30` 35.545 (34,9%), `<30` 11.357 (11,2%) |

## 1.3.3. Đặc điểm cần xử lý

1. **Giá trị khuyết** (tỉ lệ trên toàn bộ 101.766 dòng): `weight` 96,86%, `max_glu_serum` 94,75%, `A1Cresult` 83,28% (đều trên 80%); `medical_specialty` 49,08%; `payer_code` 39,56%; `race` 2,23%; `diag_3` 1,40%, `diag_2` 0,35%, `diag_1` 0,02% (đều dưới 2%). Pipeline loại các cột khuyết > 50% (bước 02) và xóa dòng khuyết ở các cột khuyết < 2% (bước 03, Mục 1.3.4); các cột còn lại (`race`, `medical_specialty`) giữ giá trị khuyết như một loại riêng `Missing` khi One-Hot.
2. **Thuộc tính hằng số hoặc gần hằng số**: `examide` và `citoglipton` chỉ nhận một giá trị duy nhất (`No`) trong toàn bộ dữ liệu nên không mang thông tin phân biệt; bảy cột thuốc khác chỉ có 2 mức (`acetohexamide`, `tolbutamide`, `troglitazone`, `glipizide-metformin`, `glimepiride-pioglitazone`, `metformin-rosiglitazone`, `metformin-pioglitazone`), hầu hết dòng đều là `No`. Sau One-Hot, `examide_No` và `citoglipton_No` toàn bằng 1 (phương sai 0) và bị bước 05 loại khỏi PCA (ngưỡng phương sai 1e-9), nên PCA chạy trên 2.394 trong 2.396 cột. Với ngưỡng cũ 1e-12, sai số làm tròn khiến hai cột này lọt vào PCA và chiếm riêng một thành phần chính.
3. **Biến phân loại bậc cao**: mã chẩn đoán ICD-9 (`diag_1/2/3`, 716 đến 789 giá trị), `medical_specialty` (72 giá trị) và ba cột `*_id` (`admission_type_id` 8 mã, `discharge_disposition_id` 26 mã, `admission_source_id` 17 mã) cần mã hóa phù hợp trước khi đưa vào mô hình. Pipeline chung mã hóa One-Hot toàn bộ 35 biến phân loại (bước 04), tạo 2.396 cột; bước 05 loại 2 cột hằng số rồi giảm 2.394 chiều xuống 60 thành phần bằng PCA.
4. **Mất cân bằng lớp**: lớp `<30` (tái nhập viện trong 30 ngày) chỉ chiếm khoảng 11,2% (11.357 dòng) — cần độ đo đánh giá phù hợp (Recall, F1, ROC-AUC/PR-AUC) thay vì chỉ accuracy. Ở bài toán nhị phân hai lớp gần cân bằng (`y_bin = 1` chiếm 46,19% tập train).
5. **Nhiều lượt nhập viện trên cùng một bệnh nhân**: 101.766 dòng nhưng chỉ có 71.518 bệnh nhân khác nhau → khi chia train/test cần nhóm theo `patient_nbr` để một bệnh nhân không xuất hiện ở cả hai tập, tránh rò rỉ dữ liệu (làm ở bước 03).

## 1.3.4. Xử lý theo yêu cầu Task 1.3

Thực hiện bằng bước 03 (`src/03_data_preprocessing_splitting.py`) của pipeline chung, đọc `output/step2.csv` (sau bước 01–02: bỏ định danh, bỏ cột khuyết > 50%) và `output/step1_patient_nbr.csv`, gồm 3 bước:

1. **Lọc xóa dòng khuyết <2%**: tính tỉ lệ khuyết từng cột, xác định các cột có tỉ lệ khuyết trong (0%, 2%) là `diag_1`, `diag_2`, `diag_3`; xóa toàn bộ dòng khuyết ở các cột này (xóa 1.522 dòng).
2. **Tạo 2 bộ nhãn mục tiêu**:
   - Đa lớp (`y_multi`): giữ nguyên 3 giá trị gốc của `readmitted`: `<30`, `>30`, `NO`.
   - Nhị phân (`y_bin`): `1` nếu `y_multi` khác `NO` (Readmitted), `0` nếu `y_multi = NO` (Not readmitted).
3. **Chia Train/Test tỉ lệ 80/20** bằng `StratifiedGroupKFold` (5 phần, lấy 1 phần làm test, `shuffle=True`, `random_state=42`): phân tầng theo `y_multi`, nhóm theo `patient_nbr` để không bệnh nhân nào xuất hiện ở cả hai tập (có `assert` kiểm tra).

**Kết quả chạy chuẩn** (Google Colab, scikit-learn 1.6.1, pandas 2.2.3, numpy 2.4.6; cách chia phụ thuộc phiên bản scikit-learn):

| Bước | Kích thước |
| --- | --- |
| Dữ liệu gốc | 101.766 × 50 |
| Sau bước 01–02 (bỏ định danh, bỏ cột khuyết > 50%) | 101.766 × 44 |
| Sau khi xóa dòng khuyết ở `diag_1/2/3` (bỏ 1.522 dòng), thêm `y_multi`, `y_bin`, bỏ `readmitted` | 100.244 × 45 |
| Train | 79.726 dòng (79,5%) |
| Test | 20.518 dòng (20,5%) |

Vì phải giữ nguyên cả nhóm bệnh nhân nên tỉ lệ chia thực tế không đúng 80/20 và phân phối nhãn của hai tập lệch nhau dưới 1 điểm phần trăm:

| Tập | NO | >30 | <30 | y_bin = 1 |
| --- | --- | --- | --- | --- |
| Train | 53,81% | 35,03% | 11,16% | 46,19% |
| Test | 53,23% | 35,30% | 11,47% | 46,77% |

Dữ liệu được ghi vào `output/train.csv` và `output/test.csv` (notebook tổng hợp tự sinh lại, không lưu bản sao riêng).
