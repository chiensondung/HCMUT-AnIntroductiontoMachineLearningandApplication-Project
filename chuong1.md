# Chương 1: Giới thiệu bài toán và Tổng quan nghiên cứu

## 1.1. Đặt vấn đề và mục tiêu bài toán

### 1.1.1. Đặt vấn đề
Đái tháo đường là bệnh mạn tính cần theo dõi và điều trị liên tục. Sau khi xuất viện, nhiều bệnh nhân phải nhập viện lại vì điều trị chưa ổn định, biến chứng hoặc thiếu chăm sóc sau xuất viện. Tái nhập viện trong vòng 30 ngày làm tăng chi phí y tế, gây quá tải cho bệnh viện và thường được dùng như một chỉ báo chất lượng chăm sóc. Nếu nhận diện được nhóm nguy cơ cao ngay lúc xuất viện, bệnh viện có thể ưu tiên nguồn lực theo dõi, tái khám và can thiệp cho nhóm này.

Báo cáo dùng bộ dữ liệu *Diabetes 130-US hospitals for years 1999–2008* (UCI): 101.766 lượt nhập viện của bệnh nhân đái tháo đường (71.518 bệnh nhân) tại 130 bệnh viện Hoa Kỳ, gồm 50 thuộc tính (nhân khẩu học, thông tin nhập viện, xét nghiệm, chẩn đoán, 23 thuốc điều trị và nhãn `readmitted`). Dữ liệu có ba khó khăn chính:
- **Giá trị khuyết nhiều**: `weight` ≈ 96,9%, `max_glu_serum` ≈ 94,7%, `A1Cresult` ≈ 83,3%, `medical_specialty` ≈ 49,1%, `payer_code` ≈ 39,6%.
- **Biến phân loại bậc cao**: mã chẩn đoán `diag_1`, `diag_2`, `diag_3`.
- **Mất cân bằng lớp**: nhãn `<30` chỉ chiếm ≈ 11,2%.

### 1.1.2. Phát biểu bài toán
Cho vector đặc trưng **x** của một lượt nhập viện, dự đoán nhãn `readmitted` ∈ {`<30`, `>30`, `NO`}. Hai bộ nhãn được xây dựng:
- **Đa lớp** (`y_multi`): giữ nguyên `<30` (tái nhập viện trong 30 ngày), `>30` (sau 30 ngày), `NO` (không tái nhập viện).
- **Nhị phân** (`y_bin`): `1` nếu tái nhập viện (`<30` hoặc `>30`), `0` nếu `NO`.

Vì nhãn `<30` hiếm, đánh giá dùng Recall, F1, ROC-AUC và PR-AUC thay vì chỉ Accuracy.

### 1.1.3. Mục tiêu
1. Khảo sát dữ liệu gốc và tiền xử lý: bỏ thuộc tính định danh, xử lý giá trị khuyết, mã hóa biến phân loại.
2. Xây dựng và so sánh các mô hình (ví dụ Logistic Regression, Random Forest, Gradient Boosting) cho bài toán dự đoán tái nhập viện.
3. Xử lý mất cân bằng lớp và đánh giá bằng các độ đo phù hợp.
4. Phân tích tầm quan trọng của đặc trưng để rút ra các yếu tố liên quan đến nguy cơ tái nhập viện.

### 1.1.4. Bước đầu: loại bỏ thuộc tính định danh
Ba thuộc tính sau không mang ý nghĩa dự đoán nên bị loại (mã: `src/01_khao_sat_va_bo_dinh_danh.py`):

| Thuộc tính | Lý do loại bỏ |
|---|---|
| `encounter_id` | Mã lượt nhập viện, duy nhất cho mỗi dòng (101.766 giá trị khác nhau); chỉ là khóa, không khái quát được. |
| `patient_nbr` | Mã bệnh nhân (71.518 giá trị khác nhau); giữ lại dễ khiến mô hình ghi nhớ bệnh nhân và rò rỉ thông tin giữa tập huấn luyện và kiểm tra. |
| `payer_code` | Mã bên thanh toán; khuyết ≈ 39,6% và là thông tin hành chính, ít liên quan lâm sàng. |

Kết quả: 101.766 × 50 → **101.766 × 47**, lưu tại `data/diabetic_data_step1.csv`.

### 1.1.5. Bước tiếp: thống kê khuyết và lọc thuộc tính khuyết > 50%
Mã: `src/02_thong_ke_khuyet_va_loc.py` (bảng đầy đủ: `data/ty_le_khuyet.csv`). Ngưỡng loại bỏ là 50%.

| Thuộc tính | Tỉ lệ khuyết | Xử lý |
|---|---|---|
| `weight` | 96,86% | **Loại bỏ** |
| `max_glu_serum` | 94,75% | **Loại bỏ** |
| `A1Cresult` | 83,28% | **Loại bỏ** |
| `medical_specialty` | 49,08% | Giữ (sát ngưỡng, xử lý ở bước sau) |
| `race`, `diag_3`, `diag_2`, `diag_1` | 2,23% / 1,40% / 0,35% / 0,02% | Giữ (khuyết ít) |

Kết quả: 101.766 × 47 → **101.766 × 44**, lưu tại `data/diabetic_data_step2.csv`.

### 1.1.6. Xóa dòng khuyết nhẹ, tạo nhãn và chia tập
Mã: `src/03_loc_dong_nhan_chia_tap.py`.
- Cột khuyết < 2% là `diag_1`, `diag_2`, `diag_3`: xóa các dòng khuyết, bỏ 1.522 dòng, còn **100.244** dòng. `race` (2,23%) vượt ngưỡng 2% nên không xóa dòng; `race` và `medical_specialty` (49,08%) còn khuyết, sẽ được xử lý ở bước sau.
- Tạo `y_multi` và `y_bin`, bỏ cột `readmitted` khỏi tập đặc trưng. Kích thước: 100.244 × 45 (43 đặc trưng + 2 nhãn).
- Chia **80/20** (train 80.195 / test 20.049), phân tầng theo `y_multi`, `random_state=42`, và không để một bệnh nhân xuất hiện ở cả hai tập (nhóm theo `patient_nbr`, lấy lại từ file gốc chỉ để chia). Tỉ lệ nhãn hai tập giống nhau: NO 53,7%, >30 35,1%, <30 11,2%; `y_bin=1` là 46,3%. Tập validation sẽ được tách từ train (hoặc cross-validation) khi huấn luyện. Lưu tại `data/train.csv`, `data/test.csv`.

## 1.2. Khảo sát các nghiên cứu liên quan

Các nghiên cứu học máy về đái tháo đường có thể chia thành hai hướng: (i) chẩn đoán/dự đoán mắc bệnh trên dữ liệu nhỏ, và (ii) dự đoán biến cố lâm sàng như nhập viện hay tái nhập viện trên dữ liệu bệnh án điện tử lớn. Bài toán của báo cáo này thuộc hướng thứ hai.

**Dữ liệu nhỏ, dự đoán mắc bệnh.** Nhiều công trình dùng bộ Pima Indians Diabetes (PIDD, khoảng 700 bệnh nhân). Khanam và Foo (2021) so sánh các thuật toán học máy trên bộ dữ liệu này; Dutta và cộng sự (2018) dùng hồi quy logistic, SVM và rừng ngẫu nhiên, đạt độ chính xác khoảng 84% và phân tích tầm quan trọng đặc trưng; Sisodia và Sisodia (2018) dùng Naive Bayes và các bộ phân loại khác. Một số báo cáo đạt tới 90% (Maniruzzaman và cộng sự, 2018, nhấn mạnh vai trò của giá trị khuyết và ngoại lai). Hạn chế chung là kích thước mẫu nhỏ, dễ quá khớp, nên độ chính xác cao khó khái quát hóa.

**Dữ liệu bệnh án lớn.** Brisimi và cộng sự (2019) dùng dữ liệu hồ sơ sức khỏe điện tử khoảng 40.000 bệnh nhân vùng Boston, kết hợp phân cụm trước khi phân loại, đạt độ chính xác khoảng 89%. Hasan và cộng sự (2020) đề xuất tổ hợp (ensemble) kNN, cây quyết định, rừng ngẫu nhiên, Naive Bayes và MLP, lấy trung bình có trọng số theo AUC của từng mô hình.

**Dữ liệu tái nhập viện.** Strack và cộng sự (2014) công bố bộ dữ liệu 130 bệnh viện Hoa Kỳ (1999–2008) và phân tích tác động của xét nghiệm HbA1c lên tỉ lệ tái nhập viện – đây chính là nguồn dữ liệu của báo cáo. Wang (Stanford CS229) áp dụng trên bộ này kNN, hồi quy logistic, cây quyết định, rừng ngẫu nhiên, mạng nơ-ron và mô hình tổ hợp, với PCA giảm chiều sau mã hóa one-hot (từ hơn 2.000 chiều xuống 60). Kết quả: độ chính xác khoảng 38–40% cho bài toán 3 lớp và khoảng 60–63% cho bài toán nhị phân; mạng nơ-ron tốt nhất nhưng chỉ nhỉnh hơn kNN không đáng kể. Tác giả chỉ ra các điểm yếu: mất cân bằng lớp khiến mô hình thiên về dự đoán "không tái nhập viện", độ chính xác trung bình theo lớp thấp, PCA làm mất khả năng diễn giải, và chưa thử huấn luyện trên dữ liệu cân bằng.

**Khoảng trống và hướng của báo cáo.** Từ khảo sát trên, báo cáo này (1) gộp nhãn thành nhị phân tái nhập viện trong 30 ngày, (2) xử lý mất cân bằng lớp và đánh giá bằng Recall/F1/AUC thay vì chỉ Accuracy, (3) tránh PCA trên toàn bộ one-hot để giữ khả năng diễn giải, ưu tiên mô hình cây/rừng và phân tích tầm quan trọng đặc trưng.

**Tài liệu tham khảo**
1. J. J. Khanam, S. Y. Foo. A comparison of machine learning algorithms for diabetes prediction. *ICT Express*, 7(4):432–439, 2021.
2. Md. Maniruzzaman et al. Accurate diabetes risk stratification using machine learning: Role of missing value and outliers. *J. Medical Systems*, 42(5):92, 2018.
3. T. S. Brisimi et al. Predicting diabetes-related hospitalizations based on electronic health records. *Stat. Methods in Medical Research*, 28(12):3667–3682, 2019.
4. D. Dutta, D. Paul, P. Ghosh. Analysing feature importances for diabetes prediction using machine learning. *IEEE IEMCON*, 924–928, 2018.
5. D. Sisodia, D. S. Sisodia. Prediction of diabetes using classification algorithms. *Procedia Computer Science*, 132:1578–1585, 2018.
6. Md. K. Hasan et al. Diabetes prediction using ensembling of different machine learning classifiers. *IEEE Access*, 8:76516–76531, 2020.
7. B. Strack et al. Impact of HbA1c measurement on hospital readmission rates: Analysis of 70,000 clinical database patient records. *BioMed Research International*, 2014:781670, 2014.
8. D. Wang. Predicting diabetes health outcomes using a variety of models. Stanford CS229 project report.

## 1.3. Mô tả chi tiết tập dữ liệu lâm sàng gốc

**Nguồn.** *Diabetes 130-US hospitals for years 1999–2008* (Strack và cộng sự, 2014; kho UCI): dữ liệu của 130 bệnh viện và mạng lưới y tế tại Hoa Kỳ, chỉ gồm các lượt nhập viện của bệnh nhân đái tháo đường (nằm viện 1–14 ngày, có xét nghiệm và dùng thuốc). File `diabetic_data.csv` có **101.766 dòng × 50 cột**, mỗi dòng là một lượt nhập viện; giá trị khuyết được mã hóa bằng `?`. Có 71.518 bệnh nhân khác nhau, tức một bệnh nhân có thể xuất hiện nhiều lần.

**Các nhóm thuộc tính (50 cột).**

| Nhóm | Thuộc tính | Ghi chú |
|---|---|---|
| Định danh | `encounter_id`, `patient_nbr`, `payer_code` | Bị loại ở bước 1.1.4 |
| Nhân khẩu học | `race` (5 nhóm), `gender` (3 giá trị, trong đó 3 dòng `Unknown/Invalid`), `age` (10 khoảng 10 tuổi, đông nhất `[70-80)`), `weight` | `weight` khuyết 96,86% |
| Thông tin lượt nhập viện | `admission_type_id` (8), `discharge_disposition_id` (26), `admission_source_id` (17), `time_in_hospital` (1–14 ngày), `medical_specialty` | Ba cột `*_id` là mã phân loại, không phải số thứ tự |
| Xét nghiệm và thủ thuật | `num_lab_procedures`, `num_procedures`, `num_medications`, `number_diagnoses` (1–16), `max_glu_serum`, `A1Cresult` | Hai cột xét nghiệm khuyết 94,75% và 83,28% |
| Tiền sử sử dụng dịch vụ | `number_outpatient`, `number_emergency`, `number_inpatient` | Số lần trong năm trước lượt nhập viện |
| Chẩn đoán | `diag_1`, `diag_2`, `diag_3` | Mã ICD-9, số giá trị khác nhau rất lớn |
| Thuốc (23 cột) | `metformin`, `insulin`, `glipizide`, `glyburide`, … | Mỗi cột có 4 mức: `No`, `Steady`, `Up`, `Down` (có/không dùng và hướng thay đổi liều) |
| Tổng hợp điều trị | `change` (`Ch`/`No`), `diabetesMed` (`Yes`/`No`) | Có thay đổi thuốc / có dùng thuốc đái tháo đường |
| Nhãn | `readmitted` | `NO` 54.864 (53,9%), `>30` 35.545 (34,9%), `<30` 11.357 (11,2%) |

**Đặc điểm cần xử lý.**
1. *Khuyết:* `weight`, `max_glu_serum`, `A1Cresult` khuyết trên 80% (bị loại), `medical_specialty` khuyết 49,08%, `race` 2,23%, `diag_1/2/3` dưới 1,5%.
2. *Thuộc tính hằng số:* `examide` và `citoglipton` chỉ có một giá trị (`No`), không mang thông tin.
3. *Biến phân loại bậc cao:* mã ICD-9 và ba cột `*_id` cần gom nhóm hoặc mã hóa phù hợp.
4. *Mất cân bằng lớp:* lớp `<30` chỉ khoảng 11%.
5. *Nhiều lượt nhập viện cùng bệnh nhân:* cần chia tập theo bệnh nhân để tránh rò rỉ dữ liệu, như đã làm ở 1.1.6.

**Tóm tắt luồng dữ liệu.** 101.766 × 50 → bỏ 3 cột định danh → 101.766 × 47 → bỏ 3 cột khuyết > 50% → 101.766 × 44 → xóa dòng khuyết < 2% và tách nhãn → 100.244 × 45 → chia train 80.195 / test 20.049.
