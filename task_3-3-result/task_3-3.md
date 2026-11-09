# Tóm tắt Task 3.3: MLP (120-60-20, ReLU) & Aggregate Ensemble theo AUC-ROC

## 1. Mục tiêu

Thiết kế, huấn luyện mạng nơ-ron đa lớp (MLP/DNN) theo kiến trúc đã đề xuất ở Mục 2.3.2 (hidden layers 120-60-20, ReLU), và lập trình một **Aggregate Ensemble Model** kết hợp MLP với các mô hình nền khác, trong đó trọng số đóng góp của mỗi mô hình tỉ lệ thuận với AUC-ROC của mô hình đó trên tập validation.

Script: `src/task-3_3_chien.py`.

## 2. Đầu vào

Script đọc **một file duy nhất** qua biến `input_dir` (mặc định `data/train.csv`), được giả định là sản phẩm đầu ra của các bước 1.1–1.3 (lọc/làm sạch, tạo nhãn `y_bin`) và 2.1–2.3 (mã hóa biến phân loại, chuẩn hóa/PCA) — tức mọi cột đặc trưng trong file này phải là số. Cần cập nhật `input_dir`, `target_col`, `cot_loai_bo` cho khớp với file đầu ra thật sự của bước 2.3 trước khi chạy chính thức.

Từ file này, script tự chia **train/validation/test theo tỉ lệ 60/20/20**, phân tầng theo nhãn `y_bin`, `random_state=42`. `StandardScaler` được fit trên tập train rồi áp dụng cho validation/test.

## 3. Các mô hình nền (base models)

| Mô hình | Cấu hình chính |
| --- | --- |
| Logistic Regression | `class_weight="balanced"`, `max_iter=1000` |
| Random Forest | 300 cây, `class_weight="balanced"` |
| Gradient Boosting | 200 cây, `max_depth=3`, `learning_rate=0.1`, trọng số mẫu nghịch tần suất lớp (vì không hỗ trợ `class_weight`) |
| **MLP (120-60-20)** | `hidden_layer_sizes=(120, 60, 20)`, `activation="relu"`, `alpha=1e-4` (L2, Mục 2.4.2), `early_stopping=True`, `n_iter_no_change=10` (Early Stopping, Mục 2.4.3) |

## 4. Aggregate Ensemble theo trọng số AUC-ROC

1. Mỗi mô hình huấn luyện trên tập train, dự đoán xác suất trên tập validation.
2. Tính $AUC_i$ = ROC-AUC của mô hình $i$ trên validation.
3. Trọng số: $w_i = \dfrac{AUC_i}{\sum_j AUC_j}$ (chuẩn hóa để tổng = 1).
4. Xác suất ensemble trên test: $\hat{p}_{ensemble} = \sum_i w_i \cdot \hat{p}_i$.
5. Đánh giá ensemble và từng mô hình trên tập test (không dùng để chọn trọng số) bằng AUC-ROC, Accuracy, Recall, F1 (ngưỡng 0.5).

## 5. Kết quả chạy thử nghiệm (pilot)

> Lưu ý: kết quả dưới đây chạy từ phiên bản trước của script (đọc trực tiếp `data/train.csv` + `data/test.csv`, tự mã hóa one-hot), **chỉ mang tính minh họa quy trình**. Khi trỏ `input_dir` vào dữ liệu thật đã qua bước 2.1–2.3 (mã hóa/chuẩn hóa/PCA chính thức), các số liệu cần chạy lại.

| Mô hình | Trọng số ensemble | AUC-ROC | Accuracy | Recall | F1 |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.2497 | 0.6902 | 0.6391 | 0.5856 | 0.6005 |
| Random Forest | 0.2474 | 0.6793 | 0.6287 | 0.5924 | 0.5964 |
| Gradient Boosting | 0.2531 | 0.6995 | 0.6442 | 0.6084 | 0.6130 |
| MLP (120-60-20) | 0.2498 | 0.6914 | 0.6404 | 0.5798 | 0.5989 |
| **Aggregate Ensemble** | — | **0.7005** | **0.6449** | 0.5907 | 0.6064 |

Nhận xét: 4 mô hình có AUC-ROC khá đồng đều (0.679–0.700) nên trọng số ensemble gần bằng nhau (~0.25 mỗi mô hình). Aggregate Ensemble đạt AUC-ROC và Accuracy cao nhất trong số các mô hình, cho thấy việc kết hợp có cải thiện nhẹ so với từng mô hình đơn lẻ.

File liên quan trong thư mục này: `ket_qua_test.csv` (bảng số liệu), `roc_curve_test.png` (đường ROC từng mô hình và ensemble), `task_3-3_tomtat.md` (tóm tắt thô do script tự xuất ra ở lần chạy pilot).
