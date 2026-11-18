# Dự đoán tái nhập viện ở bệnh nhân đái tháo đường

Bài tập lớn môn **Máy học Cơ bản và Ứng dụng** (HCMUT), **Nhóm 10, lớp L01**.

Bài toán: dự đoán bệnh nhân có tái nhập viện hay không trên bộ dữ liệu UCI *Diabetes 130-US hospitals (1999–2008)* (101.766 lượt nhập viện), mở rộng từ bài báo tham khảo `Predicting diabetes health outcomes using a variety of models.pdf`. Có hai nhãn:

- `y_multi` ∈ {`<30`, `>30`, `NO`}: tái nhập viện trong 30 ngày, sau 30 ngày, hoặc không.
- `y_bin` = 1 nếu có tái nhập viện (`<30` hoặc `>30`), ngược lại 0.

## Cấu trúc repo

| Nhánh / đường dẫn | Nội dung |
|---|---|
| `main` | Dữ liệu gốc `data/`, `requirements.txt` và notebook tổng hợp `run_local/` |
| `Dung`, `Son`, `Chien` | Script (`src/`), báo cáo từng phần (`output/*.md`) của mỗi thành viên |

Mỗi thành viên làm việc trên nhánh riêng. Notebook tổng hợp **không chứa code của mô hình**: nó clone 4 nhánh từ GitHub rồi chạy script của từng người.

## Pipeline

Mỗi bước là một script độc lập; bước sau đọc file do bước trước ghi vào `output/`.

| Bước | Công việc | Người (nhánh) |
|---|---|---|
| 01 | Bỏ thuộc tính định danh | Dũng |
| 02 | Thống kê khuyết, bỏ cột khuyết > 50% | Sơn |
| 03 | Bỏ dòng khuyết, tạo nhãn, chia train/test 80/20 | Chiến |
| 04 | One-Hot (fit trên train) | Dũng |
| 05 | PCA 60 thành phần | Sơn |
| 06 | Chuẩn hóa sau PCA | Chiến |
| 07a, 07b | k-NN và Softmax Regression tự cài bằng numpy | Dũng |
| 08a, 08b | Decision Tree, Random Forest | Sơn |
| 09a, 09b | MLP 120-60-20, Ensemble trọng số AUC | Chiến |
| 10 | Accuracy đa lớp, ma trận nhầm lẫn | Dũng |
| 11 | Đánh giá nhị phân, ROC/AUC | Sơn |
| 12 | Mất cân bằng lớp, RF trên One-Hot chưa PCA | Chiến |

## Chạy lại

Cần Python 3, `git` và kết nối Internet.

1. Mở `run_local/261 - MHCBVUD - L01 - Nhóm 10.ipynb` bằng Jupyter hoặc VS Code.
2. Chọn **Run All**. Cell đầu tiên tạo `run_local/branches/` (clone 4 nhánh), cài thư viện đã ghim, và các bước sau ghi kết quả vào `run_local/branches/main/output/`.

Việc chạy toàn bộ mất từ vài chục phút trở lên; bước 08b (Random Forest) và 09a (MLP) là hai bước lâu nhất.

## Quy ước

- `random_state = 42` ở mọi nơi.
- Mọi phép học từ dữ liệu (One-Hot, chuẩn hóa, PCA) chỉ fit trên tập train.
- Siêu tham số chọn trên validation tách từ train; test chỉ dùng để báo cáo.
- Độ đo chọn mô hình: macro-F1 (đa lớp), AUC (nhị phân).
- **Phiên bản thư viện được ghim** (`scikit-learn==1.6.1`, `pandas==2.2.3`, `numpy==2.4.6`): cách chia train/test ở bước 03 phụ thuộc phiên bản scikit-learn, khác phiên bản sẽ cho tập train/test khác (79.726 / 20.518 dòng với 1.6.1) và số liệu báo cáo không còn khớp.
