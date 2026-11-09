# Chương 2: Cơ sở lý thuyết và tiền xử lý đặc trưng

## 2.1. Mã hóa One-Hot và cơ sở toán học PCA

### 2.1.1. Mã hóa One-Hot
Các mô hình k-NN và Logistic Regression chỉ làm việc với số. Nếu mã hóa nhãn một biến phân loại thành 1, 2, 3, … thì mô hình hiểu nhầm có thứ tự và khoảng cách giữa các loại (ví dụ mã ICD-9 `250` “gần” `251` hơn `401`). One-Hot Encoding tránh điều này: biến $x$ có $K$ giá trị được thay bằng $K$ cột nhị phân

$$e_k(x) = \mathbb{1}[x = c_k],\quad k = 1,\dots,K,$$

mỗi dòng có đúng một cột bằng 1. Khoảng cách giữa hai loại khác nhau luôn bằng $\sqrt{2}$, nên không loại nào “gần” loại nào.

Áp dụng trên `data/diabetic_data_step1.csv` (mã: `src/04_one_hot_encoding.py`):
- **Biến được mã hóa (38):** 35 thuộc tính chữ (`race`, `gender`, `age`, `medical_specialty`, 3 mã chẩn đoán ICD-9 `diag_1..3`, các thuốc, `change`, `diabetesMed`, …) và 3 mã số thực chất là mã loại: `admission_type_id`, `discharge_disposition_id`, `admission_source_id`. Nhãn `readmitted` được tách ra, không mã hóa.
- **Giá trị khuyết:** coi là một loại riêng `Missing` (khuyết có thể mang thông tin, ví dụ `medical_specialty` không ghi).
- **Biến số giữ nguyên (8):** `time_in_hospital`, `num_lab_procedures`, `num_procedures`, `num_medications`, `number_outpatient`, `number_emergency`, `number_inpatient`, `number_diagnoses`.
- **Kết quả:** 101.766 × 2.500 cột. `diag_1`, `diag_2`, `diag_3` có lần lượt 717, 749, 790 cột (716 / 748 / 789 mã + `Missing`) – chiếm phần lớn số cột.
- **Lưu trữ:** vì ma trận rất thưa (mỗi dòng chỉ có 38 giá trị 1 trên 2.492 cột One-Hot), kết quả được lưu dạng thưa `data/X_onehot.npz`; tên cột ở `data/X_onehot_cols.txt`, nhãn ở `data/y.csv`.

Lưu ý: tập loại được xác định từ toàn bộ dữ liệu và không dùng nhãn nên không rò rỉ nhãn. Nếu chia train/test trước thì nên fit bộ mã hóa trên train và đặt `handle_unknown="ignore"` cho các loại mới ở test. Kích thước 2.500 chiều (nhiều mã ICD-9 hiếm) là lý do cần giảm chiều bằng PCA.

### 2.1.2. Cơ sở toán học của PCA
PCA tìm một phép chiếu tuyến tính sang không gian ít chiều sao cho giữ lại nhiều phương sai nhất. Cho ma trận dữ liệu $X \in \mathbb{R}^{n \times d}$, đã trừ trung bình từng cột ($\bar{X}$).

1. **Ma trận hiệp phương sai:** $\;C = \dfrac{1}{n-1}\bar{X}^{\top}\bar{X} \in \mathbb{R}^{d\times d}$.
2. **Phân rã trị riêng:** $C v_j = \lambda_j v_j$ với $\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_d \ge 0$ và $v_j$ trực chuẩn. Thành phần chính thứ $j$ là hướng $v_j$, phương sai dọc theo nó bằng $\lambda_j$.
3. **Chọn $m$ thành phần:** $\;Z = \bar{X} V_m,\; V_m = [v_1,\dots,v_m]$, và chọn $m$ nhỏ nhất sao cho tỉ lệ phương sai giải thích
$$\frac{\sum_{j=1}^{m}\lambda_j}{\sum_{j=1}^{d}\lambda_j} \ge \tau \quad(\text{ví dụ } \tau = 0{,}95).$$

PCA tương đương cực tiểu sai số tái tạo $\lVert \bar{X} - \bar{X}V_mV_m^{\top}\rVert_F^2$. Trong thực hành tính qua SVD $\bar{X} = U\Sigma V^{\top}$, khi đó $\lambda_j = \sigma_j^2/(n-1)$ và không cần tạo ra $C$.

Với ma trận One-Hot thưa 2.500 chiều, trừ trung bình làm mất tính thưa. Khi đó dùng `TruncatedSVD` (không trừ trung bình) trên ma trận thưa, hoặc `PCA` sau khi chuyển sang ma trận dày với số thành phần giới hạn. PCA chỉ nên fit trên tập huấn luyện, và biến số cần chuẩn hóa trước để không lấn át các cột nhị phân.

## 2.2. Các mô hình phân loại

### 2.2.1. Thuật toán k-NN và Logistic Regression (Softmax)

**k-NN.** Không có bước huấn luyện. Với mẫu mới $x$, tìm $k$ mẫu huấn luyện gần nhất theo khoảng cách (Euclid $d(x,x_i)=\lVert x-x_i\rVert_2$, hoặc Hamming/Manhattan với dữ liệu One-Hot), rồi bỏ phiếu:

$$\hat{y} = \arg\max_c \sum_{i \in N_k(x)} w_i\,\mathbb{1}[y_i = c],\qquad w_i = 1 \text{ hoặc } \frac{1}{d(x,x_i)}.$$

$k$ nhỏ thì nhạy nhiễu (phương sai cao), $k$ lớn thì quá trơn (thiên lệch cao); chọn $k$ bằng cross-validation. Ở 2.500 chiều khoảng cách mất ý nghĩa (“lời nguyền chiều”), nên k-NN nên chạy sau PCA, và các biến số phải được chuẩn hóa. Độ phức tạp dự đoán là $O(nd)$ cho mỗi mẫu.

**Logistic Regression (Softmax).** Với $C$ lớp (`<30`, `>30`, `NO`), mô hình tính điểm tuyến tính $z_c = w_c^{\top}x + b_c$ rồi chuyển thành xác suất:

$$P(y=c \mid x) = \frac{e^{z_c}}{\sum_{j=1}^{C} e^{z_j}}.$$

Với $C=2$ công thức này trở thành hàm sigmoid (Logistic Regression nhị phân). Tham số tối ưu bằng cực tiểu hóa cross-entropy có chuẩn hóa $L_2$:

$$J(W) = -\frac{1}{n}\sum_{i=1}^{n}\sum_{c=1}^{C}\mathbb{1}[y_i=c]\,\log P(y_i=c\mid x_i) + \frac{\lambda}{2}\lVert W\rVert_F^2,$$

với gradient theo $w_c$ là $\;\dfrac{1}{n}\sum_i \big(P(c\mid x_i) - \mathbb{1}[y_i=c]\big)x_i + \lambda w_c$. Hàm $J$ lồi nên gradient descent (hoặc L-BFGS) hội tụ về cực tiểu toàn cục. Vì lớp `<30` hiếm, có thể dùng trọng số lớp (`class_weight="balanced"`) và chọn ngưỡng quyết định theo Recall/F1 thay vì mặc định 0,5. Mô hình tuyến tính xử lý tốt ma trận One-Hot thưa mà không cần giảm chiều, và hệ số cho từng loại ICD-9 có thể giải thích được.
