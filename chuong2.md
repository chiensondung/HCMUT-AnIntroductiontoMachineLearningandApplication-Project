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

### 2.2.2. Cây quyết định (Decision Tree)

Khác với k-NN và Logistic Regression, cây quyết định không cần chuẩn hóa hay One-Hot: một nút chia dữ liệu theo một ngưỡng trên một đặc trưng, $x_j \le t$ hay $x_j > t$, nên hoạt động tốt trên cả biến số và biến phân loại dạng mã nguyên (không cần giả định khoảng cách). Việc chọn chỗ chia tại mỗi nút dựa trên độ **không thuần khiết (impurity)** của tập mẫu $S$ rơi vào nút, với $p_c$ là tỉ lệ lớp $c$ trong $S$:

$$\text{Gini}(S) = 1 - \sum_{c=1}^{C} p_c^2, \qquad \text{Entropy}(S) = -\sum_{c=1}^{C} p_c \log p_c.$$

Tại một nút, thuật toán duyệt qua các đặc trưng và ngưỡng khả dĩ, chọn cách chia $(j, t)$ tách $S$ thành $S_L, S_R$ sao cho **độ lợi (information gain)** lớn nhất:

$$\text{Gain}(j, t) = \text{Impurity}(S) - \frac{|S_L|}{|S|}\text{Impurity}(S_L) - \frac{|S_R|}{|S|}\text{Impurity}(S_R).$$

Quá trình lặp lại đệ quy đến khi đạt điều kiện dừng (độ sâu tối đa `max_depth`, số mẫu tối thiểu ở một nút `min_samples_leaf`, hoặc nút đã thuần khiết). Dự đoán tại nút lá là lớp chiếm số đông (hoặc phân bố tỉ lệ $p_c$ để lấy xác suất).

Ưu điểm là dễ diễn giải (có thể in ra luồng quyết định, biết đặc trưng nào được dùng để chia ở đầu cây — thường là đặc trưng quan trọng) và không cần giảm chiều hay chuẩn hóa. Nhược điểm là cây sâu dễ **quá khớp** (ghi nhớ từng mẫu huấn luyện, Gain luôn có thể tăng thêm bằng cách chia nhỏ hơn), phương sai cao (đổi một vài dòng dữ liệu có thể đổi cả cấu trúc cây), và có xu hướng thiên lệch về đặc trưng có nhiều giá trị khác nhau để chia (như mã ICD-9 hoặc One-Hot thưa 2.384 chiều, nơi rất nhiều cột nhị phân hiếm có thể tạo ra Gain giả do số mẫu nhỏ). Vì vậy trên dữ liệu này cần giới hạn `max_depth` và `min_samples_leaf`, và nên so sánh hiệu năng trên đặc trưng gốc (chưa One-Hot/PCA) với đặc trưng đã qua PCA.

## 2.3. Mô hình tổ hợp

### 2.3.1. Rừng ngẫu nhiên (Random Forest) và mô hình kết hợp (Ensemble)

**Rừng ngẫu nhiên.** Một cây quyết định đơn có phương sai cao; Random Forest giảm phương sai bằng cách lấy trung bình nhiều cây được huấn luyện **độc lập và đa dạng hóa** theo hai cơ chế:

1. **Bootstrap (bagging):** mỗi cây $T_b$ được huấn luyện trên một mẫu lấy lại có hoàn lại (bootstrap) kích thước $n$ từ tập huấn luyện gốc.
2. **Chọn ngẫu nhiên đặc trưng:** tại mỗi nút, chỉ xét một tập con ngẫu nhiên gồm $m \ll d$ đặc trưng (thường $m \approx \sqrt{d}$ cho phân loại) để tìm chỗ chia tốt nhất, thay vì toàn bộ $d$ đặc trưng.

Dự đoán cho mẫu mới là trung bình/biểu quyết của $B$ cây:

$$\hat{P}(y=c\mid x) = \frac{1}{B}\sum_{b=1}^{B} \mathbb{1}[T_b(x) = c].$$

Nếu các cây có phương sai $\sigma^2$ và tương quan đôi một trung bình $\rho$, phương sai của trung bình là $\rho\sigma^2 + \frac{1-\rho}{B}\sigma^2$: tăng $B$ chỉ giảm được phần $\frac{1-\rho}{B}\sigma^2$, nên chọn ngẫu nhiên đặc trưng (giảm $\rho$ giữa các cây) quan trọng hơn là chỉ tăng số cây. Nhờ bagging, mỗi cây có khoảng $1-1/e\approx 63\%$ mẫu huấn luyện ở trong túi (in-bag), $37\%$ còn lại (out-of-bag, OOB) có thể dùng để ước lượng lỗi mà không cần tập validation riêng. Random Forest thường ít nhạy với `max_depth` lớn hơn cây đơn (nhờ lấy trung bình), nhưng vẫn cần giới hạn số cây và độ sâu để kiểm soát thời gian huấn luyện trên 2.384 chiều One-Hot. Tầm quan trọng đặc trưng (feature importance, trung bình độ giảm Gini mỗi lần đặc trưng được dùng để chia, trên toàn bộ $B$ cây) hữu ích để xác định các cột gốc (trước One-Hot/PCA) liên quan nhiều nhất đến tái nhập viện.

**Mô hình kết hợp theo trọng số AUC (Ensemble).** Thay vì chỉ dùng một mô hình, kết hợp dự đoán xác suất của $M$ mô hình đã huấn luyện riêng (k-NN, Logistic Regression, Decision Tree, Random Forest, MLP), mỗi mô hình $m$ có trọng số $w_m$ tỉ lệ với AUC của nó trên tập validation:

$$w_m = \frac{\text{AUC}_m}{\sum_{k=1}^{M}\text{AUC}_k}, \qquad \hat{P}(y=c\mid x) = \sum_{m=1}^{M} w_m\, P_m(y=c\mid x).$$

Cách này (theo Wang, CS229, và tương tự Hasan và cộng sự, 2020) cho mô hình tốt hơn đóng góp nhiều hơn mà không cần huấn luyện lại, và thường ổn định hơn một mô hình đơn vì sai số của các mô hình khác nhau (tuyến tính, dựa khoảng cách, dựa cây, mạng nơ-ron) có xu hướng không tương quan hoàn toàn. Nhược điểm là cần lưu và chạy dự đoán cho cả $M$ mô hình (chi phí suy luận cao hơn), và nếu AUC trên validation không ổn định (ví dụ do mất cân bằng lớp `<30` ≈ 11%), trọng số $w_m$ có thể không phản ánh đúng chất lượng thật trên tập test.

## 2.4. Hạn chế của PCA trên ma trận tương quan One-Hot

PCA như cài đặt ở 2.1.2 chuẩn hóa mỗi cột về phương sai 1 trước khi tính trị riêng (ma trận tương quan, không phải hiệp phương sai thô). Với dữ liệu One-Hot, điều này có tác dụng phụ: một cột nhị phân hiếm (ví dụ một mã ICD-9 chỉ xuất hiện ở vài chục dòng trong 80.195 dòng train) có phương sai rất nhỏ ($p(1-p)$ với $p$ nhỏ), nên sau khi chia cho $\sigma$, dao động của nó bị **khuếch đại lên cùng mức với các cột phổ biến** (như `diabetesMed_Yes`, xuất hiện ở phần lớn dòng). Kết quả thực nghiệm trên `data/train.csv` xác nhận hệ quả này: sau khi One-Hot (2.384 cột còn lại, đã bỏ 2 cột hằng số `examide`, `citoglipton`), 60 thành phần đầu chỉ giải thích **khoảng 5,5%** tổng phương sai, và cần tới **hơn 2.100 thành phần** (gần như toàn bộ 2.384 chiều) để đạt 95% — khác hẳn với mức giảm chiều "hiệu quả" (2.500 → 60) mà Wang (CS229) báo cáo. Nguyên nhân là phương sai bị trải rất đều trên rất nhiều mã ICD-9/thuốc hiếm sau khi chuẩn hóa, nên không còn vài hướng chiếm phần lớn phương sai như với dữ liệu số liên tục thông thường. Đây là một hạn chế cần nêu rõ khi so sánh với hướng không giảm chiều (Logistic Regression, Decision Tree/Random Forest trực tiếp trên One-Hot hoặc trên đặc trưng gốc).
