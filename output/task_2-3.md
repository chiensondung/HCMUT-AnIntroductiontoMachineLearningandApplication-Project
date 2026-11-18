# Mục 2.3.2: Mạng nơ-ron đa lớp (Multi-Layer Perceptron / DNN)

## 2.3.2.1. Lý do lựa chọn

Ngoài các mô hình cây/rừng (Logistic Regression, Random Forest, Gradient Boosting) đã nêu ở Mục 1.1.3, báo cáo bổ sung một mô hình mạng nơ-ron đa lớp (MLP) để:

- Khai thác khả năng học các tổ hợp phi tuyến giữa các đặc trưng (ví dụ tương tác giữa nhóm thuốc, số lần nhập viện trước đó, thời gian nằm viện) mà mô hình tuyến tính khó nắm bắt.
- Làm cơ sở so sánh với các mô hình cây/rừng, qua đó kiểm chứng nhận định của Wang (CS229, dẫn ở Mục 1.2): trên bộ dữ liệu này mạng nơ-ron chỉ nhỉnh hơn các mô hình đơn giản không đáng kể khi không xử lý tốt mất cân bằng lớp — báo cáo thử nghiệm lại với dữ liệu đã được chuẩn hóa (Mục 2.3.1) và xử lý mất cân bằng lớp (Mục 2.4) để xem mức cải thiện thực tế.

## 2.3.2.2. Dữ liệu đầu vào

Đầu vào mạng là vector đặc trưng đã qua chuẩn hóa `StandardScaler` (Mục 2.3.1): các biến số (`time_in_hospital`, `num_lab_procedures`, …), biến thứ bậc `age` (mã hóa 0–9) và các biến phân loại đã one-hot (23 thuốc, `race`, `gender`, `admission_type_id`, `discharge_disposition_id`, `admission_source_id`, `change`, `diabetesMed`) — tổng cộng **142 chiều** (chưa gồm `medical_specialty`, `diag_1/2/3` do phân loại bậc cao, sẽ gom nhóm ở bước đặc trưng riêng). Có thể dùng trực tiếp 142 chiều này, hoặc dùng số thành phần chính từ PCA (≈78 PC giải thích ≥80% phương sai, Mục 2.3.1) làm đầu vào rút gọn để giảm số tham số và nguy cơ quá khớp.

## 2.3.2.3. Kiến trúc đề xuất

Kiến trúc MLP dạng *feed-forward*, truyền thẳng qua các lớp kết nối đầy đủ (fully-connected):

```
Input (d chiều: 142 hoặc ~78 nếu dùng PCA)
   │
Dense(128) → BatchNorm → ReLU → Dropout(0.3)
   │
Dense(64)  → BatchNorm → ReLU → Dropout(0.3)
   │
Dense(32)  → ReLU
   │
Output: Dense(1, sigmoid)        [bài toán nhị phân y_bin]
     hoặc Dense(3, softmax)       [bài toán đa lớp y_multi]
```

- **Số lớp ẩn:** 3 lớp (128 → 64 → 32), giảm dần số nơ-ron theo dạng "phễu" — kiến trúc phổ biến cho dữ liệu dạng bảng cỡ vài trăm nghìn mẫu, tránh quá nhiều tham số so với ~80.000 mẫu huấn luyện.
- **Hàm kích hoạt ẩn:** ReLU `f(z) = max(0, z)` — tránh bão hòa gradient (vanishing gradient) so với sigmoid/tanh ở các lớp sâu.
- **BatchNorm:** chuẩn hóa lại đầu ra mỗi lớp ẩn, giúp huấn luyện ổn định và nhanh hội tụ hơn, giảm phụ thuộc vào cách khởi tạo trọng số.
- **Dropout (p = 0.3):** ngắt ngẫu nhiên 30% nơ-ron mỗi lượt huấn luyện ở hai lớp đầu — một hình thức regularization bổ sung cho L2 (Mục 2.4.2), đặc biệt cần thiết vì dữ liệu có nhiều đặc trưng one-hot thưa và dễ quá khớp.
- **Lớp đầu ra:**
  - Bài toán nhị phân (`y_bin`): 1 nơ-ron, kích hoạt **sigmoid**, đầu ra là xác suất tái nhập viện.
  - Bài toán đa lớp (`y_multi`: `<30`/`>30`/`NO`): 3 nơ-ron, kích hoạt **softmax**, đầu ra là phân phối xác suất trên 3 lớp.

## 2.3.2.4. Công thức lan truyền thuận (forward pass)

Với lớp ẩn thứ $l$, đầu vào $h^{(l-1)}$, trọng số $W^{(l)}$, bias $b^{(l)}$:

$$z^{(l)} = W^{(l)} h^{(l-1)} + b^{(l)}, \qquad h^{(l)} = \text{ReLU}\big(\text{BatchNorm}(z^{(l)})\big)$$

Lớp đầu ra (nhị phân): $\hat{y} = \sigma(z^{(L)}) = \dfrac{1}{1 + e^{-z^{(L)}}}$

Lớp đầu ra (đa lớp): $\hat{y}_k = \dfrac{e^{z^{(L)}_k}}{\sum_{j=1}^{3} e^{z^{(L)}_j}}$

---

# Mục 2.4: Hàm mất mát Cross-Entropy, Regularization (L2) & Early Stopping

## 2.4.1. Hàm mất mát Cross-Entropy

**Nhị phân (Binary Cross-Entropy)** — dùng cho đầu ra sigmoid của `y_bin`:

$$\mathcal{L}_{BCE} = -\frac{1}{N}\sum_{i=1}^{N}\Big[y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)\Big]$$

**Đa lớp (Categorical Cross-Entropy)** — dùng cho đầu ra softmax của `y_multi` (one-hot $y_{i,k}$, 3 lớp):

$$\mathcal{L}_{CCE} = -\frac{1}{N}\sum_{i=1}^{N}\sum_{k=1}^{3} y_{i,k}\log(\hat{y}_{i,k})$$

**Trọng số theo lớp (class weight):** vì `<30` chỉ chiếm ≈11,2% (Mục 1.1.6), hàm mất mát được nhân thêm trọng số nghịch với tần suất lớp:

$$w_k = \frac{N}{C \cdot N_k}, \qquad \mathcal{L}_{weighted} = -\frac{1}{N}\sum_{i} w_{y_i}\Big[\ldots\Big]$$

trong đó $N_k$ là số mẫu lớp $k$, $C$ là số lớp — giúp mô hình không chỉ thiên về dự đoán lớp đa số (`NO`) như hạn chế đã nêu trong khảo sát của Wang (Mục 1.2).

## 2.4.2. Regularization L2 (Weight Decay)

Để giảm quá khớp, thêm số hạng phạt tỉ lệ với bình phương trọng số vào hàm mất mát:

$$\mathcal{L}_{total} = \mathcal{L}_{CE} + \lambda \sum_{l} \|W^{(l)}\|_2^2$$

- $\lambda$ (hệ số regularization) là siêu tham số, chọn qua tập validation (ví dụ thử $\lambda \in \{10^{-5}, 10^{-4}, 10^{-3}\}$).
- L2 kéo các trọng số về gần 0 nhưng không triệt tiêu hoàn toàn (khác L1), phù hợp khi đầu vào có nhiều đặc trưng one-hot tương quan (ví dụ các mức `No/Steady/Up/Down` của cùng một loại thuốc).
- Kết hợp với Dropout (Mục 2.3.2.3) tạo hai cơ chế regularization độc lập, giảm rủi ro quá khớp trên tập huấn luyện ~80.000 mẫu, 142 chiều.

## 2.4.3. Early Stopping

Huấn luyện được theo dõi trên tập validation (tách từ train, Mục 1.1.6) sau mỗi epoch:

- **Độ đo theo dõi:** do dữ liệu mất cân bằng, không dùng validation loss hay accuracy làm tiêu chí dừng duy nhất, mà theo dõi **F1-score** hoặc **PR-AUC** của lớp `<30` (hoặc `y_bin=1`) trên tập validation — nhất quán với các độ đo đánh giá đã chọn ở Mục 1.1.3.
- **Patience:** dừng huấn luyện nếu độ đo theo dõi không cải thiện sau **10 epoch liên tiếp**.
- **Khôi phục trọng số tốt nhất:** lưu lại trọng số của epoch có độ đo validation tốt nhất, khôi phục trọng số này khi dừng (tránh dùng trọng số của epoch cuối cùng, vốn có thể đã quá khớp).

Early stopping đóng vai trò regularization bổ sung theo thời gian huấn luyện (ngoài L2 theo không gian trọng số và Dropout theo cấu trúc mạng), đồng thời giúp xác định số epoch hợp lý mà không cần dò thủ công.
