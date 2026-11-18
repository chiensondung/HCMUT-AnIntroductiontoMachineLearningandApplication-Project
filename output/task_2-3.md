# Mục 2.3.2: Mạng nơ-ron đa lớp (Multi-Layer Perceptron / DNN)

## 2.3.2.1. Lý do lựa chọn

Ngoài các mô hình Logistic Regression, Random Forest, Gradient Boosting, báo cáo bổ sung một mô hình mạng nơ-ron đa lớp (MLP) để:

- Khai thác khả năng học các tổ hợp phi tuyến giữa các đặc trưng (ví dụ tương tác giữa nhóm thuốc, số lần nhập viện trước đó, thời gian nằm viện) mà mô hình tuyến tính khó nắm bắt.
- Làm một thành viên của Aggregate Ensemble theo trọng số AUC-ROC (Mục 2.2.3). Kết quả thực nghiệm (`output/mlp_ensemble_ket_qua.md`) cho thấy MLP chỉ cải thiện nhẹ so với các mô hình đơn giản, phù hợp nhận định của Wang (CS229, Mục 1.2).

## 2.3.2.2. Dữ liệu đầu vào

Đầu vào mạng là **60 thành phần chính** (PCA, bước 05) đã chuẩn hóa `StandardScaler` fit trên train (bước 06): `output/X_train_pca_scaled.npy`, `output/X_test_pca_scaled.npy`. Dùng PCA giúp giảm từ 2.396 chiều One-Hot xuống 60 chiều, giảm số tham số và thời gian huấn luyện, nhưng 60 thành phần chỉ giữ khoảng 5,6% tổng phương sai (`output/pca_summary.csv`) nên mất một phần thông tin.

## 2.3.2.3. Kiến trúc

Mạng truyền thẳng (feed-forward) kết nối đầy đủ, cài bằng `sklearn.neural_network.MLPClassifier`:

```
Input (60 chiều: thành phần chính đã chuẩn hóa)
   │
Dense(120) → ReLU
   │
Dense(60)  → ReLU
   │
Dense(20)  → ReLU
   │
Output: softmax 3 lớp            [bài toán đa lớp y_multi]
     hoặc 2 lớp / sigmoid         [bài toán nhị phân y_bin]
```

- **Số lớp ẩn:** 3 lớp (120 → 60 → 20), giảm dần theo dạng "phễu", theo kiến trúc của bài báo tham khảo.
- **Hàm kích hoạt ẩn:** ReLU `f(z) = max(0, z)`, tránh bão hòa gradient so với sigmoid/tanh ở các lớp sâu.
- **Tối ưu:** Adam, `max_iter=300`. Không dùng BatchNorm và Dropout (không có sẵn trong `MLPClassifier`); chống quá khớp bằng L2 (Mục 2.4.2) và early stopping (Mục 2.4.3).
- **Lớp đầu ra:** softmax với `y_multi` (3 lớp, `<30`/`>30`/`NO`); với `y_bin` dùng xác suất lớp 1 làm đầu ra.

## 2.3.2.4. Công thức lan truyền thuận (forward pass)

Với lớp ẩn thứ $l$, đầu vào $h^{(l-1)}$, trọng số $W^{(l)}$, bias $b^{(l)}$:

$$z^{(l)} = W^{(l)} h^{(l-1)} + b^{(l)}, \qquad h^{(l)} = \text{ReLU}\big(z^{(l)}\big)$$

Lớp đầu ra (nhị phân): $\hat{y} = \sigma(z^{(L)}) = \dfrac{1}{1 + e^{-z^{(L)}}}$

Lớp đầu ra (đa lớp): $\hat{y}_k = \dfrac{e^{z^{(L)}_k}}{\sum_{j=1}^{3} e^{z^{(L)}_j}}$

---

# Mục 2.4: Hàm mất mát Cross-Entropy, Regularization (L2) & Early Stopping

## 2.4.1. Hàm mất mát Cross-Entropy

**Nhị phân (Binary Cross-Entropy)** — dùng cho đầu ra sigmoid của `y_bin`:

$$\mathcal{L}_{BCE} = -\frac{1}{N}\sum_{i=1}^{N}\Big[y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)\Big]$$

**Đa lớp (Categorical Cross-Entropy)** — dùng cho đầu ra softmax của `y_multi` (one-hot $y_{i,k}$, 3 lớp):

$$\mathcal{L}_{CCE} = -\frac{1}{N}\sum_{i=1}^{N}\sum_{k=1}^{3} y_{i,k}\log(\hat{y}_{i,k})$$

**Trọng số theo lớp:** `<30` chỉ chiếm khoảng 11,2% (Mục 1.3.2) nên mô hình dễ thiên về lớp đa số `NO`. Thí nghiệm MLP ở Mục 2.2.3 **không** dùng trọng số lớp để các mô hình so sánh được với nhau; ảnh hưởng của mất cân bằng lớp được khảo sát riêng ở Task 4.

## 2.4.2. Regularization L2 (Weight Decay)

Để giảm quá khớp, thêm số hạng phạt tỉ lệ với bình phương trọng số vào hàm mất mát (trong `MLPClassifier` hệ số này là `alpha`):

$$\mathcal{L}_{total} = \mathcal{L}_{CE} + \frac{\alpha}{2N} \sum_{l} \|W^{(l)}\|_2^2$$

- `alpha` được quét trên {$10^{-5}$, $10^{-4}$, $10^{-3}$, $10^{-2}$, $10^{-1}$, $1$} cùng `learning_rate_init` $\in$ {$10^{-3}$, $10^{-2}$}, chọn theo tập validation (macro-F1 cho `y_multi`, AUC-ROC cho `y_bin`). Kết quả: `y_multi` chọn $\alpha$ = $10^{-4}$ (lr = $10^{-2}$), `y_bin` chọn $\alpha$ = $10^{-1}$ (lr = $10^{-3}$) (`output/mlp_results.csv`, `output/mlp_sweep.csv`).
- L2 kéo các trọng số về gần 0 nhưng không triệt tiêu hoàn toàn (khác L1), phù hợp khi đầu vào có các thành phần tương quan.

## 2.4.3. Early Stopping

`MLPClassifier(early_stopping=True, validation_fraction=0.1, n_iter_no_change=10)`: giữ lại 10% tập huấn luyện làm validation nội bộ và theo dõi **độ chính xác (accuracy) trên phần này** sau mỗi epoch (đây là độ đo mặc định của scikit-learn, không phải F1 hay PR-AUC).

- **Patience:** dừng huấn luyện nếu điểm validation không cải thiện sau **10 epoch liên tiếp**.
- **Khôi phục trọng số tốt nhất:** khi dừng, scikit-learn khôi phục trọng số của epoch có điểm validation tốt nhất (tránh dùng trọng số của epoch cuối, vốn có thể đã quá khớp).

Đường loss huấn luyện và điểm validation theo epoch của mô hình cuối cùng nằm ở `output/mlp_loss.png`. Early stopping là regularization bổ sung theo thời gian huấn luyện, ngoài L2 theo không gian trọng số, và giúp không phải dò số epoch thủ công. Vì độ đo theo dõi là accuracy, trên `y_multi` mất cân bằng nó có thể ưu tiên lớp đa số; đây là hạn chế cần lưu ý khi đọc kết quả.
