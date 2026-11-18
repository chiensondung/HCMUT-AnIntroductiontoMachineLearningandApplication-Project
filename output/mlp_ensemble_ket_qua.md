## Kết quả MLP (120-60-20, ReLU) và Aggregate Ensemble

> Chạy trên đầu ra của pipeline 01→06 (train 79.726, test 20.518). Ensemble hiện chỉ gồm MLP và Logistic Regression vì chưa có `proba_*` của Decision Tree và Random Forest (Task 1); chạy lại 09b khi có, kết quả ensemble bên dưới sẽ đổi.

### MLP: chọn siêu tham số trên validation, refit trên toàn bộ train

Quét $\alpha \in \{10^{-5}, \dots, 1\}$ và `learning_rate_init` $\in \{10^{-3}, 10^{-2}\}$ (12 cấu hình mỗi nhãn); chọn theo macro-F1 (`y_multi`) và AUC-ROC (`y_bin`) trên validation. Số liệu: `output/mlp_results.csv`, `output/mlp_sweep.csv`.

| Nhãn | $\alpha$ chọn | `learning_rate_init` | Độ đo chọn | Train | Validation | Test Accuracy | Test (macro-F1 / AUC) |
|---|---|---|---|---|---|---|---|
| `y_multi` | 0,100 | 0,001 | macro-F1 | 0,3872 | 0,3752 | 0,5655 | 0,3646 |
| `y_bin` | 0,100 | 0,001 | AUC-ROC | 0,6977 | 0,6709 | 0,6239 | 0,6696 |

Mốc "luôn đoán `NO`" trên `y_multi` có accuracy 0,5323, nên MLP chỉ hơn mốc 3,3 điểm phần trăm.

![Validation theo alpha](mlp_alpha_curve.png)

*Hình: train (nét đứt) và validation theo $\alpha$, trục x log. Với `y_bin`, train luôn cao hơn validation: chênh 0,0321 ở $\alpha = 10^{-5}$ (lr = 0,001) và 0,0268 ở $\alpha$ chọn (0,100). Chính quy hóa $L_2$ chỉ giảm nhẹ khoảng cách này; $\alpha = 1$ thì cả hai đường cùng thấp (underfit). File: `output/mlp_alpha_curve.png`, `output/mlp_loss.png`.*

### Aggregate Ensemble (nhị phân)

Trọng số tính trên validation (`output/ensemble_weights.csv`):

| Mô hình | AUC validation | $w$ theo AUC | $w$ đều | $w$ theo AUC − 0,5 |
|---|---|---|---|---|
| MLP | 0,6709 | 0,5151 | 0,5000 | 0,5650 |
| LR | 0,6316 | 0,4849 | 0,5000 | 0,4350 |

Kết quả trên test (`output/ensemble_results.csv`, ngưỡng 0,5):

| Mô hình | AUC | Accuracy | Recall | F1 |
|---|---|---|---|---|
| MLP | 0,6696 | 0,6239 | 0,5129 | 0,5606 |
| LR | 0,6348 | 0,6000 | 0,4643 | 0,5205 |
| **Ensemble, $w \propto$ AUC** | 0,6679 | 0,6197 | 0,4865 | 0,5447 |
| Ensemble, trung bình đều | 0,6675 | 0,6197 | 0,4859 | 0,5445 |
| Ensemble, $w \propto$ AUC − 0,5 | 0,6690 | 0,6202 | 0,4899 | 0,5468 |

**Nhận xét.** Với hai thành viên hiện có, Ensemble ($w \propto$ AUC) đạt AUC 0,6679, thấp hơn mô hình đơn tốt nhất là MLP (0,6696), vì Logistic Regression yếu hơn rõ (0,6348) kéo trung bình xuống. Ba cách đặt trọng số cho AUC test chênh nhau không quá 0,0015. Chưa thể kết luận ensemble có lợi hay không cho đến khi có Decision Tree và Random Forest.
