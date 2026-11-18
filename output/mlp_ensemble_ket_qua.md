## Kết quả MLP (120-60-20, ReLU) và Aggregate Ensemble

> Chạy trên đầu ra của pipeline 01→06 (train 79.726, test 20.518). Ensemble gồm 4 thành viên: Decision Tree và Random Forest (Task 1, Sơn), MLP (09a) và Logistic Regression (09b).

### MLP: chọn siêu tham số trên validation, refit trên toàn bộ train

Quét $\alpha \in \{10^{-5}, \dots, 1\}$ và `learning_rate_init` $\in \{10^{-3}, 10^{-2}\}$ (12 cấu hình mỗi nhãn); chọn theo macro-F1 (`y_multi`) và AUC-ROC (`y_bin`) trên validation. Số liệu: `output/mlp_results.csv`, `output/mlp_sweep.csv`.

| Nhãn | $\alpha$ chọn | `learning_rate_init` | Độ đo chọn | Train | Validation | Test Accuracy | Test (macro-F1 / AUC) |
|---|---|---|---|---|---|---|---|
| `y_multi` | 0,0001 | 0,010 | macro-F1 | 0,3892 | 0,3753 | 0,5647 | 0,3591 |
| `y_bin` | 0,1 | 0,001 | AUC-ROC | 0,6989 | 0,6706 | 0,6212 | 0,6683 |

Mốc "luôn đoán `NO`" trên `y_multi` có accuracy 0,5323, nên MLP chỉ hơn mốc 3,2 điểm phần trăm.

![Validation theo alpha](mlp_alpha_curve.png)

*Hình: train (nét đứt) và validation theo $\alpha$, trục x log. Với `y_bin`, train luôn cao hơn validation: chênh 0,0339 ở $\alpha = 10^{-5}$ (lr = 0,001) và 0,0283 ở $\alpha$ chọn (0,1). Chính quy hóa $L_2$ chỉ giảm nhẹ khoảng cách này; $\alpha = 1$ thì cả hai đường cùng thấp (underfit). File: `output/mlp_alpha_curve.png`, `output/mlp_loss.png`.*

### Aggregate Ensemble (nhị phân)

Trọng số tính trên validation (`output/ensemble_weights.csv`):

| Mô hình | AUC validation | $w$ theo AUC | $w$ đều | $w$ theo AUC − 0,5 |
|---|---|---|---|---|
| DT | 0,6162 | 0,2395 | 0,2500 | 0,2028 |
| RF | 0,6542 | 0,2543 | 0,2500 | 0,2693 |
| MLP | 0,6706 | 0,2607 | 0,2500 | 0,2979 |
| LR | 0,6317 | 0,2456 | 0,2500 | 0,2300 |

Kết quả trên test (`output/ensemble_results.csv`, ngưỡng 0,5):

| Mô hình | AUC | Accuracy | Recall | F1 |
|---|---|---|---|---|
| DT | 0,6140 | 0,5876 | 0,5119 | 0,5372 |
| RF | 0,6537 | 0,6085 | 0,4848 | 0,5367 |
| MLP | 0,6683 | 0,6212 | 0,5950 | 0,5950 |
| LR | 0,6344 | 0,5992 | 0,4631 | 0,5194 |
| **Ensemble, $w \propto$ AUC** | 0,6644 | 0,6192 | 0,5171 | 0,5595 |
| Ensemble, trung bình đều | 0,6639 | 0,6188 | 0,5152 | 0,5584 |
| Ensemble, $w \propto$ AUC − 0,5 | 0,6662 | 0,6198 | 0,5212 | 0,5618 |

**Nhận xét.** Ensemble ($w \propto$ AUC) đạt AUC 0,6644, thấp hơn mô hình đơn tốt nhất là MLP (0,6683) 0,0039 điểm AUC, nhưng cao hơn mô hình yếu nhất (0,6140 của DT) và Accuracy (0,6192) cao hơn mọi mô hình đơn trừ MLP. Vì AUC validation của 4 mô hình nằm trong khoảng hẹp (0,6162 đến 0,6706), trọng số theo AUC xấp xỉ 0,25 mỗi mô hình, nên ensemble gần với trung bình đơn giản; ba cách đặt trọng số cho AUC test chênh nhau không quá 0,0023. Ensemble kéo MLP (mô hình mạnh nhất) về phía DT và LR yếu hơn, nên trong thực nghiệm này ensemble theo AUC không vượt mô hình đơn tốt nhất.
