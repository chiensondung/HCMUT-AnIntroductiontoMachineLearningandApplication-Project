# Kết quả Task 2.3: Chuẩn hóa dữ liệu + PCA

- Số đặc trưng đưa vào PCA (sau mã hóa one-hot/ordinal): **142**
- Số mẫu (train): **80195**
- Số PC cần để giải thích >=80% phương sai: **78/142**
- Phương sai giải thích bởi PC1..PC5: PC1=0.0335, PC2=0.0235, PC3=0.0188, PC4=0.0168, PC5=0.0159

## Top 5 đặc trưng đóng góp lớn nhất cho mỗi PC (trong top 5 PC)

**PC1** (giải thích 3.35% phương sai):
  - `change_Ch`: trọng số 0.3959
  - `change_No`: trọng số -0.3959
  - `diabetesMed_Yes`: trọng số 0.3767
  - `diabetesMed_No`: trọng số -0.3767
  - `insulin_No`: trọng số -0.2838

**PC2** (giải thích 2.35% phương sai):
  - `admission_source_id_7`: trọng số 0.4779
  - `admission_type_id_1`: trọng số 0.4733
  - `admission_source_id_1`: trọng số -0.427
  - `admission_type_id_3`: trọng số -0.3496
  - `num_lab_procedures`: trọng số 0.2107

**PC3** (giải thích 1.88% phương sai):
  - `discharge_disposition_id_1`: trọng số 0.3758
  - `time_in_hospital`: trọng số -0.3218
  - `num_medications`: trọng số -0.2961
  - `number_diagnoses`: trọng số -0.2788
  - `discharge_disposition_id_3`: trọng số -0.2618

**PC4** (giải thích 1.68% phương sai):
  - `glyburide_No`: trọng số 0.4141
  - `glyburide_Steady`: trọng số -0.3847
  - `insulin_No`: trọng số -0.2471
  - `race_Caucasian`: trọng số -0.2363
  - `metformin_No`: trọng số 0.2224

**PC5** (giải thích 1.59% phương sai):
  - `glipizide_No`: trọng số 0.5013
  - `glipizide_Steady`: trọng số -0.4796
  - `glyburide_No`: trọng số -0.3242
  - `glyburide_Steady`: trọng số 0.2999
  - `admission_source_id_17`: trọng số -0.191
