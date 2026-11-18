#!/bin/bash
# Chạy pipeline theo đúng thứ tự notebook, từ thư mục branches/main
set -e
export PATH="/home/hdung274520/Study/261/Máy học Cơ bản và Ứng dụng/Dung/.venv/bin:$PATH"
cd "$(dirname "$0")/branches/main"
mkdir -p output
run(){ echo "=== $1"; python -u "$2"; }
run 01 ../Dung/src/01_explore_and_drop_ids.py
run 02 ../Son/src/02_missing_stats_and_filter.py
run 03 ../Chien/src/03_data_preprocessing_splitting.py
run 04 ../Dung/src/04_one_hot_encoding.py
run 05 ../Son/src/05_pca.py
run 06 ../Chien/src/06_feature_scaling_pca_analysis.py
run 07a ../Dung/src/07a_knn.py
run 07b ../Dung/src/07b_softmax_regression.py
run 08a ../Son/src/08a_decision_tree.py
run 08b ../Son/src/08b_random_forest.py
run 09a ../Chien/src/09a_mlp.py
run 09b ../Chien/src/09b_ensemble.py
run 10 ../Dung/src/10_evaluation.py
run 11 ../Son/src/11_binary_evaluation.py
run 12 ../Chien/src/12_imbalance_and_rf_raw.py
echo "=== DONE"
