#!/usr/bin/env bash
# GRD-3: стандартизация, one-hot, kNN и метрики. В knn_predict разрешён один цикл — по кускам тестовой выборки.
set -u
no_loops gradient/preprocess.py fit transform one_hot
no_loops gradient/knn.py knn_predict:1
no_loops gradient/metrics.py accuracy confusion_matrix
pytest_run "тесты GRD-3" checks/grd3
