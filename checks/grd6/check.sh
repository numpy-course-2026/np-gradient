#!/usr/bin/env bash
# GRD-6: softmax-регрессия и проверка градиента. В fit — два цикла (эпохи и батчи),
# в gradient_check — один: это единственная функция проекта, где цикл по элементам законен.
set -u
no_loops gradient/logistic.py logits predict_proba predict loss gradients fit:2
no_loops gradient/check.py gradient_check:1
pytest_run "тесты GRD-6" checks/grd6
