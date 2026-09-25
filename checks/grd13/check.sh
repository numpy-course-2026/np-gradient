#!/usr/bin/env bash
# GRD-13: картинки и сравнение оптимизаторов. Циклы по кривым, клеткам матрицы и оптимизаторам законны.
set -u
no_loops gradient/report.py learning_curves:1 compare_optimizers:3
pytest_run "тесты GRD-13" checks/grd13
