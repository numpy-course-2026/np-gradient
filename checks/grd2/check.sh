#!/usr/bin/env bash
# GRD-2: gradient/split.py — разбиение и батчи. В batches разрешён один цикл — по батчам.
set -u
no_loops gradient/split.py train_test_split batches:1
pytest_run "тесты GRD-2" checks/grd2
