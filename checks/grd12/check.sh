#!/usr/bin/env bash
# GRD-12: оптимизаторы. Цикл по параметрам разрешён.
set -u
no_loops gradient/optim.py SGD.step:1 Momentum.step:2 Adam.step:3
pytest_run "тесты GRD-12" checks/grd12
