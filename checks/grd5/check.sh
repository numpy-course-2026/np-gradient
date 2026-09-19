#!/usr/bin/env bash
# GRD-5: gradient/linear.py — линейная регрессия, нормальные уравнения, lstsq, гребневый штраф.
set -u
no_loops gradient/linear.py add_bias fit predict
no_loops gradient/metrics.py mse
pytest_run "тесты GRD-5" checks/grd5
