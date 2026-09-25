#!/usr/bin/env bash
# GRD-9: инициализация весов и dropout.
set -u
no_loops gradient/init.py init_weights
no_loops gradient/layers.py Dropout.forward Dropout.backward
pytest_run "тесты GRD-9" checks/grd9
