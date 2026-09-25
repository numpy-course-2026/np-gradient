#!/usr/bin/env bash
# GRD-8: слои Dense и ReLU, сеть MLP. Циклы: по слоям, по эпохам и батчам.
set -u
no_loops gradient/layers.py Dense.forward Dense.backward ReLU.forward ReLU.backward MLP.forward:1 MLP.backward:1 MLP.step:1 MLP.fit:2
pytest_run "тесты GRD-8" checks/grd8
