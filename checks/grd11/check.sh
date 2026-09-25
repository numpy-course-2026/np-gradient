#!/usr/bin/env bash
# GRD-11: сохранение и загрузка модели. Циклы по слоям и параметрам разрешены.
set -u
no_loops gradient/store.py describe:1 build:2 save_model:1 load_model:2
pytest_run "тесты GRD-11" checks/grd11
