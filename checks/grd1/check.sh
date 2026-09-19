#!/usr/bin/env bash
# GRD-1: gradient/data.py — загрузка цифр.
set -u
no_loops gradient/data.py load_digits as_images class_counts
pytest_run "тесты GRD-1" checks/grd1
