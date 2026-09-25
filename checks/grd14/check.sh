#!/usr/bin/env bash
# GRD-14: релиз 1.0 — свёрточная сеть, отчёт, сохранение модели и картинок.
set -u
pytest_run "тесты GRD-14" checks/grd14
