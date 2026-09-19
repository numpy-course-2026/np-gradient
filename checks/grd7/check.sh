#!/usr/bin/env bash
# GRD-7: релиз 0.1 — python -m gradient.train, отчёт, описание релиза.
set -u
pytest_run "тесты GRD-7" checks/grd7
