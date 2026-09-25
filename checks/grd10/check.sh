#!/usr/bin/env bash
# GRD-10: свёртка, пулинг, вытягивание. Циклов по пикселям и окнам быть не должно.
set -u
no_loops gradient/conv.py windows Conv2D.forward Conv2D.backward MaxPool2D.forward MaxPool2D.backward Flatten.forward Flatten.backward
no_loops gradient/data.py load_digits28
pytest_run "тесты GRD-10" checks/grd10
