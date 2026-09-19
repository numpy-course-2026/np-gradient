#!/usr/bin/env bash
# GRD-4: gradient/functional.py — устойчивые logsumexp, softmax, sigmoid, cross-entropy.
set -u
no_loops gradient/functional.py logsumexp log_softmax softmax sigmoid cross_entropy
pytest_run "тесты GRD-4" checks/grd4
