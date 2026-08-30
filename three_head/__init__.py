"""Consolidated three-head surrogate for the ideal-conductance dynamic pilot.

Self-contained: every model, feature, metric and tail-reconstruction module used
by this pipeline lives in this package. The only external import is the upstream
device simulator, `memintelli.pimpy`, which is a frozen third-party snapshot and
is deliberately not vendored.
"""
