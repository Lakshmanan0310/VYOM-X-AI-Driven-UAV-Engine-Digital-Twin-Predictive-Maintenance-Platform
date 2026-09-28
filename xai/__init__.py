"""
ENGINE-TWIN: Explainable AI (XAI) Package
Feature attribution, SHAP surrogate explanations, and structured operator alert generation.
"""
from .shap_explainer import FastSHAPExplainer
from .alert_generator import AlertGenerator, OperatorAlert

__all__ = [
    "FastSHAPExplainer",
    "AlertGenerator",
    "OperatorAlert",
]
