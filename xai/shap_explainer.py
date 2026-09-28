"""
Fast SHAP Feature Attribution Explainer (SIH26054 Section 17)
Computes local feature importance rankings on normalized residual vectors in <20ms.
"""
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import torch

from models.anomaly_autoencoder import RESIDUAL_CHANNELS
from models.fault_classifier import FaultClassifierNet, FAULT_CLASSES, CLASS_TO_IDX

# Human-readable labels for telemetry channels
CHANNEL_DISPLAY_NAMES = {
    "rpm": "Engine Crankshaft RPM",
    "manifold_pressure": "Manifold Absolute Pressure (MAP)",
    "fuel_flow": "Fuel Mass Flow Rate",
    "oil_pressure": "Engine Lubrication Oil Pressure",
    "oil_temp": "Sump Oil Temperature",
    "cht_cyl_1": "Cylinder 1 Head Temp (CHT1)",
    "cht_cyl_2": "Cylinder 2 Head Temp (CHT2)",
    "cht_cyl_3": "Cylinder 3 Head Temp (CHT3)",
    "cht_cyl_4": "Cylinder 4 Head Temp (CHT4)",
    "egt_cyl_1": "Cylinder 1 Exhaust Gas Temp (EGT1)",
    "egt_cyl_2": "Cylinder 2 Exhaust Gas Temp (EGT2)",
    "egt_cyl_3": "Cylinder 3 Exhaust Gas Temp (EGT3)",
    "egt_cyl_4": "Cylinder 4 Exhaust Gas Temp (EGT4)",
    "vibration_rms": "Tri-Axial Vibration RMS (2X Order)",
    "bus_voltage": "Avionics Bus Voltage"
}

class FastSHAPExplainer:
    """
    Gradient-based / Kernel surrogate feature attribution for real-time diagnostic transparency.
    """
    def __init__(self, classifier_model: Optional[FaultClassifierNet] = None):
        self.classifier = classifier_model
        self.device = torch.device("cpu")

    def explain(
        self,
        residual_dict: Dict[str, float],
        target_class: str,
        top_k: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Calculates local feature importance for the predicted fault class.
        Returns ranked list of top_k contributing channels with importance percentages.
        """
        if target_class == "HEALTHY":
            return []

        # Vectorize residuals
        x_raw = np.array([residual_dict.get(ch, 0.0) for ch in RESIDUAL_CHANNELS], dtype=np.float32)

        # 1. Direct Gradient * Input Attribution (Integrated Gradients / Saliency proxy)
        if self.classifier is not None:
            self.classifier.eval()
            x_t = torch.tensor(x_raw.reshape(1, -1), requires_grad=True, device=self.device)
            logits, _ = self.classifier(x_t)
            class_idx = CLASS_TO_IDX.get(target_class, 0)
            score = logits[0, class_idx]
            score.backward()
            grads = x_t.grad.cpu().numpy()[0]
            attributions = np.abs(grads * x_raw)
        else:
            # Analytical residual-magnitude proxy when classifier model is uninitialized
            attributions = np.abs(x_raw)

        # Normalize importance to sum to 100%
        total_imp = float(np.sum(attributions))
        if total_imp < 1e-6:
            norm_imp = np.zeros_like(attributions)
        else:
            norm_imp = (attributions / total_imp) * 100.0

        # Sort features by importance descending
        sorted_indices = np.argsort(norm_imp)[::-1]

        explanations = []
        for idx in sorted_indices[:top_k]:
            ch_name = RESIDUAL_CHANNELS[idx]
            imp_val = float(norm_imp[idx])
            if imp_val < 3.0: # Skip negligible features
                continue

            res_val = float(x_raw[idx])
            direction = "ABOVE_BASELINE" if res_val >= 0 else "BELOW_BASELINE"

            explanations.append({
                "channel_key": ch_name,
                "display_name": CHANNEL_DISPLAY_NAMES.get(ch_name, ch_name),
                "importance_pct": round(imp_val, 1),
                "residual_sigma": round(res_val, 2),
                "direction": direction,
                "is_positive": (res_val >= 0),
                "deviation_text": f"{abs(res_val):.1f} sigma {'Above' if res_val >= 0 else 'Below'} Expected Baseline"
            })

        return explanations
