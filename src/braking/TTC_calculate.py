import numpy as np
from typing import List, Optional, Dict

# -----------------------------
# CONFIG (Fixed Constants)
# -----------------------------
EGO_SPEED_KMH = 30.0
OBJ_SPEED_KMH = 20.0
REL_V_MPS = (EGO_SPEED_KMH - OBJ_SPEED_KMH) / 3.6  # 2.78 m/s

# TTC thresholds (seconds)
TTC_EMERGENCY = 1.1
TTC_ALERT_HIGH = 2.9
TTC_ALERT_LOW = 4.6

# PRE-CALCULATED DISTANCE THRESHOLDS (The real optimization)
# Math: Distance = TTC * Velocity
DIST_EMERGENCY = TTC_EMERGENCY * REL_V_MPS # ~3.06m
DIST_ALERT_HIGH = TTC_ALERT_HIGH * REL_V_MPS # ~8.06m
DIST_ALERT_LOW = TTC_ALERT_LOW * REL_V_MPS   # ~12.79m

class TTCSystem:
    def __init__(self, rel_v: float = REL_V_MPS):
        self.rel_v = rel_v
        # Caching thresholds as local attributes for faster lookup
        self.d_emerg = DIST_EMERGENCY
        self.d_high = DIST_ALERT_HIGH
        self.d_low = DIST_ALERT_LOW

    def update(self, depths: List[float], timestamp: float) -> Optional[Dict]:
        """
        Ultra-fast update using distance thresholding instead of TTC division.
        """
        # 1. Faster min() check
        if not depths:
            return None
        
        # Note: In your integrated script, you already filter depths, 
        # so we can assume input here is clean for max speed.
        closest_depth = min(depths)

        # 2. Distance-based decision (No division needed!)
        if closest_depth < self.d_emerg:
            status, pwm = "DARURAT", 100
        elif closest_depth < self.d_high:
            status, pwm = "PARSIAL", 50
        elif closest_depth < self.d_low:
            status, pwm = "PARSIAL", 30
        else:
            status, pwm = "AMAN", 0

        # 3. We still return TTC for your HUD display
        # (Division only happens once for the UI, not for the decision logic)
        return {
            "depth": closest_depth,
            "rel_v": self.rel_v,
            "ttc": closest_depth / self.rel_v,
            "status": status,
            "pwm": pwm
        }