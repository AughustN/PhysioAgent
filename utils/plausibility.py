from __future__ import annotations

import pandas as pd

PLAUSIBLE = {
    "age": (18, 110),
    "weight_kg": (25, 350),
    "height_cm": (120, 230),
    "systolic_bp": (50, 300),
    "diastolic_bp": (20, 200),
    "egfr": (1, 200),
}

REQUIRED_PLAUSIBLE = ["age", "weight_kg", "height_cm", "systolic_bp", "diastolic_bp"]


def implausible_mask(df: pd.DataFrame) -> pd.Series:
    bad = pd.Series(False, index=df.index)
    for col in REQUIRED_PLAUSIBLE:
        if col not in df.columns:
            continue
        lo, hi = PLAUSIBLE[col]
        bad |= df[col].notna() & ~df[col].between(lo, hi)
    return bad
