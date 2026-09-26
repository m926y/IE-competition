"""Shared horizon conversion for the final frozen predictor."""
from pathlib import Path
import json
import duckdb
import numpy as np
import pandas as pd
from scipy.optimize import brentq
ROOT = Path(__file__).resolve().parents[2]

def probability_at_horizon(p14, accident_fraction, days=40):
    """Independent stationary Poisson counts; positive iff A>=1 or N>=2.

    Invert the 14-day positive probability for total intensity. The accident
    fraction is estimated only from recorded historical events. This is an
    explicit extrapolation assumption, not observed 40-day calibration.
    """
    a = float(np.clip(accident_fraction,0,1))
    rates = np.asarray([brentq(lambda x: 1-np.exp(-x)*(1+(1-a)*x)-p,
                              0,100) for p in np.clip(p14,1e-7,1-1e-7)])
    rates *= days/14
    return np.clip(1-np.exp(-rates)*(1+(1-a)*rates),0,1)

def accident_fraction(cutoff):
    path=ROOT/'data/processed/cleaned/risk_events.parquet'
    with duckdb.connect() as con:
        a,n=con.execute('''SELECT count_if(event_type=11803),count_if(event_type=11804)
          FROM read_parquet(?) WHERE event_time < CAST(? AS DATE)+INTERVAL 1 DAY''',
          [str(path),str(pd.Timestamp(cutoff).date())]).fetchone()
    return float(a/(a+n)) if a+n else 0.0

def dump(path, value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,default=str),encoding='utf-8')
