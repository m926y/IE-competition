"""Training-only exposure priors for seven fixed 14-day risk features."""
import numpy as np
import pandas as pd

GROUPS = ['severe_history', 'fatigue', 'distraction', 'stability', 'road_risk', 'management', 'blind_spot']


def fit_rate_priors(frame):
    priors = {}
    for group in GROUPS:
        count = group + '_count_14d'
        exposure = 'run_time_hours_14d' if group in ['fatigue', 'distraction'] else 'distance_km_14d'
        valid = frame[exposure].gt(0) & frame[count].notna()
        if not valid.any():
            raise ValueError(f'No observed positive exposure for {group}')
        priors[group] = dict(mu=float(frame.loc[valid, count].sum()/frame.loc[valid, exposure].sum()),
                             tau=float(frame.loc[valid, exposure].median()), exposure=exposure)
    return priors


def transform_rates(frame, priors):
    result = frame.copy()
    columns = []
    for group, prior in priors.items():
        name = group + '_shrunk_rate'
        exposure = result[prior['exposure']]
        result[name] = ((result[group + '_count_14d'] + prior['tau'] * prior['mu']) /
                        (exposure + prior['tau'])).where(exposure.gt(0))
        result[name] = result[name].replace([np.inf, -np.inf], np.nan)
        columns.append(name)
    return result, columns
