"""Preprocess and calibrate parameters"""
import pandas as pd
import numpy as np
import json
from pathlib import Path

data = pd.merge(pd.read_csv('data/raw/co2_data.csv'),
                pd.read_csv('data/raw/emissions_data.csv'), on='year')
data = pd.merge(data, pd.read_csv('data/raw/gdp_data.csv'), on='year')

Path('data/processed').mkdir(exist_ok=True)
data.to_csv('data/processed/climate_economic_data.csv', index=False)

# Calibrate parameters
gdp = data['gdp_trillion_usd'].values
g_0 = float(np.mean(np.diff(gdp)/gdp[:-1]))
gamma = float(np.polyfit(np.log(gdp), np.log(data['emissions_GtCO2']), 1)[0])

params = {
    'A': float(data['emissions_GtCO2'].iloc[0]),
    'alpha': 0.47,
    'delta': 0.023,
    'gamma': gamma,
    'g_0': g_0,
    'eta_tau': 1.0,
    'eta_c': 0.5,
    'kappa': 0.3,
    'beta_s': 0.02,
    'beta_tau': 0.01,
    'beta_c': 0.005,
    'theta': 0.00267,
    'phi': 2.0,
    'CO2_0': float(data['co2_ppm'].iloc[-1]),
    'G_0': float(data['gdp_trillion_usd'].iloc[-1]),
    'E_0': float(data['emissions_GtCO2'].iloc[-1]),
}

with open('data/processed/parameters.json', 'w') as f:
    json.dump(params, f, indent=2)

print(f"✓ Parameters calibrated: g_0={g_0*100:.2f}%, gamma={gamma:.3f}")
