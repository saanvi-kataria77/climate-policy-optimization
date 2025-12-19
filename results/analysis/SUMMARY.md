# Climate Policy Optimization Results

## Overview
- Cities: NYC, LA
- Configurations: 4 per city (balanced, emissions_focus, growth_focus, aggressive)
- Total experiments: 8
- Training: 200 iterations per experiment

## Key Findings

### NYC (Dense, Transit-Rich)
- Average GDP Growth: 80.1%
- Average CO₂ Change: -9.0%
- Optimal Policy Mix:
  - Carbon Tax (τ): 0.399
  - Transit Subsidy (s): 0.596
  - Congestion Charge (c): 0.142

### LA (Sprawl, Car-Dependent)
- Average GDP Growth: 151.3%
- Average CO₂ Change: -7.5%
- Optimal Policy Mix:
  - Carbon Tax (τ): 0.441
  - Transit Subsidy (s): 0.600
  - Congestion Charge (c): 0.191

### Policy Differences
- LA uses 0.042 MORE carbon tax
- LA uses 0.004 MORE transit subsidy
- NYC uses -0.050 MORE congestion pricing

## Best Configurations

### NYC: growth_focus
- GDP Growth: 111.3%
- CO₂ Change: 14.3%
- Policies: τ=0.00, s=0.60, c=0.00

### LA: growth_focus
- GDP Growth: 190.9%
- CO₂ Change: 21.4%
- Policies: τ=0.00, s=0.60, c=0.00