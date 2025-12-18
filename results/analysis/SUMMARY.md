# Climate Policy Optimization Results

## Overview
- Cities: NYC, LA
- Configurations: 4 per city (balanced, emissions_focus, growth_focus, aggressive)
- Total experiments: 8
- Training: 200 iterations per experiment

## Key Findings

### NYC (Dense, Transit-Rich)
- Average GDP Growth: -52.0%
- Average CO₂ Change: -62.1%
- Optimal Policy Mix:
  - Carbon Tax (τ): 1.000
  - Transit Subsidy (s): 1.000
  - Congestion Charge (c): 1.000

### LA (Sprawl, Car-Dependent)
- Average GDP Growth: 41.0%
- Average CO₂ Change: -58.7%
- Optimal Policy Mix:
  - Carbon Tax (τ): 1.000
  - Transit Subsidy (s): 1.000
  - Congestion Charge (c): 1.000

### Policy Differences
- LA uses -0.000 MORE carbon tax
- LA uses -0.000 MORE transit subsidy
- NYC uses 0.000 MORE congestion pricing

## Best Configurations

### NYC: growth_focus
- GDP Growth: -52.0%
- CO₂ Change: -62.1%
- Policies: τ=1.00, s=1.00, c=1.00

### LA: growth_focus
- GDP Growth: 41.0%
- CO₂ Change: -58.7%
- Policies: τ=1.00, s=1.00, c=1.00