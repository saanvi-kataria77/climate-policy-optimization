# Urban Climate Policy Optimization

Differentiable programming approach for city-specific climate policy optimization.

## Overview

This project uses JAX and gradient-based optimization to find optimal climate policies (carbon tax, transit subsidies, congestion charges) for different urban archetypes.

## Key Features

- Differentiable climate-economic model
- City-specific optimization (NYC vs LA)
- Multi-objective loss function
- Parallel training on GPU cluster

## Installation
```bash
pip install jax optax numpy pandas matplotlib seaborn
```

## Usage
```bash
# Generate data
python data/download_data.py
python data/preprocess_data.py

# Run optimization
sbatch jobs/multi_city_experiments.sh

# Analyze results
python experiments/analyze_results.py
```

## Results

[Add plots and key findings after tomorrow]

## Citation
```
@misc{yourname2024climate,
  title={Urban Climate Policy Optimization via Differentiable Programming},
  author={Your Name},
  year={2024}
}
```

## License

MIT