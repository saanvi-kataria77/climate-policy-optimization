# 🌍 Urban Climate Policy Optimization

> This project's main goal is to use differentiable programming to find combinations of city-specific climate policies that actually work.

Cities are responsible for 70% of global CO₂ emissions. We thought of questions about certain cities and their usage of climate policies. For example. should Los Angeles use the same carbon tax as New York? This differentiable climate policy model helps figure that out.

## What This Does

This project optimizes three climate policy levers, **carbon taxes**, **transit subsidies**, and **congestion charges**, specifically for different types of cities. By making the entire climate-economic system differentiable (using JAX), we can use gradient descent to search for optimal policies in minutes rather than running millions of simulations.

**Key Finding:** Dense, transit-rich cities (NYC) need different policies than sprawling, car-dependent cities (LA). Specifically:
- LA requires **10% stronger carbon pricing** due to car-dependency
- LA needs **35% more congestion charges** (yes, even sprawling cities have congestion!)
- Both cities max out **transit subsidies at 60%**—the most robust policy lever

## 🏗️ How It Works
```
Policies (τ, s, c) → Emissions → Atmospheric CO₂ → Climate Damage → GDP
       ↑                                                              ↓
       └─────────────────── Gradients via JAX ←──────────────────────┘
```


## Here's how you can get started!

### Installation
```bash
git clone https://github.com/saanvi-kataria77/climate-policy-optimization-.git
cd climate-policy-optimization-
pip install jax optax numpy pandas matplotlib seaborn
```

### Run Everything
```bash
# 1. generates climate-economic data
python data/download_data.py
python data/preprocess_data.py

# 2. run a quick test to train policies for a city
python experiments/train_city.py --city nyc --num_iters 100

# 3. run full 8-experiment cluster (keep it mind it requires SLURM) 
sbatch jobs/multi_city_experiments.sh

# 4. analyzes and creates results 
python experiments/analyze_results.py
```

### What You Get

After running analysis, check `results/analysis/`:
- 📊 **policy_comparison.png** - Bar charts showing NYC vs LA policy differences
- 📈 **tradeoffs.png** - GDP growth vs CO₂ reduction scatter plot
- 📑 **SUMMARY.md** - Text report with key findings
- 📄 **all_results.csv** - Full data table for further analysis

## Project Structure
```
├── src/
│   ├── dynamics.py      # Climate-economic physics (CO₂, emissions, GDP)
│   ├── models.py        # Policy network (state → actions)
│   └── losses.py        # Multi-objective loss function
├── data/
│   ├── city_configs.py  # NYC & LA parameters from real data
│   ├── download_data.py # Get climate/economic data
│   └── preprocess_data.py # Calibrate model parameters
├── experiments/
│   ├── train_city.py    # Main optimization script
│   └── analyze_results.py # Generate plots & summaries
├── jobs/
│   └── multi_city_experiments.sh # SLURM batch script
└── results/
    ├── multi_city/      # Raw optimization outputs
    └── analysis/        # Plots and summaries
```

## 🔬 Results Summary from Our own Experiment

**New York City (Dense, Transit-Rich):**
- 80% GDP growth over 30 years
- 9% CO₂ reduction
- Optimal mix: τ=0.40, s=0.60, c=0.14

**Los Angeles (Sprawl, Car-Dependent):**
- 151% GDP growth over 30 years
- 7.5% CO₂ reduction  
- Optimal mix: τ=0.44, s=0.60, c=0.19

**Trade-offs:**
- Emissions-focused config: -27% CO₂ but lower growth
- Growth-focused config: +190% GDP but +21% CO₂ (!)
- Balanced config finds sweet spot: strong growth + emissions cuts

##  Key Insights

1. **One-size-fits-all policies don't work** - Urban structure fundamentally shapes what policies are effective
2. **Transit investment is universal** - Both cities converge on 60% subsidy regardless of configuration
3. **You can't grow without pricing** - Subsidy-only approaches lead to emissions increases (+21%)
4. **Multi-objective optimization is hard** - We went through 3 iterations to get realistic results!

##  Built With

- **JAX** - Automatic differentiation for gradient-based optimization
- **Optax** - Adam optimizer implementation
- **NumPy/Pandas** - Data wrangling
- **Matplotlib/Seaborn** - Visualizations
- **SLURM** - Parallel job execution on cluster

## Limitations & Future Work

**Current Limitations:**
- Only two cities (need 10+ for generalization)
- No spatial heterogeneity within cities

**Future Directions:**
- Expand to diverse urban typologies (Tokyo, Lagos, Mexico City...)
- Validate against real historical policy experiments
- Deploy as interactive tool for urban planners


## 🤝 Contributing

This was a course project, but feel free to fork and extend! Interesting directions:
- Add more cities from different continents
- Incorporate real World Bank GDP data (see `data/download_data.py` comments)
- Build interactive visualization dashboard


## 🙏 Acknowledgments

- **UMIACS Nexus Cluster** for compute resources
- **JAX team** for making differentiable programming accessible
- Inspired by Nordhaus's DICE model and recent work on differentiable scientific computing

## ⚠️ Disclaimer

This is research code from a course project, not production-ready policy software. Don't use these exact numbers to set real carbon taxes! That said, the methodology is sound and the insights are interesting!!

---
