# Climate Policy Optimization — City Configurations Analysis

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![HPC](https://img.shields.io/badge/HPC-UMD%20Nexus%20%2F%20SLURM-orange)](https://hpcc.umd.edu/)

This is a Python-based tool for analyzing and comparing city configurations to support climate policy research. It generates summary statistics for cities, which includes GDP, emissions, traffic volume, and transit share and is designed to help urban planners and researchers evaluate city-level data and make informed, data-driven decisions.

---

## ✨ Features

- **City Summary Statistics**: Generate detailed summaries of city configurations, including derived metrics like emissions per capita and traffic volume-to-capacity ratios.
- **Metric Comparison**: Compare specific metrics (e.g., EV share, transit share) across multiple cities using a simple dot-notation path.
- **Customizable Configurations**: Easily extend or modify city configurations to suit your analysis needs.
- **HPC-Ready**: Designed to run on high-performance computing clusters via SLURM job scheduling.


## ⚙️ Setup Instructions

### Prerequisites

- Python 3.8 or higher
- `pip` (Python package manager)
- * Access to UMD's Nexus HPC cluster with SLURM scheduler

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/saanvi-kataria77/climate-policy-optimization.git
   cd climate-policy-optimization
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate        # macOS/Linux
   venv\Scripts\activate           # Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. *(Optional)* **HPC/SLURM setup** — If running on UMD's Nexus cluster, ensure you have access to the SLURM scheduler and configure your environment module accordingly (see [Running on SLURM](#-running-on-slurm-umd-nexus) below).

---

## 🚀 Usage

### Generate City Summary

Run the main script to print a summary of all cities:

```bash
python city_configs.py
```

### Compare Specific Metrics

Use the `compare_cities` function to compare a specific metric across cities. For example, to compare EV share:

```python
compare_cities('traffic_params.base_ev_share')
```

You can pass any dot-notation path corresponding to a nested field in the city config dictionaries.

---

## 🖥️ Running on SLURM (UMD Nexus)

This project is compatible with UMD's [Nexus HPC cluster](https://hpcc.umd.edu/). A basic SLURM job script might look like:

```bash
#!/bin/bash
#SBATCH --job-name=climate-policy
#SBATCH --output=output_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00

module load python/3.9
source venv/bin/activate
python city_configs.py
```

Submit with:
```bash
sbatch job.slurm
```

---

## My Note on Commit History

The commit history for this repository is relatively small. Development was done locally and connected to UMD's Nexus HPC cluster via a remote tunnel using SLURM for job execution. This workflow, where the code is edited locally, was synced to the cluster, and run compute nodes, so my team and I's development steps happened outside of standard Git commit cycles and were integrated in larger batches rather than committed in real time.

---

## Contributing

Contributions are welcome! To get started:

1. Fork the repository.
2. Create a new branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Commit your changes with a descriptive message:
   ```bash
   git commit -m "Add: description of your feature"
   ```
4. Push to your fork and open a pull request against `main`.

Please make sure your code follows existing style conventions and includes comments where appropriate.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 📬 Contact

Questions, suggestions, or feedback? Feel free to [open an issue](https://github.com/saanvi-kataria77/climate-policy-optimization/issues) or reach out directly via GitHub, thanks so much! :)
