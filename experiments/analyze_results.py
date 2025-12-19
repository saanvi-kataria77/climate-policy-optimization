"""Quick analysis of 2-city results"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

JOBID = "5968803"

def load_all_results():
    """Load all experiment results"""
    results = []
    
    base_dir = Path(f"results/multi_city/{JOBID}")
    
    for city_dir in base_dir.iterdir():
        if not city_dir.is_dir():
            continue
        city = city_dir.name
        
        for config_dir in city_dir.iterdir():
            if not config_dir.is_dir():
                continue
            config = config_dir.name
            
            # Load trajectory
            traj_file = config_dir / "trajectory.json"
            if not traj_file.exists():
                print(f"Missing: {city}/{config}")
                continue
            
            with open(traj_file) as f:
                trajectory = json.load(f)
            
            # Load policies
            policies = np.load(config_dir / "policies.npy")
            
            # Load losses
            with open(config_dir / "losses.json") as f:
                losses = json.load(f)
            
            # Extract key metrics
            initial_gdp = trajectory[0]['GDP']
            final_gdp = trajectory[-1]['GDP']
            initial_co2 = trajectory[0]['CO2']
            final_co2 = trajectory[-1]['CO2']
            
            total_emissions = sum(s.get('emissions', 0) for s in trajectory[1:])
            
            results.append({
                'city': city,
                'config': config,
                'initial_gdp': initial_gdp,
                'final_gdp': final_gdp,
                'gdp_growth_pct': (final_gdp/initial_gdp - 1) * 100,
                'initial_co2': initial_co2,
                'final_co2': final_co2,
                'co2_change_pct': (final_co2/initial_co2 - 1) * 100,
                'total_emissions': total_emissions,
                'avg_tau': float(np.mean(policies[:, 0])),
                'avg_s': float(np.mean(policies[:, 1])),
                'avg_c': float(np.mean(policies[:, 2])),
                'final_loss': losses[-1],
                'trajectory': trajectory,
                'policies': policies
            })
    
    return pd.DataFrame(results)

def create_plots(df):
    """Create comparison plots"""
    
    output_dir = Path("results/analysis")
    output_dir.mkdir(exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # Plot 1: Policy Comparison by City
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, (policy, label) in enumerate([('avg_tau', 'Carbon Tax (τ)'), 
                                            ('avg_s', 'Transit Subsidy (s)'),
                                            ('avg_c', 'Congestion Charge (c)')]):
        ax = axes[idx]
        pivot = df.pivot(index='config', columns='city', values=policy)
        pivot.plot(kind='bar', ax=ax)
        ax.set_title(f'{label}', fontsize=14, fontweight='bold')
        ax.set_ylabel('Policy Level (0-1)', fontsize=12)
        ax.set_xlabel('Configuration', fontsize=12)
        ax.legend(title='City')
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'policy_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'policy_comparison.png'}")
    plt.close()
    
    # Plot 2: Trade-offs
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for city in df['city'].unique():
        city_data = df[df['city'] == city]
        ax.scatter(city_data['co2_change_pct'], city_data['gdp_growth_pct'], 
                  s=200, alpha=0.7, label=city.upper())
        
        # Add config labels
        for _, row in city_data.iterrows():
            ax.annotate(row['config'][:3], 
                       (row['co2_change_pct'], row['gdp_growth_pct']),
                       fontsize=8, ha='center')
    
    ax.set_xlabel('CO₂ Change (%)', fontsize=14, fontweight='bold')
    ax.set_ylabel('GDP Growth (%)', fontsize=14, fontweight='bold')
    ax.set_title('Climate-Economic Trade-offs: NYC vs LA', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax.axvline(0, color='gray', linestyle='--', linewidth=1)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'tradeoffs.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'tradeoffs.png'}")
    plt.close()
    
    # Plot 3: City Comparison Summary
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # GDP Growth
    ax = axes[0, 0]
    df.groupby('city')['gdp_growth_pct'].mean().plot(kind='bar', ax=ax, color=['#1f77b4', '#ff7f0e'])
    ax.set_title('Average GDP Growth by City', fontsize=14, fontweight='bold')
    ax.set_ylabel('GDP Growth (%)', fontsize=12)
    ax.set_xlabel('')
    ax.grid(alpha=0.3)
    
    # CO2 Change
    ax = axes[0, 1]
    df.groupby('city')['co2_change_pct'].mean().plot(kind='bar', ax=ax, color=['#1f77b4', '#ff7f0e'])
    ax.set_title('Average CO₂ Change by City', fontsize=14, fontweight='bold')
    ax.set_ylabel('CO₂ Change (%)', fontsize=12)
    ax.set_xlabel('')
    ax.grid(alpha=0.3)
    
    # Policy Mix - NYC
    ax = axes[1, 0]
    nyc_policies = df[df['city'] == 'nyc'][['avg_tau', 'avg_s', 'avg_c']].mean()
    nyc_policies.plot(kind='bar', ax=ax, color='#1f77b4')
    ax.set_title('NYC: Average Policy Mix', fontsize=14, fontweight='bold')
    ax.set_ylabel('Policy Level (0-1)', fontsize=12)
    ax.set_xticklabels(['Carbon Tax', 'Subsidy', 'Congestion'], rotation=45)
    ax.grid(alpha=0.3)
    
    # Policy Mix - LA
    ax = axes[1, 1]
    la_policies = df[df['city'] == 'la'][['avg_tau', 'avg_s', 'avg_c']].mean()
    la_policies.plot(kind='bar', ax=ax, color='#ff7f0e')
    ax.set_title('LA: Average Policy Mix', fontsize=14, fontweight='bold')
    ax.set_ylabel('Policy Level (0-1)', fontsize=12)
    ax.set_xticklabels(['Carbon Tax', 'Subsidy', 'Congestion'], rotation=45)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'city_summary.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'city_summary.png'}")
    plt.close()

def generate_summary(df):
    """Generate text summary"""
    
    output_dir = Path("results/analysis")
    
    summary = []
    summary.append("# Climate Policy Optimization Results")
    summary.append(f"\n## Overview")
    summary.append(f"- Cities: NYC, LA")
    summary.append(f"- Configurations: 4 per city (balanced, emissions_focus, growth_focus, aggressive)")
    summary.append(f"- Total experiments: {len(df)}")
    summary.append(f"- Training: 200 iterations per experiment")
    
    summary.append(f"\n## Key Findings")
    
    # NYC vs LA comparison
    nyc_avg = df[df['city'] == 'nyc'].mean(numeric_only=True)
    la_avg = df[df['city'] == 'la'].mean(numeric_only=True)
    
    summary.append(f"\n### NYC (Dense, Transit-Rich)")
    summary.append(f"- Average GDP Growth: {nyc_avg['gdp_growth_pct']:.1f}%")
    summary.append(f"- Average CO₂ Change: {nyc_avg['co2_change_pct']:.1f}%")
    summary.append(f"- Optimal Policy Mix:")
    summary.append(f"  - Carbon Tax (τ): {nyc_avg['avg_tau']:.3f}")
    summary.append(f"  - Transit Subsidy (s): {nyc_avg['avg_s']:.3f}")
    summary.append(f"  - Congestion Charge (c): {nyc_avg['avg_c']:.3f}")
    
    summary.append(f"\n### LA (Sprawl, Car-Dependent)")
    summary.append(f"- Average GDP Growth: {la_avg['gdp_growth_pct']:.1f}%")
    summary.append(f"- Average CO₂ Change: {la_avg['co2_change_pct']:.1f}%")
    summary.append(f"- Optimal Policy Mix:")
    summary.append(f"  - Carbon Tax (τ): {la_avg['avg_tau']:.3f}")
    summary.append(f"  - Transit Subsidy (s): {la_avg['avg_s']:.3f}")
    summary.append(f"  - Congestion Charge (c): {la_avg['avg_c']:.3f}")
    
    summary.append(f"\n### Policy Differences")
    summary.append(f"- LA uses {(la_avg['avg_tau'] - nyc_avg['avg_tau']):.3f} MORE carbon tax")
    summary.append(f"- LA uses {(la_avg['avg_s'] - nyc_avg['avg_s']):.3f} MORE transit subsidy")
    summary.append(f"- NYC uses {(nyc_avg['avg_c'] - la_avg['avg_c']):.3f} MORE congestion pricing")
    
    summary.append(f"\n## Best Configurations")
    
    for city in ['nyc', 'la']:
        city_data = df[df['city'] == city]
        best = city_data.loc[city_data['final_loss'].idxmin()]
        
        summary.append(f"\n### {city.upper()}: {best['config']}")
        summary.append(f"- GDP Growth: {best['gdp_growth_pct']:.1f}%")
        summary.append(f"- CO₂ Change: {best['co2_change_pct']:.1f}%")
        summary.append(f"- Policies: τ={best['avg_tau']:.2f}, s={best['avg_s']:.2f}, c={best['avg_c']:.2f}")
    
    text = "\n".join(summary)
    
    with open(output_dir / "SUMMARY.md", "w") as f:
        f.write(text)
    
    print(f"✓ Saved: {output_dir / 'SUMMARY.md'}")
    print("\n" + text)
    
    return text

if __name__ == "__main__":
    print("="*60)
    print("ANALYZING RESULTS")
    print("="*60)
    
    # Load data
    print("\nLoading results...")
    df = load_all_results()
    print(f"✓ Loaded {len(df)} experiments")
    
    # Save CSV
    output_dir = Path("results/analysis")
    output_dir.mkdir(exist_ok=True)
    df.to_csv(output_dir / "all_results.csv", index=False)
    print(f"✓ Saved: {output_dir / 'all_results.csv'}")
    
    # Create plots
    print("\nCreating visualizations...")
    create_plots(df)
    
    # Generate summary
    print("\nGenerating summary...")
    generate_summary(df)
    
    print("\n" + "="*60)
    print("✓ ANALYSIS COMPLETE!")
    print("="*60)
    print("\nOutputs in: results/analysis/")
    print("  - all_results.csv")
    print("  - policy_comparison.png")
    print("  - tradeoffs.png")
    print("  - city_summary.png")
    print("  - SUMMARY.md")