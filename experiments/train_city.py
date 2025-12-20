"""Train city-specific climate policies"""
import argparse
import json
import pickle
import sys
from pathlib import Path
sys.path.append('.')

import jax
import jax.numpy as jnp
import optax
import numpy as np
import pandas as pd

from data.city_configs import load_city_config
from src.dynamics import simulate_trajectory
from src.losses import compute_loss, compute_loss_components
from src.models import init_policy_params, get_trajectory_policies

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', type=str, required=True, choices=['nyc', 'la'])
    parser.add_argument('--w_E', type=float, default=1.0)
    parser.add_argument('--w_G', type=float, default=1.0)
    parser.add_argument('--lambda_term', type=float, default=10.0)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--num_iters', type=int, default=1000)
    parser.add_argument('--T', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--use_traffic', type=str, default='true')
    return parser.parse_args()

def setup_parameters(args):
    print("\n" + "="*60)
    print("PARAMETER SETUP")
    print("="*60)
    
    with open('data/processed/parameters.json') as f:
        global_params = json.load(f)
    
    city_config = load_city_config(args.city)
    
    print(f"\nCity: {city_config['name']}")
    print(f"  Archetype: {city_config['archetype']}")
    
    params = {**global_params}
    params.update(city_config['traffic_params'])
    params.update(city_config['economic_params'])
    params.update(city_config['initial_conditions'])
    params['use_traffic_model'] = args.use_traffic.lower() == 'true'
    
    loss_weights = {'w_E': args.w_E, 'w_G': args.w_G, 'lambda': args.lambda_term}
    
    print(f"\nParameters:")
    print(f"  GDP: ${params['G_0']:.1f}B")
    print(f"  Emissions: {params['E_0']:.1f} MtCO2/yr")
    print(f"  Traffic: {params['base_traffic_volume']} veh/hr")
    print(f"  Transit share: {params['base_transit_share']*100:.0f}%")
    
    print(f"\nLoss weights: w_E={loss_weights['w_E']}, w_G={loss_weights['w_G']}, λ={loss_weights['lambda']}")
    
    return params, loss_weights, city_config

def train(args):
    print("\n" + "="*60)
    print(f"TRAINING: {args.name}")
    print("="*60)
    print(f"Device: {jax.devices()}")
    
    params, loss_weights, city_config = setup_parameters(args)
    
    initial_state = {'CO2': params['CO2_0'], 'GDP': params['G_0'], 't': 0}
    state_params = {'CO2_0': params['CO2_0'], 'G_0': params['G_0']}
    
    print(f"\nInitial: CO2={initial_state['CO2']:.1f} ppm, GDP=${initial_state['GDP']:.1f}B")
    
    key = jax.random.PRNGKey(args.seed)
    policy_params = init_policy_params(key)
    
    optimizer = optax.adam(args.lr)
    opt_state = optimizer.init(policy_params)
    
    print(f"\nTraining: T={args.T} years, lr={args.lr}, iters={args.num_iters}")
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)
    
    losses = []
    best_loss = float('inf')
    best_params = None
    
    for iteration in range(args.num_iters):
        def loss_fn(policy_params):
            policies = get_trajectory_policies(policy_params, initial_state, args.T, state_params)
            
            # Policy regularization - penalize extreme policies
            policy_penalty = jnp.mean(policies ** 2) * 2.0
            
            subsidy_penalty = jnp.mean(policies[:, 1] ** 3) * 1.0
            trajectory = simulate_trajectory(initial_state, policies, params)
            base_loss = compute_loss(trajectory, params, loss_weights)
            
            return base_loss + policy_penalty + subsidy_penalty
        
        loss_value, grads = jax.value_and_grad(loss_fn)(policy_params)
        updates, opt_state = optimizer.update(grads, opt_state)
        policy_params = optax.apply_updates(policy_params, updates)
        
        losses.append(float(loss_value))
        
        if loss_value < best_loss:
            best_loss = loss_value
            best_params = policy_params
        
        if iteration % 100 == 0 or iteration == args.num_iters - 1:
            print(f"  Iter {iteration:4d}: Loss = {loss_value:8.2f} (best: {best_loss:8.2f})")
    
    print(f"\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"  Final: {losses[-1]:.2f}")
    print(f"  Best: {best_loss:.2f}")
    print(f"  Improvement: {(losses[0]-best_loss)/losses[0]*100:.1f}%")
    
    policy_params = best_params
    
    # Save results
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n" + "="*60)
    print(f"SAVING RESULTS")
    print("="*60)
    print(f"Output: {output_path}")
    
    with open(output_path / 'policy_params.pkl', 'wb') as f:
        pickle.dump(policy_params, f)
    
    with open(output_path / 'losses.json', 'w') as f:
        json.dump(losses, f)
    
    final_policies = get_trajectory_policies(policy_params, initial_state, args.T, state_params)
    final_trajectory = simulate_trajectory(initial_state, final_policies, params)
    
    np.save(output_path / 'policies.npy', final_policies)
    
    trajectory_json = []
    for state in final_trajectory:
        state_json = {k: float(v) if not isinstance(v, (dict, int)) else v for k, v in state.items()}
        trajectory_json.append(state_json)
    
    with open(output_path / 'trajectory.json', 'w') as f:
        json.dump(trajectory_json, f, indent=2)
    
    loss_breakdown = compute_loss_components(final_trajectory, params, loss_weights)
    
    summary = {
        'experiment': {
            'name': args.name,
            'city': args.city,
            'city_name': city_config['name'],
            'archetype': city_config['archetype'],
            'timestamp': str(pd.Timestamp.now())
        },
        'config': vars(args),
        'city_params': {
            'initial_GDP': params['G_0'],
            'initial_emissions': params['E_0'],
            'initial_CO2': params['CO2_0'],
            'base_traffic': params['base_traffic_volume'],
            'base_transit_share': params['base_transit_share'],
            'base_ev_share': params['base_ev_share']
        },
        'training': {
            'iterations': len(losses),
            'final_loss': losses[-1],
            'best_loss': best_loss,
            'initial_loss': losses[0],
            'improvement_pct': (losses[0]-best_loss)/losses[0]*100
        },
        'results': {
            'final_CO2': final_trajectory[-1]['CO2'],
            'final_GDP': final_trajectory[-1]['GDP'],
            'co2_change_pct': (final_trajectory[-1]['CO2']/initial_state['CO2']-1)*100,
            'gdp_growth_pct': (final_trajectory[-1]['GDP']/initial_state['GDP']-1)*100,
            'total_emissions': sum(s.get('emissions',0) for s in final_trajectory[1:]),
            'avg_emissions': sum(s.get('emissions',0) for s in final_trajectory[1:])/args.T,
            'emissions_reduction_pct': (1-sum(s.get('emissions',0) for s in final_trajectory[1:])/(args.T*params['E_0']))*100,
            'avg_policies': {
                'tau': float(jnp.mean(final_policies[:,0])),
                's': float(jnp.mean(final_policies[:,1])),
                'c': float(jnp.mean(final_policies[:,2]))
            },
            'final_policies': {
                'tau': float(final_policies[-1,0]),
                's': float(final_policies[-1,1]),
                'c': float(final_policies[-1,2])
            }
        },
        'loss_breakdown': loss_breakdown
    }
    
    if params['use_traffic_model'] and 'ev_share' in final_trajectory[-1]:
        summary['results']['final_ev_share'] = final_trajectory[-1]['ev_share']
        summary['results']['ev_adoption_pct'] = (final_trajectory[-1]['ev_share']-params['base_ev_share'])*100
    
    with open(output_path / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"\nCity: {city_config['name']}")
    print(f"\nEconomic:")
    print(f"  GDP growth: +{summary['results']['gdp_growth_pct']:.1f}%")
    print(f"\nClimate:")
    print(f"  CO2 change: +{summary['results']['co2_change_pct']:.1f}%")
    print(f"  Emissions reduction: {summary['results']['emissions_reduction_pct']:.1f}%")
    print(f"\nOptimal Policies (avg):")
    print(f"  τ={summary['results']['avg_policies']['tau']:.3f}")
    print(f"  s={summary['results']['avg_policies']['s']:.3f}")
    print(f"  c={summary['results']['avg_policies']['c']:.3f}")
    
    print(f"\n" + "="*60)
    print(" COMPLETE!")
    print("="*60)
    
    return policy_params, final_policies, final_trajectory, summary

if __name__ == "__main__":
    args = parse_args()
    train(args)
