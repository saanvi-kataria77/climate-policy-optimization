"""Loss function"""
import jax.numpy as jnp

def compute_loss(trajectory, params, loss_weights):
    """L = Σ[w_E*(E/E_0) - w_G*(G/G_0)] + λ*(CO2_T/CO2_0)²"""
    w_E = loss_weights['w_E']
    w_G = loss_weights['w_G']
    lambda_term = loss_weights['lambda']
    
    E_0 = params['E_0']
    G_0 = params['G_0']
    CO2_0 = params['CO2_0']
    
    total_loss = 0.0
    
    # Running cost
    for state in trajectory[1:-1]:
        emissions_penalty = w_E * (state['emissions'] / E_0)
        gdp_reward = -w_G * (state['GDP'] / G_0)
        total_loss += emissions_penalty + gdp_reward
    
    # Terminal penalty
    final_CO2 = trajectory[-1]['CO2']
    terminal_penalty = lambda_term * (final_CO2 / CO2_0) ** 2
    total_loss += terminal_penalty
    
    return total_loss

def compute_loss_components(trajectory, params, loss_weights):
    """Break down loss for analysis"""
    w_E = loss_weights['w_E']
    w_G = loss_weights['w_G']
    lambda_term = loss_weights['lambda']
    
    E_0 = params['E_0']
    G_0 = params['G_0']
    CO2_0 = params['CO2_0']
    
    emissions_cost = 0.0
    gdp_cost = 0.0
    
    for state in trajectory[1:-1]:
        emissions_cost += w_E * (state['emissions'] / E_0)
        gdp_cost += -w_G * (state['GDP'] / G_0)
    
    final_CO2 = trajectory[-1]['CO2']
    terminal_cost = lambda_term * (final_CO2 / CO2_0) ** 2
    
    total_loss = emissions_cost + gdp_cost + terminal_cost
    
    return {
        'total_loss': float(total_loss),
        'emissions_cost': float(emissions_cost),
        'gdp_cost': float(gdp_cost),
        'terminal_cost': float(terminal_cost),
        'avg_emissions_normalized': float(jnp.mean(jnp.array([s['emissions']/E_0 for s in trajectory[1:-1]]))),
        'avg_gdp_normalized': float(jnp.mean(jnp.array([s['GDP']/G_0 for s in trajectory[1:-1]]))),
        'final_co2_normalized': float(final_CO2 / CO2_0),
    }
