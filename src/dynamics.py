"""Core climate-economic-traffic dynamics"""
import jax.numpy as jnp

def compute_emissions(state, action, params):
    """E_t = A * exp(-η_τ*τ - η_c*c) * (1-κ*s) * (G/G_0)^γ * traffic_factor"""
    G_t = state['GDP']
    τ, s, c = action
    
    policy_effect = jnp.exp(-params['eta_tau']*τ - params['eta_c']*c)
    subsidy_effect = 1 - params['kappa']*s
    economic_driver = (G_t / params['G_0']) ** params['gamma']
    base_emissions = params['A'] * policy_effect * subsidy_effect * economic_driver
    
    if params.get('use_traffic_model', False) and 'traffic_congestion' in state:
        congestion_mult = 1 + params['congestion_emission_factor']*state['traffic_congestion']
        fleet_mult = 1 - state.get('ev_share', 0)
        return base_emissions * congestion_mult * fleet_mult
    return base_emissions

def update_co2(state, E_t, params):
    """CO2_{t+1} = CO2_t + α*E_t - δ*CO2_t"""
    return state['CO2'] + params['alpha']*E_t - params['delta']*state['CO2']

def update_gdp(state, action, params):
    """GDP growth with policy costs/benefits and climate damage"""
    G_t = state['GDP']
    CO2_t = state['CO2']
    τ, s, c = action
    
    baseline_growth = 1 + params['g_0']
    subsidy_cost = 1 - params.get('beta_subsidy_cost', 0.015) * (s ** 2)
    subsidy_benefit = 1 + params['beta_s']*s
    carbon_tax_cost = 1 - params['beta_tau']*τ
    congestion_charge_cost = 1 - params['beta_c']*c
    climate_damage = 1 - params['theta']*(CO2_t/params['CO2_0'])**params['phi']
    
    mobility_factor = 1.0
    if params.get('use_traffic_model', False) and 'traffic_congestion' in state:
        congestion_cost = params['mobility_elasticity']*state['traffic_congestion']
        mobility_factor = 1 - congestion_cost
    
    return (G_t * baseline_growth * subsidy_benefit * carbon_tax_cost * subsidy_cost *
            congestion_charge_cost * climate_damage * mobility_factor)

def compute_traffic_metrics(state, action, params):
    """Simplified traffic model"""
    τ, s, c = action
    
    base_traffic = params['base_traffic_volume']
    gdp_elasticity = params['gdp_traffic_elasticity']
    traffic_from_gdp = base_traffic * (state['GDP']/params['GDP_baseline'])**gdp_elasticity
    
    tax_reduction = params['eta_tau_traffic']*τ
    subsidy_reduction = params['eta_s_traffic']*s
    charge_reduction = params['eta_c_traffic']*c
    total_reduction = tax_reduction + subsidy_reduction + charge_reduction
    traffic_volume = traffic_from_gdp * (1 - total_reduction)
    
    road_capacity = params['road_capacity']
    v_c_ratio = traffic_volume / road_capacity
    
    alpha_bpr = params['bpr_alpha']
    beta_bpr = params['bpr_beta']
    congestion_multiplier = 1 + alpha_bpr*(v_c_ratio**beta_bpr)
    congestion_level = jnp.clip(v_c_ratio, 0, 1)
    
    free_flow_speed = params['free_flow_speed']
    actual_speed = free_flow_speed / congestion_multiplier
    
    base_ev_share = params['base_ev_share']
    ev_sensitivity = params['ev_tax_sensitivity']
    ev_share = jnp.clip(base_ev_share + ev_sensitivity*τ, 0, 0.95)
    
    return {
        'traffic_volume': traffic_volume,
        'congestion': congestion_level,
        'speed': actual_speed,
        'ev_share': ev_share,
        'congestion_multiplier': congestion_multiplier
    }

def dynamics_step(state, action, params):
    """One timestep of coupled system"""
    if params.get('use_traffic_model', False):
        traffic = compute_traffic_metrics(state, action, params)
        state_with_traffic = {**state, 'traffic_congestion': traffic['congestion'], 
                              'ev_share': traffic['ev_share']}
    else:
        traffic = {}
        state_with_traffic = state
    
    E_t = compute_emissions(state_with_traffic, action, params)
    CO2_next = update_co2(state_with_traffic, E_t, params)
    G_next = update_gdp(state_with_traffic, action, params)
    
    next_state = {'CO2': CO2_next, 'GDP': G_next, 'emissions': E_t, 't': state['t']+1}
    
    if traffic:
        next_state.update({
            'traffic_volume': traffic['traffic_volume'],
            'traffic_congestion': traffic['congestion'],
            'traffic_speed': traffic['speed'],
            'ev_share': traffic['ev_share']
        })
    
    return next_state

def simulate_trajectory(initial_state, policies, params):
    """Simulate full trajectory"""
    trajectory = [initial_state]
    state = initial_state
    for t in range(len(policies)):
        state = dynamics_step(state, policies[t], params)
        trajectory.append(state)
    return trajectory
