"""Policy network"""
import jax
import jax.numpy as jnp

def init_policy_params(key, state_dim=2, action_dim=3, hidden_dim=64):
    """Initialize neural network parameters"""
    keys = jax.random.split(key, 6)
    return {
        'W1': jax.random.normal(keys[0], (state_dim, hidden_dim)) * 0.1,
        'b1': jax.random.normal(keys[1], (hidden_dim,)) * 0.01,
        'W2': jax.random.normal(keys[2], (hidden_dim, hidden_dim)) * 0.1,
        'b2': jax.random.normal(keys[3], (hidden_dim,)) * 0.01,
        'W3': jax.random.normal(keys[4], (hidden_dim, action_dim)) * 0.1,
        'b3': jax.random.normal(keys[5], (action_dim,)) * 0.01,
    }

def normalize_state(state, params):
    """Normalize state to [0,1] range"""
    CO2_norm = (state['CO2'] - params['CO2_0']) / (2*params['CO2_0'])
    GDP_norm = (state['GDP'] - params['G_0']) / (2*params['G_0'])
    return jnp.array([jnp.clip(CO2_norm, -1, 2), jnp.clip(GDP_norm, -1, 2)])

def policy_network(params, state, state_params):
    """Neural network: state → action"""
    x = normalize_state(state, state_params)
    h1 = jnp.tanh(x @ params['W1'] + params['b1'])
    h2 = jnp.tanh(h1 @ params['W2'] + params['b2'])
    action = jax.nn.sigmoid(h2 @ params['W3'] + params['b3'])
    return action

def get_trajectory_policies(policy_params, initial_state, T, state_params):
    """Generate policy sequence for trajectory"""
    policies = []
    state = initial_state
    for t in range(T):
        action = policy_network(policy_params, state, state_params)
        policies.append(action)
        state = {'CO2': state['CO2'] + 0.5, 'GDP': state['GDP']*1.02, 't': t+1}
    return jnp.array(policies)
